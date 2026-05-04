"""Tools for the PELNI e-Billing domain."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from tau2.domains.pelni_ebill.data_model import KnowledgeChunk, PelniEbillDB
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

FETCH_RESPONSE_TIMEOUT = 10
MAX_RETRIES = 3
RETRY_DELAY = 1
MAX_CHUNK_LENGTH = 1500

STOPWORDS = {
    "a",
    "ada",
    "adalah",
    "apa",
    "bagaimana",
    "berapa",
    "cara",
    "dan",
    "di",
    "dengan",
    "dari",
    "for",
    "how",
    "invoice",
    "itu",
    "ke",
    "of",
    "pada",
    "pelni",
    "the",
    "to",
    "untuk",
    "yang",
}


STATUS_KETERANGAN = {
    "Draft": "Disimpan drafter, belum dikirim untuk diproses",
    "Waiting Approval": "Sedang dicek oleh checker",
    "Approved": "Disetujui approver akhir di user",
    "Checked": "Diverifikasi tim Akuntansi",
    "Validate": "Disetujui VP Akuntansi",
    "Payment Verification": "Disetujui Manager Treasury, siap bayar",
    "Payment Process": "Sedang diproses pembayaran oleh Treasury",
    "Payment Approved": "Pengajuan pembayaran disetujui Dirkeu",
    "Verification": "Dikembalikan ke VP Akuntansi untuk revisi",
    "Canceled": "Invoice/payment dibatalkan",
    "Waiting Release": "Proses input dokumen rilis ke EBS",
    "Release": "Invoice sudah dirilis",
    "Waiting Clearing": "Proses input dokumen clearing ke EBS",
    "Cleared": "Invoice sudah di-clearing",
}


class PelniEbillTools(ToolKitBase):
    """Deterministic mock tools for the PELNI e-Billing chatbot."""

    db: PelniEbillDB

    def __init__(self, db: PelniEbillDB) -> None:
        super().__init__(db)

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [token for token in tokens if token not in STOPWORDS]

    def _score_chunk(self, query_tokens: list[str], chunk: KnowledgeChunk) -> float:
        searchable_text = " ".join(
            [chunk.title, chunk.topic, chunk.content, *chunk.keywords]
        )
        chunk_tokens = self._tokenize(searchable_text)
        if not query_tokens or not chunk_tokens:
            return 0.0

        chunk_counter = Counter(chunk_tokens)
        raw_hits = sum(chunk_counter[token] for token in query_tokens)
        if raw_hits == 0:
            return 0.0

        unique_overlap = len(set(query_tokens) & set(chunk_tokens))
        coverage = unique_overlap / len(set(query_tokens))
        density = min(1.0, raw_hits / max(len(query_tokens), 1))
        return round(min(0.99, 0.45 + (coverage * 0.4) + (density * 0.15)), 3)

    def _truncate_chunk(self, content: str) -> str:
        if len(content) <= MAX_CHUNK_LENGTH:
            return content

        truncated = content[:MAX_CHUNK_LENGTH]
        last_period = truncated.rfind(".")
        last_newline = truncated.rfind("\n")
        cut_point = max(last_period, last_newline)
        if cut_point > MAX_CHUNK_LENGTH * 0.7:
            return truncated[: cut_point + 1] + "..."
        return truncated + "..."

    def get_status_keterangan(self, status: str) -> str:
        """Return the production chatbot's short Indonesian status explanation."""

        return STATUS_KETERANGAN.get(status, "Status tidak dikenali")

    def format_nominal(self, nominal: Any) -> str:
        """Format a value as Indonesian Rupiah, matching the production helper."""

        if nominal is None or nominal == "":
            return "Tidak tersedia"

        try:
            value = float(nominal)
        except (TypeError, ValueError):
            return "Tidak tersedia"
        return f"Rp {value:,.0f}".replace(",", ".")

    def format_location(self, location: Any) -> str:
        """Format an uppercase operating unit/location as title case."""

        if location is None or location == "":
            return "Tidak tersedia"

        try:
            return str(location).title()
        except (TypeError, ValueError):
            return "Tidak tersedia"

    def render_invoice_info(self, invoice_data: dict[str, Any]) -> str:
        """Render one invoice in the production chatbot's text format."""

        return f"""
    - Nomor Invoice: {invoice_data.get("invoice_number", "Tidak tersedia")}
    - Divisi: {invoice_data.get("division_name", "Tidak tersedia")}
    - Operating Unit/OU: {self.format_location(invoice_data.get("location"))}
    - Deskripsi: {invoice_data.get("description", "Tidak tersedia")}
    - Supplier: {invoice_data.get("supplier_name", "Tidak tersedia")}
    - Grand Total: {self.format_nominal(invoice_data.get("grand_total"))}
    - VAT Total: {self.format_nominal(invoice_data.get("vat_total"))}
    - Status: {invoice_data.get("status", "Tidak tersedia")}
    - Keterangan: {invoice_data.get("keterangan_singkat", "Tidak tersedia")}
    - Tanggal Terakhir Diperbarui: {invoice_data.get("updated_date", "Tidak tersedia")}
    """

    @is_tool(ToolType.READ, mutates_state=False)
    def search_knowledge_base(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
    ) -> str:
        """Mencari informasi di knowledge base PELNI menggunakan AssistX Suite API.

        Gunakan tool ini untuk mencari informasi tentang layanan PELNI seperti
        jadwal kapal, harga tiket, rute pelayaran, kebijakan penumpang,
        prosedur e-billing, pedoman akuntansi keuangan PELNI, tarif PPN/PPH,
        dan informasi terkait lainnya.

        PENTING: Tool ini mengembalikan maksimal 5 chunk dengan panjang
        terbatas. Anda WAJIB merangkum hasil dan memilih hanya informasi yang
        PALING RELEVAN untuk dijawabkan ke user. Jangan tampilkan skor
        similarity kepada user meskipun tool mengembalikannya.

        Contoh penggunaan:
        - search_knowledge_base(query="Cara input invoice di pusat?")
        - search_knowledge_base(query="keterangan status Approved", top_k=3)
        - search_knowledge_base(query="berapa pph pasal 23?", top_k=1)

        Args:
            query: Pertanyaan atau kata kunci pencarian. Kirim query apa adanya
                dari user, jangan diubah atau diparafrase.
            top_k: Jumlah hasil maksimal yang dikembalikan, antara 1 sampai 20.
                Default 5. Untuk "berapa pph pasal 23?", gunakan top_k=1.
            similarity_threshold: Ambang batas similarity score antara 0.0
                sampai 1.0. Default 0.5.

        Returns:
            Konten dokumen yang paling relevan dengan query atau pesan error.
            Setiap chunk dibatasi maksimal 1500 karakter dan dipisahkan dengan
            `---`, mengikuti output tool produksi.
        """

        if not query or query.strip() == "":
            return "Error: Query tidak boleh kosong"

        if top_k < 1 or top_k > 20:
            return "Error: top_k harus antara 1 dan 20"

        if similarity_threshold < 0.0 or similarity_threshold > 1.0:
            return "Error: similarity_threshold harus antara 0.0 dan 1.0"

        query_tokens = self._tokenize(query)
        ranked_chunks = []
        for chunk in self.db.knowledge_chunks:
            similarity = self._score_chunk(query_tokens, chunk)
            if similarity >= similarity_threshold:
                ranked_chunks.append((similarity, chunk))

        ranked_chunks.sort(key=lambda item: (-item[0], item[1].id))
        top_chunks = ranked_chunks[:top_k]

        if not top_chunks:
            return "Tidak ada informasi yang ditemukan untuk query tersebut."

        formatted_results = []
        for similarity, chunk in top_chunks:
            content = self._truncate_chunk(chunk.content.strip())
            formatted_results.append(f"{content}\n(Similarity: {similarity:.3f})")

        return "\n\n---\n\n".join(formatted_results)

    @is_tool(ToolType.READ, mutates_state=False)
    def get_invoice_progress(self, search_term: str) -> str:
        """Mencari progress invoice dari database PELNI API berdasarkan kata kunci.

        PENTING: Gunakan tool ini setiap kali user menyebutkan nomor invoice,
        nomor tagihan, status invoice, progres invoice, atau kata kunci apapun
        terkait invoice.

        CARA PENGGUNAAN:
        - Ekstrak kata kunci/nomor yang disebutkan user.
        - Isi parameter search_term dengan PERSIS kata kunci tersebut.
        - Tool otomatis menentukan field pencarian:
          * Jika ANGKA MURNI 8 DIGIT, contoh "24120003", cari di invoice_num.
          * Jika ANGKA LEBIH/KURANG DARI 8 DIGIT, contoh "2341123", cari di
            vendor_bill_num dan description.
          * Jika TEKS/ALFANUMERIK, contoh "198/INV/IX/2024" atau "training",
            cari di vendor_bill_num dan description.

        Contoh penggunaan:
        - User: "Cek invoice 24120003"
          -> get_invoice_progress(search_term="24120003")
        - User: "Tolong cek tagihan 198/INV/IX/2024"
          -> get_invoice_progress(search_term="198/INV/IX/2024")
        - User: "Ada invoice tentang training?"
          -> get_invoice_progress(search_term="training")

        Args:
            search_term: Kata kunci atau nomor yang disebutkan user untuk
                mencari invoice. Gunakan nilai persis dari user, misalnya
                "24120003", "198/INV/IX/2024", atau "training".

        Returns:
            Informasi progress invoice dalam format teks produksi, maksimal 3
            invoice terbaru, atau pesan error/tidak ditemukan.
        """

        context = self.db.session_context
        normalized_search = search_term.strip()
        if normalized_search.isdigit() and len(normalized_search) == 8:
            matches = [
                invoice
                for invoice in self.db.invoices
                if invoice.invoice_number == normalized_search
            ]
        else:
            lowered_search = normalized_search.lower()
            matches = [
                invoice
                for invoice in self.db.invoices
                if lowered_search in invoice.vendor_bill_num.lower()
                or lowered_search in invoice.description.lower()
            ]

        if context.is_all == "false":
            matches = [
                invoice
                for invoice in matches
                if invoice.division_code == context.segment_divisi
            ]

        if not matches:
            return (
                f'Invoice tidak ditemukan di sistem PELNI untuk kata kunci "{search_term}". '
                "Silakan cek kembali atau gunakan kata kunci lain."
            )

        matches.sort(key=lambda invoice: invoice.updated_date, reverse=True)
        total_found = len(matches)
        selected = matches[:3]

        rendered_invoices = []
        for invoice in selected:
            invoice_payload = invoice.model_dump()
            invoice_payload["keterangan_singkat"] = self.get_status_keterangan(
                invoice.status
            )
            rendered_invoices.append(self.render_invoice_info(invoice_payload))

        if len(selected) == 1:
            return rendered_invoices[0]

        result = f"Ditemukan {total_found} invoice"
        if total_found > 3:
            result += " (menampilkan 3 terbaru)"
        result += ":\n"
        for index, rendered in enumerate(rendered_invoices, 1):
            result += f"\n{'=' * 50}\nInvoice #{index}:\n"
            result += rendered
        return result
