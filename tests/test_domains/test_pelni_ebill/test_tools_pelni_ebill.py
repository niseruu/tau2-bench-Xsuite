import pytest

from tau2.domains.pelni_ebill.data_model import PelniEbillDB
from tau2.domains.pelni_ebill.environment import (
    get_environment,
    get_tasks,
    get_tasks_split,
)
from tau2.domains.pelni_ebill.tools import PelniEbillTools
from tau2.registry import registry


@pytest.fixture
def pelni_ebill_db() -> PelniEbillDB:
    return PelniEbillDB.load("data/tau2/domains/pelni_ebill/db.json")


@pytest.fixture
def tools(pelni_ebill_db: PelniEbillDB) -> PelniEbillTools:
    return PelniEbillTools(pelni_ebill_db)


def test_invoice_progress_eight_digit_search(tools: PelniEbillTools):
    result = tools.get_invoice_progress("24120003")

    assert "Nomor Invoice: 24120003" in result
    assert "PT Sarel Sentra Inspira" in result
    assert "Rp 20.400.000" in result
    assert "Operating Unit/OU: Jakarta Pusat" in result
    assert "Status: Cleared" in result
    assert "Keterangan: Invoice sudah di-clearing" in result


def test_invoice_progress_vendor_bill_search(tools: PelniEbillTools):
    result = tools.get_invoice_progress("198/INV/IX/2024")

    assert "Nomor Invoice: 24120004" in result
    assert "PT Konsultan Nusantara" in result
    assert "Rp 55.000.000" in result
    assert "Status: Payment Process" in result
    assert "Keterangan: Sedang diproses pembayaran oleh Treasury" in result


@pytest.mark.parametrize(
    ("search_term", "invoice_number", "expected_status"),
    [
        ("2341123", "24120009", "Validate"),
        ("241200035", "24120010", "Payment Verification"),
        ("SQIVA.02/11/2023", "23110011", "Checked"),
        ("27 Nov 2025/002", "25110002", "Waiting Release"),
        ("jul", "24070001", "Payment Approved"),
    ],
)
def test_invoice_progress_production_prompt_examples(
    tools: PelniEbillTools,
    search_term: str,
    invoice_number: str,
    expected_status: str,
):
    result = tools.get_invoice_progress(search_term)

    assert f"Nomor Invoice: {invoice_number}" in result
    assert f"Status: {expected_status}" in result


def test_invoice_progress_keyword_filters_segment_and_limits_results(
    tools: PelniEbillTools,
):
    result = tools.get_invoice_progress("training")

    assert "Ditemukan 4 invoice (menampilkan 3 terbaru)" in result
    assert "Nomor Invoice: 24120005" in result
    assert "Nomor Invoice: 24120003" in result
    assert "Nomor Invoice: 24120007" in result
    assert "Nomor Invoice: 24120006" not in result
    assert "Nomor Invoice: 24120008" not in result
    assert result.index("24120005") < result.index("24120003")
    assert result.index("24120003") < result.index("24120007")


def test_invoice_progress_admin_context_can_see_other_divisions(
    pelni_ebill_db: PelniEbillDB,
):
    pelni_ebill_db.session_context.is_all = "true"
    result = PelniEbillTools(pelni_ebill_db).get_invoice_progress("training")

    assert "Ditemukan 5 invoice (menampilkan 3 terbaru)" in result
    assert "Nomor Invoice: 24120006" in result
    assert result.index("24120006") < result.index("24120005")


def test_knowledge_base_validation_errors(tools: PelniEbillTools):
    assert tools.search_knowledge_base("") == "Error: Query tidak boleh kosong"
    assert (
        tools.search_knowledge_base("invoice", top_k=0)
        == "Error: top_k harus antara 1 dan 20"
    )
    assert (
        tools.search_knowledge_base("invoice", top_k=21)
        == "Error: top_k harus antara 1 dan 20"
    )
    assert (
        tools.search_knowledge_base("invoice", similarity_threshold=-0.1)
        == "Error: similarity_threshold harus antara 0.0 dan 1.0"
    )
    assert (
        tools.search_knowledge_base("invoice", similarity_threshold=1.1)
        == "Error: similarity_threshold harus antara 0.0 dan 1.0"
    )


def test_search_knowledge_base_retrieves_deterministic_chunks(
    tools: PelniEbillTools,
):
    result = tools.search_knowledge_base("Cara input invoice di pusat?", top_k=1)

    assert "Input Invoice" in result
    assert "supplier" in result
    assert "Draft" in result
    assert "(Similarity:" in result


def test_search_knowledge_base_tax_top_k(tools: PelniEbillTools):
    result = tools.search_knowledge_base("berapa pph pasal 23?", top_k=1)

    assert "PPh 23" in result
    assert "2%" in result
    assert "badan usaha dalam negeri" in result
    assert "---" not in result


@pytest.mark.parametrize(
    ("query", "expected_text"),
    [
        ("pph 21 rate berapa?", "Layer 1"),
        ("tarif pph 22 pembelian barang di atas 10jt", "PPh 22"),
        ("pph 4(2) sewa bangunan", "PPh Final Pasal 4 ayat 2"),
        ("PPh untuk pekerjaan konstruksi yang memiliki sertifikat", "sertifikat"),
        ("tarif ppn", "Tarif PPN adalah 12%"),
    ],
)
def test_search_knowledge_base_covers_production_tax_topics(
    tools: PelniEbillTools,
    query: str,
    expected_text: str,
):
    result = tools.search_knowledge_base(query, top_k=1)

    assert expected_text in result
    assert "(Similarity:" in result


def test_tool_schemas_include_production_guidance(pelni_ebill_db: PelniEbillDB):
    schemas = {
        tool.name: tool.openai_schema
        for tool in get_environment(pelni_ebill_db).get_tools()
    }

    kb_schema = schemas["search_knowledge_base"]["function"]
    invoice_schema = schemas["get_invoice_progress"]["function"]

    assert "knowledge base PELNI" in kb_schema["description"]
    assert "AssistX Suite API" in kb_schema["description"]
    assert "similarity" in kb_schema["description"]
    assert "top_k" in kb_schema["parameters"]["properties"]
    assert (
        'Untuk "berapa pph pasal 23?", gunakan top_k=1'
        in kb_schema["parameters"]["properties"]["top_k"]["description"]
    )
    assert "progress invoice" in invoice_schema["description"]
    assert "ANGKA MURNI 8 DIGIT" in invoice_schema["description"]
    assert "vendor_bill_num" in invoice_schema["description"]
    assert (
        "Gunakan nilai persis dari user"
        in invoice_schema["parameters"]["properties"]["search_term"]["description"]
    )


def test_get_environment_and_task_wiring():
    environment = get_environment()
    tasks = get_tasks()
    task_splits = get_tasks_split()

    assert environment.get_domain_name() == "pelni_ebill"
    assert {tool.name for tool in environment.get_tools()} == {
        "search_knowledge_base",
        "get_invoice_progress",
    }
    assert len(tasks) == 7
    assert task_splits["base"] == [
        "pelni_ebill_invoice_num_001",
        "pelni_ebill_vendor_bill_001",
        "pelni_ebill_invoice_keyword_001",
        "pelni_ebill_tutorial_clarify_001",
        "pelni_ebill_tutorial_kb_001",
        "pelni_ebill_tax_tariff_001",
        "pelni_ebill_refusal_001",
    ]


def test_registry_exposes_pelni_ebill_domain_and_tasks():
    assert "pelni_ebill" in registry.get_domains()
    assert "pelni_ebill" in registry.get_task_sets()

    env_constructor = registry.get_env_constructor("pelni_ebill")
    tasks_loader = registry.get_tasks_loader("pelni_ebill")

    assert env_constructor().get_domain_name() == "pelni_ebill"
    assert len(tasks_loader()) == 7
