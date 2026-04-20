"""Tools for the AssistXSuite domain."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from tau2.domains.assistxsuite.data_model import (
    AgentDefinition,
    AssistXSuiteDB,
    ChatAssistant,
    Chunk,
    LegalDocument,
)
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "before",
    "by",
    "for",
    "from",
    "how",
    "if",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "the",
    "their",
    "to",
    "us",
    "what",
    "which",
    "with",
}


class AssistXSuiteTools(ToolKitBase):
    """Mock tools that emulate AssistXSuite legal-RAG surfaces."""

    db: AssistXSuiteDB

    def __init__(self, db: AssistXSuiteDB) -> None:
        super().__init__(db)

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return [token for token in tokens if token not in STOPWORDS]

    def _get_latest_user_message(self, messages: list[dict[str, Any]]) -> str:
        if not messages:
            raise ValueError("At least one message is required.")

        for message in reversed(messages):
            role = str(message.get("role", "")).strip().lower()
            content = message.get("content")
            if role == "user" and isinstance(content, str) and content.strip():
                return content.strip()

        raise ValueError("A non-empty user message is required.")

    def _get_chat_assistant(self, chat_id: str) -> ChatAssistant:
        chat = self.db.chat_assistants.get(chat_id)
        if chat is None:
            raise ValueError(f"Unknown chat assistant: {chat_id}")
        return chat

    def _get_agent_definition(self, agent_id: str) -> AgentDefinition:
        agent = self.db.agents.get(agent_id)
        if agent is None:
            raise ValueError(f"Unknown agent: {agent_id}")
        return agent

    def _resolve_dataset_ids(self, dataset_ids: Iterable[str] | None) -> list[str]:
        if dataset_ids:
            resolved = list(dict.fromkeys(dataset_ids))
        else:
            resolved = list(self.db.datasets.keys())

        unknown = [dataset_id for dataset_id in resolved if dataset_id not in self.db.datasets]
        if unknown:
            raise ValueError(f"Unknown dataset ids: {unknown}")
        return resolved

    def _matches_metadata_condition(
        self,
        document: LegalDocument,
        metadata_condition: dict[str, Any] | None,
    ) -> bool:
        if not metadata_condition:
            return True

        conditions = metadata_condition.get("conditions") or []
        if not conditions:
            return True

        logic = str(metadata_condition.get("logic", "and")).lower()
        results = [
            self._evaluate_metadata_condition(document.metadata, condition)
            for condition in conditions
        ]
        if logic == "or":
            return any(results)
        return all(results)

    def _evaluate_metadata_condition(
        self,
        metadata: dict[str, Any],
        condition: dict[str, Any],
    ) -> bool:
        name = str(condition.get("name", "")).strip()
        operator = str(condition.get("comparison_operator", "is")).strip().lower()
        value = condition.get("value")
        current = metadata.get(name)

        if operator in {"empty", "is empty"}:
            return current in (None, "", [], {})
        if operator in {"not empty", "is not empty"}:
            return current not in (None, "", [], {})
        if current is None:
            return False

        current_str = str(current).lower()
        value_str = "" if value is None else str(value).lower()

        if operator in {"is", "=", "=="}:
            return current_str == value_str
        if operator in {"not is", "!=", "<>"}:
            return current_str != value_str
        if operator == "contains":
            return value_str in current_str
        if operator == "not contains":
            return value_str not in current_str
        if operator == "start with":
            return current_str.startswith(value_str)
        if operator == "end with":
            return current_str.endswith(value_str)

        if operator in {">", "<", ">=", "≥", "<=", "≤"}:
            try:
                current_num = float(current)
                value_num = float(value)
            except (TypeError, ValueError):
                return False
            if operator == ">":
                return current_num > value_num
            if operator == "<":
                return current_num < value_num
            if operator in {">=", "≥"}:
                return current_num >= value_num
            return current_num <= value_num

        return False

    def _iter_candidate_documents(
        self,
        dataset_ids: list[str],
        document_ids: list[str] | None,
        metadata_condition: dict[str, Any] | None,
    ) -> list[LegalDocument]:
        allowed_document_ids: set[str] = set()
        for dataset_id in dataset_ids:
            allowed_document_ids.update(self.db.datasets[dataset_id].document_ids)

        if document_ids:
            requested_document_ids = set(document_ids)
            unknown = requested_document_ids - set(self.db.documents.keys())
            if unknown:
                raise ValueError(f"Unknown document ids: {sorted(unknown)}")
            allowed_document_ids &= requested_document_ids

        documents = [
            self.db.documents[doc_id]
            for doc_id in sorted(allowed_document_ids)
            if self._matches_metadata_condition(
                self.db.documents[doc_id], metadata_condition
            )
        ]
        return documents

    def _score_chunk(
        self,
        query_tokens: list[str],
        chunk: Chunk,
        document: LegalDocument,
        vector_similarity_weight: float,
    ) -> dict[str, Any] | None:
        chunk_tokens = self._tokenize(chunk.content)
        if not query_tokens or not chunk_tokens:
            return None

        chunk_counter = Counter(chunk_tokens)
        raw_hits = sum(chunk_counter[token] for token in query_tokens)
        if raw_hits == 0:
            return None

        unique_overlap = len(set(query_tokens) & set(chunk_tokens))
        term_similarity = unique_overlap / len(set(query_tokens))
        vector_similarity = min(1.0, term_similarity + 0.1)
        similarity = ((1 - vector_similarity_weight) * term_similarity) + (
            vector_similarity_weight * vector_similarity
        )

        return {
            "id": chunk.id,
            "content": chunk.content,
            "document_id": document.id,
            "document_name": document.title,
            "dataset_id": document.dataset_id,
            "image_id": "",
            "url": document.metadata.get("url"),
            "similarity": round(similarity, 4),
            "vector_similarity": round(vector_similarity, 4),
            "term_similarity": round(term_similarity, 4),
            "doc_type": document.doc_type,
            "positions": chunk.positions or [[1, 1, 1, 1, 1]],
            "document_metadata": document.metadata,
            "_raw_hits": raw_hits,
        }

    def _retrieve_chunks(
        self,
        *,
        question: str,
        dataset_ids: list[str] | None,
        document_ids: list[str] | None,
        page: int,
        page_size: int,
        similarity_threshold: float,
        vector_similarity_weight: float,
        keyword: bool,
        top_k: int,
        metadata_condition: dict[str, Any] | None,
    ) -> dict[str, Any]:
        query = question.strip()
        if not query:
            raise ValueError("A non-empty question is required.")
        if page < 1:
            raise ValueError("page must be >= 1")
        if page_size < 1:
            raise ValueError("page_size must be >= 1")

        resolved_dataset_ids = self._resolve_dataset_ids(dataset_ids)
        documents = self._iter_candidate_documents(
            resolved_dataset_ids,
            document_ids,
            metadata_condition,
        )
        query_tokens = self._tokenize(query)

        ranked_chunks = []
        for document in documents:
            for chunk in document.chunks:
                scored = self._score_chunk(
                    query_tokens,
                    chunk,
                    document,
                    vector_similarity_weight,
                )
                if scored is None:
                    continue
                if scored["similarity"] < similarity_threshold:
                    continue
                ranked_chunks.append(scored)

        ranked_chunks.sort(
            key=lambda item: (
                -item["similarity"],
                -item["_raw_hits"],
                item["document_name"],
                item["id"],
            )
        )
        ranked_chunks = ranked_chunks[:top_k]

        total = len(ranked_chunks)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paged_chunks = ranked_chunks[start_idx:end_idx]

        clean_chunks = []
        doc_counts: Counter[tuple[str, str]] = Counter()
        for chunk in paged_chunks:
            chunk_payload = {k: v for k, v in chunk.items() if not k.startswith("_")}
            clean_chunks.append(chunk_payload)
            doc_counts[(chunk_payload["document_id"], chunk_payload["document_name"])] += 1

        reference_chunks = {
            str(index): chunk for index, chunk in enumerate(clean_chunks, start=1)
        }
        doc_aggs = {
            doc_name: {
                "doc_name": doc_name,
                "doc_id": doc_id,
                "count": count,
            }
            for (doc_id, doc_name), count in doc_counts.items()
        }

        return {
            "chunks": clean_chunks,
            "total": total,
            "page": page,
            "page_size": page_size,
            "reference": {
                "chunks": reference_chunks,
                "doc_aggs": doc_aggs,
            },
            "query_info": {
                "question": query,
                "similarity_threshold": similarity_threshold,
                "vector_weight": vector_similarity_weight,
                "keyword_search": keyword,
                "dataset_count": len(resolved_dataset_ids),
            },
        }

    def _compose_grounded_answer(
        self,
        *,
        question: str,
        chunks: list[dict[str, Any]],
        prefix: str,
    ) -> str:
        if not chunks:
            return (
                f"{prefix} I could not find supporting language in the mock legal "
                "corpus for that request."
            )

        primary_document_id = chunks[0]["document_id"]
        primary_chunks = [
            chunk for chunk in chunks if chunk["document_id"] == primary_document_id
        ]
        selected_chunks = primary_chunks[:2] or chunks[:2]
        snippets = []
        for chunk in selected_chunks:
            snippets.append(f"{chunk['document_name']}: {chunk['content']}")
        sources = ", ".join(
            dict.fromkeys(chunk["document_name"] for chunk in selected_chunks)
        )
        return f"{prefix} {' '.join(snippets)} Sources: {sources}."

    @is_tool(ToolType.READ, mutates_state=False)
    def chat_pipeline_completion(
        self,
        chat_id: str,
        messages: list[dict[str, Any]],
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a mock OpenAI-style chat completion against the legal corpus."""

        chat = self._get_chat_assistant(chat_id)
        question = self._get_latest_user_message(messages)
        extra_body = extra_body or {}
        retrieval = self._retrieve_chunks(
            question=question,
            dataset_ids=chat.dataset_ids,
            document_ids=None,
            page=1,
            page_size=10,
            similarity_threshold=0.2,
            vector_similarity_weight=0.3,
            keyword=False,
            top_k=10,
            metadata_condition=extra_body.get("metadata_condition"),
        )
        answer = self._compose_grounded_answer(
            question=question,
            chunks=retrieval["chunks"],
            prefix="Grounded legal answer:",
        )

        message = {
            "role": "assistant",
            "content": answer,
        }
        if extra_body.get("reference"):
            message["reference"] = retrieval["reference"]

        return {
            "id": f"chatcmpl-{chat_id}",
            "object": "chat.completion",
            "created": 1_765_000_000,
            "model": "assistxsuite-mock-chat",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "logprobs": None,
                    "message": message,
                }
            ],
            "usage": {
                "prompt_tokens": len(self._tokenize(question)),
                "completion_tokens": len(self._tokenize(answer)),
                "total_tokens": len(self._tokenize(question))
                + len(self._tokenize(answer)),
            },
        }

    @is_tool(ToolType.READ, mutates_state=False)
    def agent_pipeline_completion(
        self,
        agent_id: str,
        messages: list[dict[str, Any]],
        session_id: str | None = None,
        inputs: dict[str, Any] | None = None,
        return_trace: bool = False,
    ) -> dict[str, Any]:
        """Run a mock agent completion using message-based input."""

        agent = self._get_agent_definition(agent_id)
        question = self._get_latest_user_message(messages)
        retrieval = self._retrieve_chunks(
            question=question,
            dataset_ids=agent.dataset_ids,
            document_ids=None,
            page=1,
            page_size=10,
            similarity_threshold=0.2,
            vector_similarity_weight=0.3,
            keyword=False,
            top_k=10,
            metadata_condition=None,
        )
        answer = self._compose_grounded_answer(
            question=question,
            chunks=retrieval["chunks"],
            prefix="Contract review summary:",
        )

        payload = {
            "id": f"agentcmpl-{agent_id}",
            "object": "chat.completion",
            "created": 1_765_000_100,
            "model": "assistxsuite-mock-agent",
            "session_id": session_id or f"mock-session-{agent_id}",
            "inputs": inputs or {},
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "logprobs": None,
                    "message": {
                        "role": "assistant",
                        "content": answer,
                        "reference": retrieval["reference"],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": len(self._tokenize(question)),
                "completion_tokens": len(self._tokenize(answer)),
                "total_tokens": len(self._tokenize(question))
                + len(self._tokenize(answer)),
            },
        }

        if return_trace:
            payload["trace"] = [
                {
                    "node": "begin",
                    "status": "finished",
                    "inputs": {"question": question, "session_id": payload["session_id"]},
                    "outputs": {"validated": True},
                    "elapsed_time": 0.0,
                },
                {
                    "node": "retrieval",
                    "status": "finished",
                    "inputs": {"dataset_ids": agent.dataset_ids},
                    "outputs": {
                        "documents": [
                            chunk["document_id"] for chunk in retrieval["chunks"]
                        ]
                    },
                    "elapsed_time": 0.0,
                },
                {
                    "node": "message",
                    "status": "finished",
                    "inputs": {"return_trace": True},
                    "outputs": {"answer": answer},
                    "elapsed_time": 0.0,
                },
            ]

        return payload

    @is_tool(ToolType.READ, mutates_state=False)
    def ragflow_retrieval(
        self,
        question: str,
        dataset_ids: list[str],
        document_ids: list[str] | None = None,
        page: int = 1,
        page_size: int = 10,
        similarity_threshold: float = 0.2,
        vector_similarity_weight: float = 0.3,
        top_k: int = 1024,
        keyword: bool = False,
        metadata_condition: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a mock MCP-style retrieval call over the legal corpus."""

        retrieval = self._retrieve_chunks(
            question=question,
            dataset_ids=dataset_ids,
            document_ids=document_ids,
            page=page,
            page_size=page_size,
            similarity_threshold=similarity_threshold,
            vector_similarity_weight=vector_similarity_weight,
            keyword=keyword,
            top_k=top_k,
            metadata_condition=metadata_condition,
        )
        return {
            "chunks": retrieval["chunks"],
            "pagination": {
                "page": retrieval["page"],
                "page_size": retrieval["page_size"],
                "total_chunks": retrieval["total"],
                "total_pages": (
                    (retrieval["total"] + retrieval["page_size"] - 1)
                    // retrieval["page_size"]
                ),
            },
            "query_info": retrieval["query_info"],
        }
