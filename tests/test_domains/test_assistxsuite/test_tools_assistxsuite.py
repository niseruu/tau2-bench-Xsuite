import pytest

from tau2.domains.assistxsuite.data_model import AssistXSuiteDB
from tau2.domains.assistxsuite.environment import (
    get_db,
    get_environment,
    get_tasks,
    get_tasks_split,
)
from tau2.registry import registry


@pytest.fixture
def assistxsuite_db() -> AssistXSuiteDB:
    return get_db()


@pytest.fixture
def environment(assistxsuite_db: AssistXSuiteDB):
    return get_environment(assistxsuite_db)


def test_chat_pipeline_completion_happy_path(environment):
    payload = environment.tools.chat_pipeline_completion(
        chat_id="chat_legal_assistant",
        messages=[
            {
                "role": "user",
                "content": (
                    "How quickly must the vendor notify us after a confirmed "
                    "breach, and what security controls are required?"
                ),
            }
        ],
        extra_body={"reference": True},
    )

    message = payload["choices"][0]["message"]
    assert "72 hours" in message["content"]
    assert "encryption at rest" in message["content"]
    assert "reference" in message
    assert "doc_dpa_security_001" in {
        chunk["document_id"]
        for chunk in message["reference"]["chunks"].values()
    }


def test_chat_pipeline_completion_without_reference(environment):
    payload = environment.tools.chat_pipeline_completion(
        chat_id="chat_legal_assistant",
        messages=[{"role": "user", "content": "Summarize the breach notice clause."}],
    )

    message = payload["choices"][0]["message"]
    assert "reference" not in message


def test_agent_pipeline_completion_happy_path(environment):
    payload = environment.tools.agent_pipeline_completion(
        agent_id="agent_contract_reviewer",
        messages=[
            {
                "role": "user",
                "content": (
                    "Can the vendor add subprocessors silently and how much "
                    "notice is required before a material change?"
                ),
            }
        ],
    )

    message = payload["choices"][0]["message"]
    assert "prior written notice" in message["content"]
    assert "10 business days" in message["content"]
    assert "trace" not in payload


def test_agent_pipeline_completion_trace_only_when_requested(environment):
    without_trace = environment.tools.agent_pipeline_completion(
        agent_id="agent_contract_reviewer",
        messages=[{"role": "user", "content": "Review subprocessor notice terms."}],
        return_trace=False,
    )
    with_trace = environment.tools.agent_pipeline_completion(
        agent_id="agent_contract_reviewer",
        messages=[{"role": "user", "content": "Review subprocessor notice terms."}],
        return_trace=True,
    )

    assert "trace" not in without_trace
    assert [node["node"] for node in with_trace["trace"]] == [
        "begin",
        "retrieval",
        "message",
    ]


@pytest.mark.parametrize(
    ("tool_name", "kwargs", "expected_error"),
    [
        (
            "chat_pipeline_completion",
            {"chat_id": "missing_chat", "messages": [{"role": "user", "content": "Hi"}]},
            "Unknown chat assistant",
        ),
        (
            "agent_pipeline_completion",
            {"agent_id": "missing_agent", "messages": [{"role": "user", "content": "Hi"}]},
            "Unknown agent",
        ),
        (
            "chat_pipeline_completion",
            {"chat_id": "chat_legal_assistant", "messages": []},
            "At least one message is required.",
        ),
        (
            "agent_pipeline_completion",
            {
                "agent_id": "agent_contract_reviewer",
                "messages": [{"role": "assistant", "content": "No user turn"}],
            },
            "A non-empty user message is required.",
        ),
    ],
)
def test_pipeline_validation_errors(environment, tool_name, kwargs, expected_error):
    tool = getattr(environment.tools, tool_name)
    with pytest.raises(ValueError, match=expected_error):
        tool(**kwargs)


def test_ragflow_retrieval_happy_path(environment):
    payload = environment.tools.ragflow_retrieval(
        question="Find the auto-renewal clause and the notice required to opt out.",
        dataset_ids=["dataset_legal_core"],
    )

    assert payload["pagination"]["total_chunks"] >= 1
    assert payload["query_info"]["dataset_count"] == 1
    assert any(
        chunk["document_id"] == "doc_renewal_termination_001"
        for chunk in payload["chunks"]
    )


def test_ragflow_retrieval_document_filter(environment):
    payload = environment.tools.ragflow_retrieval(
        question="What is the breach notice timing?",
        dataset_ids=["dataset_legal_core"],
        document_ids=["doc_dpa_security_001"],
    )

    assert payload["chunks"]
    assert all(
        chunk["document_id"] == "doc_dpa_security_001"
        for chunk in payload["chunks"]
    )


def test_ragflow_retrieval_metadata_filter(environment):
    matching = environment.tools.ragflow_retrieval(
        question="Find the renewal clause and the opt-out notice.",
        dataset_ids=["dataset_legal_core"],
        metadata_condition={
            "logic": "and",
            "conditions": [
                {
                    "name": "category",
                    "comparison_operator": "is",
                    "value": "commercial-terms",
                }
            ],
        },
    )
    filtered_out = environment.tools.ragflow_retrieval(
        question="Find the renewal clause and the opt-out notice.",
        dataset_ids=["dataset_legal_core"],
        metadata_condition={
            "logic": "and",
            "conditions": [
                {
                    "name": "category",
                    "comparison_operator": "is",
                    "value": "security",
                }
            ],
        },
    )

    assert matching["chunks"]
    assert filtered_out["chunks"] == []


def test_get_environment_and_task_wiring():
    environment = get_environment()
    tasks = get_tasks()
    task_splits = get_tasks_split()

    assert environment.get_domain_name() == "assistxsuite"
    assert len(tasks) == 3
    assert task_splits["base"] == [
        "assistxsuite_chat_001",
        "assistxsuite_agent_001",
        "assistxsuite_retrieval_001",
    ]


def test_registry_exposes_assistxsuite_domain_and_tasks():
    assert "assistxsuite" in registry.get_domains()
    assert "assistxsuite" in registry.get_task_sets()
    env_constructor = registry.get_env_constructor("assistxsuite")
    tasks_loader = registry.get_tasks_loader("assistxsuite")

    assert env_constructor().get_domain_name() == "assistxsuite"
    assert len(tasks_loader("base")) == 3

