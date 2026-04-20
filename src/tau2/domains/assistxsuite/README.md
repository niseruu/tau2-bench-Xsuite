# AssistXSuite Domain Guide

This document explains two things:

1. how the `assistxsuite` domain works end-to-end
2. how to build a new `tau2` domain using the same pattern

## What This Domain Is

`assistxsuite` is a fully local, deterministic mock domain that imitates three
common AssistXSuite or RAGFlow interaction surfaces:

- chat pipeline
- agent pipeline
- direct retrieval tool call

It does **not** call a live service. There is no HTTP client, no auth flow, no
streaming, and no external retrieval stack. Everything runs against a seeded
legal corpus stored in repo data files.

That makes it useful for:

- testing tool-call behavior without API keys
- creating stable benchmark tasks
- showing how to model a domain around message-based RAG flows

## File Layout

Code:

- `src/tau2/domains/assistxsuite/data_model.py`
- `src/tau2/domains/assistxsuite/tools.py`
- `src/tau2/domains/assistxsuite/environment.py`
- `src/tau2/domains/assistxsuite/utils.py`

Data:

- `data/tau2/domains/assistxsuite/db.json`
- `data/tau2/domains/assistxsuite/policy.md`
- `data/tau2/domains/assistxsuite/tasks.json`
- `data/tau2/domains/assistxsuite/split_tasks.json`

Tests:

- `tests/test_domains/test_assistxsuite/test_tools_assistxsuite.py`

## Runtime Flow

### 1. Environment construction

`get_environment()` in `environment.py` is the entrypoint used by the registry.

It:

- loads the domain database from `db.json`
- creates `AssistXSuiteTools`
- reads the agent policy from `policy.md`
- returns a `tau2.environment.environment.Environment`

This is the object the runner and orchestrator interact with.

### 2. Data model

`data_model.py` defines a small database that is enough for the mocked
AssistXSuite workflows:

- `Dataset`
- `LegalDocument`
- `Chunk`
- `ChatAssistant`
- `AgentDefinition`
- `AssistXSuiteDB`

The important point is that this domain stores both configuration and content in
the same DB object:

- dataset membership
- documents and chunks
- chat assistant definitions
- agent definitions

Unlike `banking_knowledge`, there is no separate retrieval pipeline module or
runtime index. The seeded chunks in `db.json` are the retrieval source.

### 3. Policy-driven tool selection

`policy.md` tells the LLM which tool to call:

- `chat_pipeline_completion` for standard legal Q&A
- `agent_pipeline_completion` for review-style answers
- `ragflow_retrieval` for direct clause lookup

The policy also tells the agent to:

- pass message lists
- use the latest `user` message as the question
- ask for references on chat completions when needed
- avoid inventing facts outside retrieved text

### 4. Shared retrieval core

All three tools use the same internal retrieval helpers in `tools.py`.

The retrieval flow is:

1. tokenize the query and chunk text
2. remove basic stopwords
3. resolve allowed datasets
4. optionally narrow by `document_ids`
5. optionally filter documents with `metadata_condition`
6. score every chunk with lexical overlap
7. derive a synthetic `vector_similarity`
8. combine term and vector scores into a final `similarity`
9. sort, paginate, and shape the response

This is intentionally simple. The goal is not to emulate production ranking
quality; it is to emulate the *API surface and tool-call pattern* with stable,
predictable outputs.

### 5. Tool behaviors

#### `chat_pipeline_completion(...)`

This tool:

- validates `chat_id`
- extracts the latest non-empty `user` message
- retrieves chunks from the chat assistant's dataset list
- builds an OpenAI-like non-stream response
- optionally includes `message.reference` when `extra_body.reference=true`

This models `/api/v1/chats_openai/.../chat/completions` in a mocked form.

#### `agent_pipeline_completion(...)`

This tool:

- validates `agent_id`
- extracts the latest non-empty `user` message
- retrieves chunks from the agent's dataset list
- returns a non-stream response with a grounded answer and references
- optionally adds a deterministic mock trace when `return_trace=true`

This models a simplified agent-completion surface without SSE events.

#### `ragflow_retrieval(...)`

This tool:

- runs the same retrieval engine directly
- returns structured retrieval output:
  - `chunks`
  - `pagination`
  - `query_info`

This models the direct retrieval tool-call path rather than a natural-language
answering path.

## Task Design In This Domain

`tasks.json` contains three seed tasks:

- one chat-pipeline task
- one agent-pipeline task
- one retrieval task

Each task defines:

- a `user_scenario`
- one expected tool call in `evaluation_criteria.actions`
- required facts in `communicate_info`
- `reward_basis: ["ACTION", "COMMUNICATE"]`

This means the benchmark checks two things:

- did the agent call the correct tool with the expected arguments?
- did the agent communicate the key legal facts back to the user?

The tasks also include `required_documents` to make the intended knowledge
source explicit.

## Why This Domain Is Simple

This domain deliberately avoids a lot of complexity:

- no mutable business state
- no `user_tools`
- no replay-sensitive write tools
- no embeddings or rerankers
- no live AssistXSuite service integration

That simplicity is useful because it makes the domain an example of how to add a
new benchmark domain with minimal moving parts.

## How To Create A New Domain

Use this as the shortest practical recipe.

### Step 1: decide the domain shape

Pick whether your domain is:

- pure read-only mock data like `assistxsuite`
- transactional CRUD like `mock` or `retail`
- mixed transactional plus retrieval like `banking_knowledge`

This decision drives almost everything else:

- your DB model
- whether you need `user_tools`
- whether tools mutate state
- how tasks should be scored

### Step 2: create the domain package

Add a new folder:

- `src/tau2/domains/<domain_name>/`

At minimum add:

- `__init__.py`
- `data_model.py`
- `tools.py`
- `environment.py`
- `utils.py`

Optional files:

- `user_data_model.py`
- `user_tools.py`
- retrieval helpers or task generators if the domain is more complex

### Step 3: create the data directory

Add:

- `data/tau2/domains/<domain_name>/`

Usually include:

- `db.json` or `db.toml`
- `policy.md`
- `tasks.json`
- `split_tasks.json`

Optional:

- `user_db.json`
- `tasks_voice.json`
- extra prompt or document folders for more advanced domains

### Step 4: define the DB schema

In `data_model.py`, define Pydantic models for the domain state and wrap them in
a `DB` subclass.

For example:

```python
class MyDomainDB(DB):
    users: dict[str, User]
    orders: dict[str, Order]
```

Keep the DB shape aligned with the data file shape. The easiest way to avoid
drift is to model the exact JSON structure you want to store.

### Step 5: define tools

In `tools.py`, create a `ToolKitBase` subclass and decorate callable tools with
`@is_tool(...)`.

Use the tool type intentionally:

- `ToolType.READ` for lookups and retrieval
- `ToolType.WRITE` for mutations
- `ToolType.THINK` for non-observable reasoning helpers
- `ToolType.GENERIC` for miscellaneous helpers

If a tool should not affect replayed state, set `mutates_state=False`.

That matters because `Environment.set_state()` replays mutating tool calls when
reconstructing task state from history.

### Step 6: implement the environment entrypoints

In `environment.py`, add:

- `get_db()`
- `get_environment()`
- `get_tasks()`
- `get_tasks_split()`

These are the functions the registry and runner expect.

### Step 7: write the policy

The policy is the contract between your LLM agent and the domain tools.

It should explain:

- what the tools are for
- when to use each tool
- any required calling conventions
- any domain safety or workflow rules

If the policy is vague, the benchmark behavior becomes noisy.

### Step 8: write tasks

A task usually needs:

- `id`
- `description`
- `user_scenario`
- `evaluation_criteria`

Common evaluation pieces:

- `actions` for expected tool calls
- `communicate_info` for required user-facing facts
- `nl_assertions` when you need freer-form judging
- `env_assertions` for direct environment checks

Choose `reward_basis` carefully. For a read-only RAG task, `ACTION` and
`COMMUNICATE` are often enough. For transactional tasks, `DB` may matter more.

### Step 9: register the domain

Add imports and registration in `src/tau2/registry.py`:

```python
registry.register_domain(my_domain_get_environment, "my_domain")
registry.register_tasks(
    my_domain_get_tasks,
    "my_domain",
    get_task_splits=my_domain_get_tasks_split,
)
```

If you skip this step, the CLI and runner cannot find your domain.

### Step 10: add tests

Create:

- `tests/test_domains/test_<domain_name>/`

At minimum test:

- environment construction
- task loading and split loading
- happy path for every tool
- validation errors for bad arguments
- domain registration

For write-heavy domains, also test state mutation and replay behavior.

## Recommended Authoring Checklist

Before considering a new domain done, verify:

- the DB file loads without validation errors
- every tool has a clear docstring
- `get_tasks("base")` works
- the domain is registered in `registry.py`
- the policy matches the actual tool names and argument shapes
- tests exist for both happy-path and failure-path behavior

## Commands

Run the AssistXSuite domain tests:

```bash
uv run --extra dev python -m pytest tests/test_domains/test_assistxsuite -q
```

Run all tests for a new domain you add:

```bash
pytest tests/test_domains/test_<domain_name>
```

## When To Copy This Pattern

Use `assistxsuite` as your template when:

- the domain is read-only
- the benchmark is mostly about choosing the right tool surface
- deterministic retrieval is more important than realistic ranking quality
- you want a minimal example of a new domain without external dependencies

Use `retail`, `telecom`, or `banking_knowledge` as your reference instead when
you need richer state mutation, user tools, or configurable retrieval pipelines.
