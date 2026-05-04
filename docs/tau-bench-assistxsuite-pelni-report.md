# Tau-bench Evaluation for AssistX Suite / PELNI e-Billing

Yes, Tau-bench can be used to evaluate AssistX Suite style agent workflows, as
long as we model the workflow as a Tau domain: policy, tools, mock data, user
scenarios, and scoring criteria. It is best used as a regression harness for
agentic behavior: did the agent pick the right tool, pass the right arguments,
follow policy, complete the task, communicate the required facts, and refuse or
clarify when needed.

For the first prototype, we added a generic `assistxsuite` domain to prove that
AssistX/RAGFlow-like chat, agent, and retrieval surfaces can be evaluated inside
Tau-bench. After the follow-up request to make this closer to a real case, we
added `pelni_ebill`, which is the more relevant prototype for this ticket.

The PELNI mock domain lives in:

```text
src/tau2/domains/pelni_ebill/
data/tau2/domains/pelni_ebill/
tests/test_domains/test_pelni_ebill/
```

It mirrors the real PELNI chatbot tools from:

```text
/home/shaf/Assistx/pelni-ebill-ai/src/services/chatbot/tools/kb/kb_tools.py
/home/shaf/Assistx/pelni-ebill-ai/src/services/chatbot/tools/invoice/progress_tools.py
```

The mock keeps the same core behavior without calling live services. For
`search_knowledge_base`, it preserves the `query`, `top_k`, and
`similarity_threshold` contract, validation behavior, chunk truncation, and
similarity-style output, but uses local deterministic KB chunks. For
`get_invoice_progress`, it preserves the 8-digit invoice-number lookup rule,
vendor bill / description keyword lookup, division filtering, status
explanations, newest-first sorting, and max-three-invoices response format.

So the PELNI domain is not a generic demo. It specifically tests the real
e-Billing chatbot behavior: invoice progress lookup, e-Billing tutorial KB
answers, PPh/PPN tariff KB answers, clarification, and refusal for unsupported
requests.

The current task set covers seven representative cases: exact invoice lookup,
vendor bill lookup, keyword invoice lookup, generic tutorial clarification,
Pusat tutorial lookup, PPh 23 tariff lookup, and out-of-scope refusal. This is a
minimal but useful evaluation setup because it checks both tool use and final
communication.

To run it:

```bash
uv sync --extra dev
uv run tau2 check-data
uv run --extra dev python -m pytest tests/test_domains/test_pelni_ebill -q
uv run tau2 run \
  --domain pelni_ebill \
  --agent llm_agent \
  --agent-llm openrouter/qwen/qwen3.5-9b \
  --user-llm openrouter/qwen/qwen3.5-9b \
  --num-trials 1 \
  --save-to pelni_ebill_qwen35_9b_all_2
uv run tau2 view
```

A prototype run already exists at:

```text
data/simulations/pelni_ebill_qwen35_9b_all_2/results.json
```

It ran seven tasks with one trial using `llm_agent` and
`openrouter/qwen/qwen3.5-9b` for both agent and user simulator.

```text
7 tasks run
6 passed
1 failed
average reward: 0.8571
```

The failed task was the `training` keyword invoice lookup. The agent returned
the right invoice records, but missed the exact expected count phrase:
`Ditemukan 4 invoice (menampilkan 3 terbaru)`. That is a useful example of what
Tau-bench surfaces: not only wrong tool calls, but also response-format and
evaluation-strictness issues.

Recommendation: adopt Tau-bench partially for AssistX Suite agent development.
Use it for workflow reliability, tool correctness, policy compliance, grounding,
refusal behavior, and regression testing. Do not use it as the only evaluator
for everything: BNI document extraction still needs field-level accuracy checks,
and BSI campaign generation still needs rubric or human quality review.

Proposed decision: use `pelni_ebill` as the reference mock domain pattern, then
add one document-extraction domain next. Final decision can be filled after
stakeholder review.
