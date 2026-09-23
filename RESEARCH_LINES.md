# Research framework — local SAS-run Q&A and error hypotheses

Status: proposal, 2026-09. Supersedes the earlier cross-run (two-version diff) framing.

## 0. Fixed decisions (scope)

| Decision | Value | Why |
|---|---|---|
| Input | **One** SAS EG 8.6 run (`.egp` = zip of `project.xml` + code nodes; logs if saved) + its output tables | User scope; no second version to diff against |
| Task | Answer questions about the DB and propose **error hypotheses**, each backed by an executed query | "¿Por qué X vale V?", "¿Dónde se pierden filas?", "¿Qué ciclos incumplen R?" |
| Runtime | Local, **≤ 32 GB RAM**, CPU (llama.cpp) | Bank constraint; the 35B MoE left no headroom |
| Model | `Qwen3.5-9B` Q4_K_M (~6 GB); `Qwen3.5-4B` fallback | Leaves ~20 GB for DuckDB/SASPy/OS |
| Data access | SASPy (read-only user) → local DuckDB snapshot of the run's tables | Queries never hit prod; auditable |
| Division of labour | Deterministic tools compute; model plans, hypothesises, explains | Small models are unreliable at long tool chains |
| Training | None until §4 RL3 gate clears; GPUs (2×5060) only for later QLoRA → GGUF | Fine-tuning changes behaviour, not memory/speed |

**Known limit (by construction).** A single run can show *what happened*, not that
it is *wrong*, unless the question or a rule supplies an expectation. The system
must say so rather than guess.

## 1. What the current repo contributes

Reusable: execution-verified grading pattern (`slm/oracle.py`), llama.cpp backend
(`slm/backends.py`), rule catalog (`vendor/defect_catalog.py` — intra-table
invariants fit the single-run scope), SSE demo shell.
Not evidence for this scope: the 0.70/0.90 figures (same 67 defect classes in
train/test; best-of-N picks with the planted-defect oracle). Not present: real
`.egp` parsing, real lineage, NL questions, retrieval, sufficiency control.

## 2. Evaluation first (everything below is measured against this)

**Bench-real (primary).** 30 → 100 real questions the analyst already solved. Each
item stores: question, gold answer/hypothesis, **gold evidence set** (code nodes,
log lines, tables/columns, confirming query), and whether more info was needed.

**Bench-mut (secondary, scalable).** Mutate one SAS step (join key, `<`/`<=`,
missing-value semantics, `NODUPKEY` key, stale period, `LENGTH` truncation), run,
auto-generate a symptom question from the observed effect. Ground truth = mutated step.
Split by operator and by project, never by seed only.

**Metrics.**
- Answer: correct hypothesis (step + column), confirmed by an executed query.
- Context: gold-evidence recall @ token budget (1k / 2k / 4k); tokens per turn.
- Sufficiency: selective accuracy (accuracy vs. coverage curve), missed-info rate,
  unnecessary tool calls.
- Cost: wall-clock per question on the target CPU; peak RSS.

## 3. Reference loop

```
question
  → anchor extraction        (tables, columns, keys, steps, periods, values)
  → candidate retrieval      (lineage graph walk from anchors; log index by step;
                              BM25/dense only over prose docs)
  → trimming to budget       (RL1)
  → reason → hypotheses      (model, working memory JSON)
  → sufficiency gate         (RL2)  ── insufficient → tool call / ask user ─┐
  → confirming query (DuckDB, read-only) → verdict                           │
  → answer | abstain with "what is missing"  ←──────────────────────────────┘
```

Working memory = one JSON rewritten each turn: `{question, anchors, visited_steps,
hypotheses[{claim, step, cols, query, verdict}], missing[]}`. Raw chat history is
dropped. Per-turn budget ≈ 1k system/tools + 0.5k memory + 2k context + 0.5k last
tool result.

## 4. Research lines

### RL1 — Context construction and trimming (GLiNER2.5 as candidate)

Note: there is no "GLiNER 5.2"; the current release is **GLiNER2.5** (Fastino,
Apache-2.0: small 74M, base 0.2B, multi 0.3B; schema-driven NER + classification +
structured extraction, CPU-friendly).

**What GLiNER can and cannot do here.**
- Good fit: **anchor extraction** from the question and from prose docs (table,
  column, key, period, metric, business rule). Anchors drive the lineage graph walk,
  which is the retrieval that matters for code and logs.
- Weak fit: **query-conditioned relevance over a top-100**. GLiNER classifies text
  against labels; relevance to a specific question is a pair task (cross-encoder /
  pruner). Feeding `question + chunk` into GLiNER classification is possible but
  off-design and untested.
- Unknown: accuracy on SAS identifiers (`LGD_FINAL`, `WORK.T_L3`) — it was trained
  on natural text. Must be measured, with an exact-match dictionary of schema names
  as the baseline it has to beat.

**Arms (all on the same top-100 candidates, same budget).**
1. Dictionary anchors + graph walk (no ML) — baseline.
2. GLiNER2.5 anchors + graph walk.
3. (2) + cross-encoder reranker (~0.6B) over top-100 → top-k.
4. (2) + sentence-level pruner (Provence/XProvence style) → keep relevant sentences.
5. (2) + GLiNER2.5 relevance classification over `question+chunk` (the user's idea).

**Question.** Which arm maximises gold-evidence recall at 2k tokens with the lowest
CPU cost, and does it move end-task accuracy?
**Kill.** If arm 1 is within noise of the best arm, drop the ML trimming.

### RL2 — "Do we need more information?" (sufficiency control)

Evidence: adding retrieved context makes models *less* likely to abstain and more
confident when wrong (Google, *Sufficient Context*, ICLR'25). Self-report alone is
not a reliable gate.

**Arms.**
1. **Deterministic checklist** (no model): for the asked field, are all lineage-path
   nodes' code and log entries loaded? Does each hypothesis have an executed query?
   Is there an expectation (value, rule, source) to judge "wrong"? Any "no" →
   fetch or ask.
2. **Model self-assessment**: explicit `need_more_info{what, why}` action before
   answering.
3. **Small sufficiency classifier** (the "intervention model" idea): scores
   `(question, context)` as sufficient/insufficient; could be GLiNER2.5
   classification or a fine-tuned small encoder — ties back to RL1.
4. Combination: checklist as hard gate, (2)/(3) as soft signal.

Actions when insufficient: fetch (graph neighbour, log step, profile query) or ask
the user a single targeted question (typically: the expectation).
**Question.** Which gate gives the best selective-accuracy curve at the lowest
number of tool calls?
**Kill.** If the checklist alone matches (4), keep it — auditable and free.

### RL3 — Does fine-tuning earn its place?

Arms: 9B base + RL1/RL2 best; 4B base + same; 9B QLoRA distilled from verified
trajectories (teacher: larger model on the GPU box; keep only trajectories whose
confirming query passes). Export to GGUF, re-measure on CPU.
**Gate.** Train only if base 9B < ~60–70% on Bench-real *and* the failures are
behavioural (tool-use format, hypothesis quality), not missing context (→ RL1) or
missing expectations (→ RL2).

## 5. Order and milestones

1. SASPy connectivity spike on EG 8.6's SAS server (read-only user). Blocker check.
2. `.egp` reader + lineage graph + log index by step; measure parser coverage
   (% of steps/columns resolved; macros, `%include`, dynamic code).
3. Bench-real v0 (30 items) + Bench-mut generator.
4. RL1 and RL2 in parallel on Bench-real v0.
5. RL3 only if the gate clears.

## 6. Open questions (answers change the design)

- Do the saved `.egp` files include logs? If not, can the run be re-executed to capture them?
- Typical table volume (10^4 / 10^6 / 10^8 rows) → DuckDB snapshot feasibility.
- Is there written methodology / business rules, or is the code the only spec?
- Same 32 GB machine for model + SAS client, or separate?

## Sources

- GLiNER2.5 — https://fastino.ai/blog/gliner2-5-span-free-information-extraction ;
  GLiNER2 — https://arxiv.org/html/2507.18546v1 , https://github.com/fastino-ai/GLiNER2
- Sufficient Context (ICLR'25) — https://arxiv.org/abs/2411.06037
- Provence / XProvence context pruning — https://arxiv.org/pdf/2601.18886
- Information Gain Pruning — https://arxiv.org/abs/2601.17532
- QFix (complaint-driven diagnosis) — https://arxiv.org/abs/1601.07539
- SWE-SQL / BIRD-CRITIC — https://arxiv.org/abs/2506.18951
- SASPy — https://sassoftware.github.io/saspy/ ; SAS MCP servers —
  https://github.com/sassoftware/sas-mcp-server , https://pypi.org/project/sas-mcp/0.1.0/
- Qwen3.5 small models — https://artificialanalysis.ai/articles/qwen3-5-small-models
