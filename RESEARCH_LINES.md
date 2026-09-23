# Research lines — from synthetic demo to real data at a financial entity

Status: proposal (2026-09). Scope: what the literature says about local agents that
debug databases from natural-language complaints, where this repo stands against it,
and at most three research lines to make it work on real SAS pipelines in a bank.

---

## 1. Where the repo actually is (verified in code)

| Claim in README | What the code does | Consequence on real data |
|---|---|---|
| "held-out success 0.70" | Gold `diagnostic_sql` **is** `defect.oracle_sql` verbatim (`slm/episodes.py:gold_trace`). Test split = same 67 defect classes, new seeds (`generate_episodes`). | The eval measures recall of 67 memorised templates, not diagnosis. A defect outside the catalog is out of distribution; expected performance is unknown and there is no reason to assume it is > 0. |
| "best-of-N 0.90" | Candidates are ranked with `reward.compute_reward`, which runs them against the **planted** trap/clean DBs (`slm/evaluate.py:run_best_of_n`). | This is pass@6 with an oracle. In production there is no planted defect and no clean DB, so the selector does not exist. The deployable number is the greedy 0.70 — on in-distribution data. |
| "discrepancy between two SAS-produced tables" | The task is *intra-table invariant violation on one row* (e.g. `ECL != PD*LGD*EAD`). There is no second run, no diff. | The real complaint ("these cycles differ between `src_basilea` and `rep_lgd`") is a **cross-run** problem; many real discrepancies do not violate any invariant (wrong cut-off date, wrong join key, stale FX table). |
| "uploads `.egp` projects" | `.egp` is a fake zip with `manifest.json` (`examples/app/agent.py:_egp_lineage`); the model prompt uses a hard-coded schema/lineage (`model_agent.py:schema_text/lineage_text`, `slm/lineage.py:_LAYERS`). | The SAS code never reaches the model. Real `.egp` (zip of `project.xml` + embedded programs, macros, `%include`, WORK tables) is not parsed. |
| "natural-language complaint" | Input is one suspect row; the 1k–100k-row Excel is not aggregated. | No complaint understanding, no set-level reasoning ("all RETAIL_HIP cycles after 202403"). |
| single defect | Exactly one planted defect per episode. | Real incidents mix several causes and *legitimate* differences (versioned data cuts, methodology changes). |

Net: the repo is a sound **execution-verifiable harness** (oracle, trap/clean,
reward) wrapped around a task that is easier and differently shaped than the target.
The harness idea is worth keeping; the task definition and the eval are not.

## 2. What the literature says (condensed)

- **Complaint-driven diagnosis is a database problem first.** QFix (Wang, Meliou,
  Wu, SIGMOD'17) takes a query log + complaints about wrong values and returns which
  query introduced the error, with exact (MILP) and scalable approximations. Rain /
  Reptile (Wu et al.) generalise "complaints" over aggregates and pipelines. DIFF
  (Abuzaid et al., VLDB'19) explains *which attribute combination* separates two
  sets of rows. None of these needs an LLM; all assume lineage/query history is known.
- **LLM SQL debugging is hard even for larger models.** BIRD-CRITIC / SWE-SQL
  (NeurIPS'25): Bird-Fixer (Qwen2.5-Coder-14B, agentic, trained) solves ~38% of real
  PostgreSQL user issues. Spider 2.0-DBT / ELT-Bench: repository-level pipeline
  repair is agentic, iterative, and execution-feedback driven.
- **Small models + agentic RL work when the reward is executable.** 2025–26 text-to-SQL
  work (Reward-SQL, ReToolSQL, SERL-SQL, AGRO-SQL) reports gains from GRPO with
  execution rewards at 2–8B, with small absolute improvements at the ~4B scale.
- **Lineage extraction from legacy code with LMs is feasible but imperfect**
  (schema-lineage benchmarks 2025; COBOL business-rule extraction). SQL-level lineage
  is well served by deterministic parsers (SQLGlot, LineageX); SAS DATA-step +
  macro code is the gap.
- Gap: I found **no** published work on SLM agents debugging SAS pipelines from
  complaints. Confidence: medium (arXiv full text was not readable from this
  environment; several 2026 papers were seen by title/abstract only).

Implication: the winning architecture in the literature is *deterministic core
(lineage + execution + diff), model at the edges (intent, hypothesis ranking,
SQL drafting, explanation)*. The repo does the reverse: the 0.5B model is asked to
produce the whole diagnosis in one shot.

## 3. Research lines (ordered; each has a kill criterion)

### L1 — Deterministic localisation core: lineage DAG + first-divergence search

**Hypothesis.** For cross-run discrepancies, most localisation can be done without a
model: parse `.egp` → column-level lineage DAG; for the suspect keys, compare the two
runs node by node in topological order; the first node where values diverge (within
tolerance) is the culprit step; DIFF-style explanation over the suspect set gives the
population pattern (segment, period, fusion flag).

**Work.**
1. Real `.egp` reader (`project.xml` + code nodes) → SAS lineage via the existing
   from-scratch parser / `alektebel/sas-lineage`; PROC SQL through SQLGlot.
2. Snapshot strategy for intermediates: SAS `WORK` tables vanish at session end.
   Either persist them (`options` + libname redirect in a debug run) or re-execute
   the two projects under instrumentation. **This is the main technical risk.**
3. First-divergence search (binary search over the DAG path when snapshots are
   expensive) + DIFF over the Excel of suspect cycles.

**Measure.** Top-1/top-3 step localisation and time-to-diagnosis on historical,
already-closed incidents from the team (even 20–30 is enough to start).

**Kill / pivot.** If intermediates cannot be persisted or re-run with bank data
under IT rules, cross-run localisation degrades to static reasoning over code and
L1 alone is not viable — that finding is itself decisive for the project scope.

### L2 — Realistic benchmark: mutate the SAS code, not the rows

**Hypothesis.** The target distribution is "two runs of the same pipeline differ
because one step changed". It can be generated with ground truth by construction by
applying mutation operators to the real SAS programs and running original vs mutant
on masked or real data inside the bank.

**Operators (SAS-specific, from typical incidents).** Join key / `MERGE` without
`IN=` or wrong `BY`; boundary `<` vs `<=` on dates/DPD; stale reference period or FX
table; SAS missing-value semantics (`.` sorts below every number, so `x < 0.03`
is true for missing); character truncation by `LENGTH`; `NODUPKEY` on the wrong key;
units/sign/percent vs fraction; `FIRST./LAST.` logic; format/informat changes.

**Protocol.** Split by *operator* and by *pipeline* (never by seed only). Include
multi-defect and "legitimate difference" episodes (answer = no bug). Replace the
oracle-only selector with a **reference-free verifier** that also exists in
production: apply the proposed fix to the localised step, re-run downstream, and
check the complaint disappears for suspect keys and nothing else moves.

**Kill / pivot.** If the verifier cannot be run in production (no re-execution),
best-of-N and GRPO have no deployable reward; stay with L1 + retrieval.

### L3 — Does a trained SLM earn its place? Ablation under the bank's envelope

**Hypothesis.** Given L1's core, an off-the-shelf 2–4B instruct model with a small
tool set (`lineage(field)`, `diff(node, keys)`, `sample(node, keys)`,
`run_sql(readonly)`) matches a fine-tuned 0.5B one-shot model on L2's held-out
operators. Training is justified only for the residual gap.

**Arms.** (a) L1 alone + templated report; (b) + untrained 2–4B tool loop;
(c) + SFT on L2 trajectories; (d) + GRPO with the L2 verifier as reward.
Report success, cost per diagnosis, and wall-clock on the CPU/llama.cpp envelope.

**Governance constraints to design in, not bolt on.** On-prem only; read-only
credentials; the model sees schema, lineage and aggregated diffs by default, raw rows
only on explicit extraction with masking; every query logged for audit; the output
is a hypothesis for an analyst, not an automatic correction (keeps it out of scope
for model-change processes that apply to regulatory calculations).

**Kill.** If (b) ≥ (c) within noise, drop fine-tuning entirely — leaner and easier to
validate internally.

## 4. What to reuse from the repo

Keep: execution-verifiable grading (`slm/oracle.py` pattern), trap/clean harness idea,
llama.cpp backend (`slm/backends.py`), SSE demo shell. Retire as evidence: the 0.70 /
0.90 figures (in-distribution, oracle-selected). Rewrite: `slm/episodes.py` task
definition (cross-run, set-level, code-mutation ground truth) and `slm/lineage.py`
(real parsed lineage instead of `_LAYERS`).

## Sources

- QFix — https://arxiv.org/abs/1601.07539
- Complaint-driven training data debugging (Rain) — https://arxiv.org/pdf/2004.05722
- DIFF — http://www.vldb.org/pvldb/vol12/p419-abuzaid.pdf
- SWE-SQL / BIRD-CRITIC — https://arxiv.org/abs/2506.18951
- Spider 2.0 — https://arxiv.org/pdf/2411.07763 ; ELT-Bench — https://arxiv.org/pdf/2504.04808
- Reward-SQL — https://arxiv.org/html/2505.04671 ; ReToolSQL — https://arxiv.org/pdf/2608.27796 ;
  SERL-SQL — https://arxiv.org/html/2608.00485 ; AGRO-SQL — https://arxiv.org/pdf/2512.23366
- Schema lineage extraction benchmarks — https://arxiv.org/pdf/2508.07179 ;
  LineageX — https://arxiv.org/pdf/2505.23133
