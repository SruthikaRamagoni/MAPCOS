# MAPCOS architecture and verified status

```
image ──► CNN (Grad-CAM) ─┐
      └─► Follicle (watershed) ─┴─► Grounding ─┐
lab values ─► Lab model ─────────────────────────┤
symptoms ──► Symptoms model (+dampener) ─────────┼─► Synthesis (Rotterdam 2-of-3)
clinical findings (per-ovary count, volume,       │        │
  AMH+assay cut-off, hirsutism, oligo/anov.) ─────┘        ▼
TSH / prolactin + lab's reference range ──────────────► Exclusion
                                         guideline KG + RAG (cited passages) ─► Report (JSON + Markdown)
```
Orchestrator: `mapcos/orchestrator/master_orchestrator.py` (threads; per-agent error isolation; missing modality = `skipped`).

## Key design decisions (say these at review)
1. **Models are evidence, not criteria.** CNN / follicle / lab / symptom outputs are listed as *supporting evidence*. A Rotterdam criterion is MET only from confirmed clinical information (see `synthesis_agent.py` docstring).
2. **Three-state logic** (MET / NOT_MET / UNKNOWN): missing never becomes negative.
3. **No invented reference ranges**: exclusion checks need the reporting lab's range with the value.
4. **Every guideline sentence is a stored chunk with a source**; low retrieval score → `INSUFFICIENT_EVIDENCE`.
5. **Unlinked modalities**: `modalities_linked` defaults to False and the report says so.

## Status (what the code + tests actually show)
| Component | Files | Tests | Status |
|---|---|---|---|
| Dedup + leakage-safe split | `data/dedup.py` | `tests/test_dedup_metrics.py` | Implemented, tested on synthetic images. **Not yet run on the real Kaggle data.** |
| CNN comparison on clean split | `training/compare_cnns.py` | – (needs GPU/data) | Patched; **must be rerun on Kaggle** – old CNN numbers are unverified. |
| Synthesis | `agents/synthesis_agent.py` | `tests/test_synthesis.py` | Implemented + tested |
| Exclusion | `agents/exclusion_agent.py` | `tests/test_exclusion.py` | Implemented + tested |
| Knowledge graph | `knowledge_graph/*` | `tests/test_rag_kg.py` | Local JSON backend tested; Neo4j loader written, untested (no server) |
| RAG | `rag/retriever.py`, `rag/corpus/` | `tests/test_rag_kg.py` | TF-IDF index tested. Corpus = paraphrased passages; ingest the real guideline PDF for verbatim text. |
| Report | `reports/report_generator.py` | `tests/test_orchestrator_report.py` | Implemented + tested |
| Orchestrator | `orchestrator/master_orchestrator.py` | `tests/test_orchestrator_report.py` | Implemented + tested with stub agents; real-agent run needs TF + models |
| Metrics (acc/prec/rec/F1/spec/AUC/CM/calibration) | `evaluation/metrics.py` | `tests/test_dedup_metrics.py` | Implemented + tested |

## Known limitations
- Public image and tabular datasets are from different people: no patient-level multimodal validation.
- Earlier CNN scores (≈1.0) are likely inflated by duplicate leakage; the Review-II CNN table could not be reproduced from the repo/notebook.
- Whole-image follicle count is not a clinical per-ovary count; follicle agent has no ground-truth validation.
- Lab model is weak (hold-out AUC 0.637). Symptoms metrics exclude the rule-based dampener; dampener counts weight gain as an "androgenic sign", which is not clinically justified (left unchanged, flagged).
- Dataset TSH/prolactin column names were not verified; supply the lab's reference ranges.
