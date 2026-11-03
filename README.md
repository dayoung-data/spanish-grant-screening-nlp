# Sample data (synthetic)

The real application texts, annotations and trained models are confidential and are **not** included.
The two files below are small, fully invented examples that show the *format* each notebook expects.

| File | Used by | Columns |
|---|---|---|
| `claim_evidence_examples.csv` | Stage 1 (claim / evidence / other sentence classifier) | `text`, `label` |
| `support_pairs_example.csv` | Stage 2 (claim-support cross-encoder) | `sentence1` (claim), `sentence2` (candidate support), `label` (0 none, 1 partial, 2 full) |
