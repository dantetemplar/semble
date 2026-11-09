# Copilot results on Semble's benchmark annotations

Implementation: [`2cc75fddcdf3`](https://github.com/dantetemplar/semble/commit/2cc75fddcdf3).
These results use this repository's original 1251 queries and target labels, not
the separate Loc-Bench pilot discussed earlier in issue #327.

## Full remote-index run

[Audited per-query JSON](results/copilot-remote-index-2cc75fddcdf3.json).

| Measurement | Result |
| --- | ---: |
| Repositories completed | 63 / 63 |
| Queries successfully evaluated | 1251 / 1251 |
| Errors / pending queries | 0 / 0 |
| Query-micro file NDCG@10 | **0.8212245** |
| Complete-repository macro file NDCG@10 | **0.8216343** |
| Median client-observed query latency | 1203 ms |

Model: `metis-1024-I16-Binary`. Each original query is sent unchanged, requesting
up to 30 chunks. Results are checked against the actual indexed SHA/repository,
post-filtered to `benchmark_root`, deduplicated by first file occurrence, and
capped at 10 files. Scoring reuses `file_rank` and `ndcg_at_k`, including secondary
targets, as in the file-level probe baseline. Annotation spans are not used by
this file-level metric.

**Revision caveat:** only `httpx`, `lazy.nvim`, and `zig` match `repos.json` exactly.
Those 3 repositories contain 61 queries; their query-micro file NDCG@10 is
**0.8813946**. The other 60 repositories were searched at different remote indexed
commits, all recorded in the JSON. This is not a controlled fixed-revision
comparison or a replacement row for the local CPU-only leaderboard. In particular,
Semble's main benchmark uses span-aware chunk ranking, not this file-level metric.
Post-filtering a global remote result pool also differs from indexing only
`benchmark_root`, and the opaque remote model/backend cannot be version-pinned
beyond the returned model identifier.

Index status was verified before and after each repository; every search response
and hit was validated by the runner. A separate audit recomputed all 1251 task
fingerprints, query hashes, scope filters, file ranks, individual NDCG values,
per-language aggregates, and overall summaries from the saved data.

Raw JSON SHA256:
`ba1fa64d2a62eba95f2a3107aa88ffe4e02160c95c0ce8faa14e2b22b68e92a2`.

## Pinned diff-overlay run

The full overlay run is still in progress and resumes from per-query checkpoints.
There is no final full-suite overlay score yet. Large-diff guards and service
failures are explicit unavailable queries, not silently replaced by remote HEAD.
Conditional means from a partial run must not be compared with the complete
remote-index result above.

The earlier end-to-end smoke on the 20 `requests` queries completed without
errors: file NDCG@10 **0.9775325**, versus **0.9550651** for remote-only on those
queries. This is only a smoke test. Besides correcting changed-file contents, the
overlay adds a second candidate source and the inspected VS Code same-model merge
and score cutoff, so the difference is not an isolated revision-correction effect.
