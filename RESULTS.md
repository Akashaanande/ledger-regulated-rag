# Results

Populated by `make eval`. Not yet run — no retriever exists yet by design;
the eval harness ships first.

## Expected shape

| Configuration | recall@10 | MRR | Faithfulness | p95 latency |
|---|---|---|---|---|
| Naive 512 + dense | — | — | — | — |
| Structural + dense | — | — | — | — |
| Structural + hybrid RRF | — | — | — | — |
| Contextual + hybrid RRF | — | — | — | — |
| + cross-encoder rerank | — | — | — | — |

Numbers above will be filled in as each configuration ships, per the build
sequence in [`docs/jira_backlog.csv`](./docs/jira_backlog.csv).
