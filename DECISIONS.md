# Decisions

Architecture decision records for Ledger. Each entry: context, decision,
consequences. Updated as the build progresses — see the corresponding Jira
ticket for status.

---

## ADR-001: OpenSearch over a dedicated vector DB

**Status:** Proposed

**Context:** Need a store for both dense (kNN) and lexical (BM25) retrieval.

**Decision:** Use OpenSearch for both, combined via Reciprocal Rank Fusion,
rather than pairing a dedicated vector DB (Qdrant/Pinecone) with a separate
lexical store.

**Consequences:**
- One engine, no dual-write or cross-store consistency problem.
- Ten years of operational OpenSearch experience transfers directly.
- Appears by name as an accepted vector store in real-world data engineering
  job descriptions.
- Tradeoff: less specialized ANN performance than a purpose-built vector DB
  at very large scale — not a constraint at this corpus size.

---

## ADR-002: Hand-built eval set over synthetic/LLM-generated

**Status:** Proposed

**Context:** Need 120 gold Q/A pairs to evaluate retrieval quality.

**Decision:** Write every question and verify every answer span by hand.

**Consequences:**
- Costs real time (~1 week of the build).
- LLM-generated eval sets measure whether the retriever agrees with the
  generator, not whether it's correct — that's a different, weaker signal.
- The resulting eval set is defensible in an interview.

---

## ADR-003: Contextual chunking cost/benefit

**Status:** Pending measurement

**Context:** Contextual chunking (LLM-prefixed summaries) costs extra LLM
calls at ingest time.

**Decision:** TBD — will be settled by the recall/faithfulness delta measured
in `RESULTS.md` against structural chunking alone.

---

## ADR-004: Cross-encoder rerank latency tradeoff

**Status:** Pending measurement

**Context:** Reranking adds meaningful p95 latency.

**Decision:** TBD — will be settled once `RESULTS.md` shows the recall gain
per added millisecond. Expected conclusion: worth it for a compliance
workflow, not for a low-latency autocomplete.

---

## ADR-005: Table serialisation approach

**Status:** Proposed

**Context:** SEC filings contain multi-page tables that are meaningless if
split naively by token count.

**Decision:** Serialise tables to markdown with an attached caption before
chunking, keeping them intact as a unit rather than splitting by token
window.
