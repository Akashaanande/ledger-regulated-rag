# Ledger

**Regulated-document RAG, with the eval harness built first.**

> *"I measure retrieval before I tune it."*

Most RAG portfolio projects load PDFs, chunk at 512 tokens, embed, stuff into a
vector DB, call an LLM, and screenshot a good answer. None of them can tell
you whether it actually works.

Ledger inverts that: **the eval harness is committed before the retriever
exists.** This README leads with a curve, not a demo.

The corpus is SEC EDGAR 10-K / 10-Q filings — public, messy, table-heavy.
A table spanning four pages, a footnote that materially changes the meaning
of the paragraph above it, and a wrong answer that's a compliance event
rather than a bad autocomplete. That's the problem worth solving.

---

## Architecture

```mermaid
flowchart TB
    subgraph Ingest["1 · Ingestion"]
        A[SEC EDGAR API<br/>10-K / 10-Q filings]
        B[Docling<br/>PDF to structured doc]
        C{Element router}
        A --> B --> C
    end

    subgraph Chunk["2 · Chunking strategies"]
        D1[Naive<br/>512 tokens, fixed]
        D2[Structural<br/>section-aware]
        D3[Contextual<br/>LLM-prefixed summary]
        C -->|prose| D1
        C -->|prose| D2
        C -->|prose| D3
        C -->|tables| T[Table serialiser<br/>markdown + caption]
        T --> D2
        T --> D3
    end

    subgraph Index["3 · Indexing"]
        E[bge-large embeddings]
        F[(OpenSearch<br/>kNN vector index)]
        G[(OpenSearch<br/>BM25 lexical index)]
        D1 --> E
        D2 --> E
        D3 --> E
        E --> F
        D1 --> G
        D2 --> G
        D3 --> G
    end

    subgraph Retrieve["4 · Retrieval"]
        H1[Dense only]
        H2[BM25 only]
        H3[Hybrid + RRF]
        H4[Hybrid + RRF<br/>+ cross-encoder rerank]
        F --> H1
        G --> H2
        F --> H3
        G --> H3
        H3 --> H4
    end

    subgraph Eval["5 · Eval harness — BUILT FIRST"]
        I[Gold Q/A set<br/>120 questions]
        J[Ragas metrics<br/>context recall / precision<br/>faithfulness]
        K[recall@k · MRR · nDCG]
        L[[RESULTS.md<br/>before/after table]]
        I --> J
        H1 --> J
        H2 --> J
        H3 --> J
        H4 --> J
        J --> K --> L
    end

    subgraph Serve["6 · Answer"]
        M[LLM w/ citation<br/>enforcement]
        N[Streamlit<br/>answer + source spans]
        H4 --> M --> N
    end

    style Eval fill:#1f6feb22,stroke:#1f6feb,stroke-width:3px
    style L fill:#1f6feb44,stroke:#1f6feb
```

---

## Results

Populated once the eval harness and baseline retriever exist. See
[`RESULTS.md`](./RESULTS.md) for the current before/after table.

## Scope

**In:** SEC EDGAR 10-K/10-Q filings, a hand-built gold question set, three
retrieval strategies benchmarked against each other, a measured improvement
curve.

**Out:** No chat UI beyond a bare Streamlit page, no multi-tenancy, no
fine-tuning, no agent.

## Getting started

```bash
cp .env.example .env
docker compose up -d        # starts OpenSearch
make eval                   # runs the current retrieval configs against the gold set
```

See [`DECISIONS.md`](./DECISIONS.md) for the reasoning behind the major
architecture choices, and [`docs/jira_backlog.csv`](./docs/jira_backlog.csv)
for the full build breakdown.

## Definition of done

- [ ] 120 hand-verified gold questions committed
- [ ] Five configurations benchmarked, results reproducible via `make eval`
- [ ] `RESULTS.md` with the full before/after table and the latency tradeoff called out
- [ ] `DECISIONS.md` covering: OpenSearch over Qdrant, hand-built evals over synthetic, contextual chunking cost/benefit, rerank latency tradeoff, table serialisation approach
- [ ] `docker compose up` runs the whole stack including OpenSearch
- [ ] Mermaid architecture diagram in README, above the fold
- [ ] Answers cite source spans; a wrong-but-confident answer is a bug, not a limitation

## License

[MIT](./LICENSE)
