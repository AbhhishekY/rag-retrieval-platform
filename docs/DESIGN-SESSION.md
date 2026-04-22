---
stepsCompleted: [1, 2]
inputDocuments: []
session_topic: 'Production-grade retrieval platform (Assignment 1): 1000+ doc ingest, intelligent chunking, hybrid search (BM25+semantic+cross-encoder rerank), metadata filtering, search API, eval framework (P@5/R@5/NDCG across 3 configs), p95 <500ms latency, cost/1K queries'
session_goals: 'Make defensible technical decisions that can be built in a few hours AND justified with numbers; produce a concrete experiment matrix; surface failure modes upfront'
selected_approach: 'ai-recommended'
techniques_used: ['First Principles Thinking', 'Morphological Analysis', 'Failure Analysis (Pre-mortem)']
ideas_generated: []
context_file: ''
benchmark_corpus: 'MultiHop-RAG (yixuantt/MultiHopRAG) — 609 news articles + 2,556 Q/A pairs with evidence mapping'
time_budget: 'a few hours'
---

# Brainstorming Session Results

**Facilitator:** Abhi
**Date:** 2026-04-22

## Session Overview

**Topic:** Production-grade retrieval platform build (Assignment 1)

**Goals:**
- Architect a system meeting all must-haves: 1K+ doc ingest (incremental), intelligent chunking, hybrid search + rerank, metadata filtering, search API with score breakdown, eval with P@5/R@5/NDCG, p95 <500ms latency, cost projection
- Produce experiment matrix to answer "hard mode" signals: which chunk size wins, when BM25 beats semantic, cold/warm latency, $/1K queries
- Avoid time-sinks — pre-mortem to mitigate risks upfront

**Benchmark Corpus:** MultiHop-RAG (`yixuantt/MultiHopRAG`) — 609 news articles + 2,556 Q/A pairs with evidence-passage ground truth, ~18 MB, ODC-BY license

## Technique Selection

**Approach:** AI-Recommended Techniques

**Recommended Techniques:**
- **First Principles Thinking** — strip chunking/retrieval/rerank to fundamentals so every choice has a one-line defense
- **Morphological Analysis** — systematic parameter grid to design the minimum set of experiments that answer "which config wins"
- **Failure Analysis / Pre-mortem** — fast-forward to stuck-at-hour-2 scenarios and mitigate now

**AI Rationale:** Time-boxed technical design problem with hard-mode defensibility requirements. Need convergent-fast ideation (not 100+ wild ideas) — these three techniques produce: defensible defaults (Phase 1), a concrete experiment plan (Phase 2), risk-mitigated execution (Phase 3).

## Technique Execution Results

### Phase 1: First Principles Thinking

**[Principle #1] One vector = one topic**
_Concept_: Embedding compresses a chunk into ONE vector. If that chunk spans two topics, the vector becomes an average that represents neither well. Topical coherence > raw size.
_Novelty_: Reframes chunking from a "size" decision to a "topology" decision — where you cut matters more than how many tokens fit.

**[Principle #2] Overlap = duplication insurance, not "context continuity"**
_Concept_: Embedding models process chunks independently, not sequentially. Overlap exists so a key sentence that sits on a boundary gets captured cleanly by at least one chunk's interior.
_Novelty_: Common tutorials mis-sell overlap as "preserving narrative flow." The real mechanism is boundary-hedging.

**[Principle #3] Query distribution dictates chunk size**
_Concept_: MultiHop-RAG's multi-hop queries want SMALLER chunks + higher top-k (stitch evidence across docs). Single-hop QA wants LARGER chunks (answer lives in one place).
_Novelty_: "Right chunk size" isn't a corpus property — it's a query-distribution property. This is why chunk-size sweeps matter per-benchmark.

**[Decision #1 — LOCKED] Recursive chunking as default**
_Rationale_: MultiHop-RAG is clean news text → recursive captures paragraph/sentence structure without paying the semantic-chunker embed cost. Run ONE comparison cell with semantic chunking on a subset to defend the choice with numbers.
_Defense one-liner_: "Recursive costs zero ingest-time embedding calls and respects natural boundaries on clean text; semantic was evaluated but added X% ingest latency for <Y% NDCG@5 gain."

**[Principle #4] BM25 and semantic have orthogonal failure modes**
_Concept_: BM25 scores rare-token overlap (bag of words); semantic scores meaning via learned embeddings. BM25 wins on rare proper nouns, identifiers, exact phrases, OOV tokens. Semantic wins on paraphrase, synonymy, conceptual queries.
_Novelty_: Hybrid isn't "more retrieval = better." It's covering blind spots. The α weight in α·BM25 + (1-α)·semantic is really a bet on the query distribution's rare-token ratio.

**[Decision #2 — LOCKED] Query-by-query diff for "BM25 beats semantic" evidence**
_Concept_: For the hard-mode signal, run all ~2,556 MultiHop-RAG queries through (a) BM25-only and (b) semantic-only. Rank queries by BM25_NDCG@5 − semantic_NDCG@5. Take the top 10 biggest wins for BM25 → label them → inspect for patterns (rare entities, exact phrases, numeric filters).
_Defense_: Turns "BM25 wins sometimes" hand-waving into a specific, replicable failure-mode case study with numbers.

**[Decision #3 — LOCKED] Stack = Azure OpenAI embeddings + local everything else**
_Components_: Azure OpenAI `text-embedding-3-small` for embeddings; local `rank_bm25` for BM25; local FAISS `IndexFlatIP` for vectors; local sentence-transformers `cross-encoder/ms-marco-MiniLM-L-6-v2` for rerank.
_Rationale_: Azure AI Search would hide the score composition — kills the "tunable weights + score breakdown" requirement. Local stack gives full control. Azure used only where it excels (embeddings).

**[Decision #4 — FINAL] Embedding model = LOCAL sentence-transformers/all-MiniLM-L6-v2**
_Rationale_: CPU-only hardware + p95 <500ms budget makes network embedding calls a non-starter. MiniLM runs in-process (~8ms), zero network, zero cost, zero cold-cache penalty. Already validated in user's semantic_engine codebase.
_Azure OpenAI role_: RESERVED for optional generation step (gpt-4_1_dev_1) if answer-synthesis is added later. Not used in retrieval path.
_Defense one-liner_: "On CPU-only hardware with p95 <500ms SLO, in-process embeddings are non-negotiable. MiniLM-L6 costs 8ms and 0 dollars; any network embedder blows ~100ms and adds cold-start tails."

**[Decision #5 — LOCKED] Rerank model = cross-encoder/ms-marco-MiniLM-L-6-v2 (LOCAL)**
_Rationale_: Azure has no cross-encoder API. LLM-as-reranker is ~$0.50/1K queries AND +300ms. Local MiniLM is free and ~25ms/pair on CPU.

**[Decision #6 — LOCKED] Vector DB = FAISS IndexFlatIP (in-memory)**
_Rationale_: For <100K vectors, exhaustive search is faster than ANN due to zero graph-traversal overhead. Zero recall loss. <2ms for 5K-chunk MultiHop-RAG. Drop-in swap to IndexHNSWFlat if corpus scales past 100K.

**[Environment Constraint] CPU-only (no NVIDIA GPU detected)**
_Implication_: Rerank top-10 (not top-20) on CPU to stay under p95 500ms. Use async I/O to overlap Azure embedding call with BM25 scoring.

**[Decision #7 — LOCKED] Async/parallel query path**
_Concept_: Fire Azure embedding call + BM25 scoring in parallel via `asyncio.gather`. BM25 finishes in ~5ms while embed call takes ~80-150ms — effectively free overlap.
_Saves_: ~80-100ms off every query.

**[Cost Projection — REVISED for all-local retrieval]**
_Ingest_: $0 (all local)
_Per 1K queries (retrieval only)_: $0
_Per 1K queries (with optional gpt-4_1_dev_1 generation)_: ~$0.10-$0.50 depending on context size — populate from actual Azure billing rates.

### Phase 2: Morphological Analysis — Experiment Matrix

**Full grid:** 162 possible combinations across chunk_size × strategy × overlap × retriever × α × rerank. Non-viable in time budget.

**Shortlist — 9 runs organized by what they prove:**

**Tier 1 — REQUIRED by assignment (3 runs, full 2,556 queries):**
1. `semantic-only` @ recursive-512-10% — baseline
2. `hybrid-RRF` @ recursive-512-10% — proves hybrid > semantic
3. `hybrid+rerank` @ recursive-512-10% — production config (top-20 → top-5)

**Tier 2 — Chunk size sweep (200-query subset):**
4. chunk=256 @ hybrid+rerank — multi-hop hypothesis test
5. chunk=1024 @ hybrid+rerank — big-chunk baseline

**Tier 3 — Alpha sweep (200-query subset):**
6. α=0.3 (lean semantic)
7. α=0.7 (lean BM25)

**Tier 4 — Chunking strategy ablation (200-query subset):**
8. fixed-window-512 — strawman to defend recursive
9. semantic chunker — ceiling check

**Tier 5 — Derived (no new runs):**
- "When does BM25 beat semantic?" → diff Run #1 vs BM25-only scores from Run #2, take top-10 wins for BM25, inspect patterns
- Cold vs warm latency → separate first-query latency from p50/p95/p99 during Run #3
- Cost/1K queries → $0 for all-local retrieval

**API contract locked:**
- `POST /search` with `{query, filters, top_k, fusion_method, alpha, rerank}`
- Response includes `scores: {bm25, semantic, hybrid_fused, rerank, final}` per result
- Metadata filters: `source`, `date`, `category`
- Incremental ingest: SHA-256 content hash in SQLite manifest, only re-embed on change

### Phase 3: Pre-mortem — Failure Modes + Mitigations

**TRAP #1 — Corpus mismatch:** MultiHop-RAG is JSON (not PDFs), 609 docs (not 1000+).
_Mitigation_: Pad with 400 arXiv CS PDFs (background download) → 1,009 mixed-format docs, only MultiHop-RAG in eval set.

**TRAP #2 — Score fusion is nonsense without normalization.**
_Mitigation_: RRF as default (rank-based, no scale issue). α-weighted with per-query min-max normalization for the "tunable" story.

**TRAP #3 — Model download hangs on first use.**
_Mitigation_: Pre-download both MiniLM embedding + cross-encoder in Hour 0 via one-liner script.

**TRAP #4 — Async I/O with sync libs blocks event loop.**
_Mitigation_: Use `loop.run_in_executor(None, ...)` for both embed and BM25. `asyncio.gather` the futures.

**TRAP #5 — Cross-encoder not batched.**
_Mitigation_: Pass all 20 pairs to `cross_encoder.predict(pairs, batch_size=32)` in one call, not a loop.

**TRAP #6 — Sequential query eval takes 30+ min per config.**
_Mitigation_: `ThreadPoolExecutor(8)` parallel eval. ~3-4× speedup.

**TRAP #7 — Qrels format mismatch with pytrec_eval.**
_Mitigation_: Inspect MultiHopRAG.json structure first. Write NDCG@k manually (30 lines) rather than forcing library compatibility.

**Pre-flight (Hour 0) checklist:**
1. `pip install fastapi uvicorn sentence-transformers rank-bm25 faiss-cpu pymupdf datasets numpy pandas pydantic httpx pytest`
2. Pre-download both HF models
3. Load MultiHop-RAG, inspect one record's qrels shape
4. Kick off arXiv PDF background download (400 cs.AI papers)
5. Verify Python environment + no CUDA surprises

## Creative Facilitation Narrative

Three-phase convergent-fast brainstorm calibrated to a few-hours technical build: First Principles → Morphological Analysis → Pre-mortem. User's environment (CPU-only Windows, existing semantic_engine codebase with MiniLM already wired, Azure OpenAI available for generation) drove the all-local retrieval decision. 7 decisions locked, 9 experiments scoped, 7 top failure modes mitigated pre-build.

## Session Highlights

**Key decisions locked:**
1. Recursive chunking (512 tokens, 10% overlap)
2. RRF hybrid fusion (+ α-weighted as comparison)
3. All-local retrieval stack (MiniLM + rank_bm25 + FAISS + cross-encoder)
4. FAISS IndexFlatIP (exhaustive for <100K vectors, zero recall loss)
5. CPU-only rerank top-10 → top-5 (reranking top-20 optional if budget allows)
6. Azure OpenAI reserved for optional generation (gpt-4_1_dev_1), NOT retrieval
7. Corpus = MultiHop-RAG (609, eval) + arXiv PDFs (400, noise) = 1,009 mixed-format

**Hard-mode signals — how we answer each:**
- "Which chunk size wins?" → Tier 2 (3-point sweep at best config)
- "When does BM25 beat semantic?" → Tier 5 derived (query-by-query diff, top-10 wins labelled)
- "Cold vs warm latency" → First-query vs steady-state during Run #3
- "Cost/1K queries" → $0 retrieval + optional Azure generation cost table


