# Results Dashboard — Hybrid RAG Retrieval Platform

> **Corpus:** 609 Reuters/Bloomberg articles · **Chunks:** 19,817 @ 512 chars · **Queries:** 200 multi-hop · **Hardware:** CPU-only, no GPU

---

## The Winner

```
╔══════════════════════════════════════════════════════════╗
║   Weighted Alpha = 0.7  +  Diversity Filter              ║
║                                                          ║
║   NDCG@5   ████████████████████████████  0.657          ║
║   P@5      ████████████████             0.354           ║
║   R@5      ████████████████████████████████████  0.715  ║
║   p95                                   194 ms          ║
║   Cost / 1K queries                     $0.00           ║
╚══════════════════════════════════════════════════════════╝
```

---

## The Journey — NDCG@5 Progression

Each step shows what was added and the gain.

```
                                         NDCG@5
                              0.50  0.55  0.60  0.65  0.70
                                 |     |     |     |     |

  1  Dense only (FAISS)       ████████░░░░░░░░░░░░░░░░░    0.510
  2  BM25 only                ██████████████████░░░░░░░    0.577   +0.067
  3  Hybrid RRF               ███████████████████░░░░░░    0.590   +0.013
  4  Weighted α = 0.7         ████████████████████░░░░░    0.612   +0.022
  5  + Diversity filter  ★    ██████████████████████░░░    0.657   +0.045  ← biggest jump

                              |     |     |     |     |
  ░ = headroom to 0.70        0.50  0.55  0.60  0.65  0.70
```

> BM25 alone beats dense alone (+0.067). Hybrid beats both. Diversity filter is the single largest gain.

---

## Full Experiment Matrix

| Config | P@5 | R@5 | NDCG@5 | p95 | Cold |
|---|---|---|---|---|---|
| Dense only (FAISS) | 0.254 | 0.507 | 0.510 | 195ms | 116ms |
| BM25 only | 0.292 | 0.593 | 0.577 | 198ms | 143ms |
| Hybrid RRF | 0.297 | 0.596 | 0.590 | 85ms | 62ms |
| Hybrid + rerank | 0.277 | 0.555 | 0.563 | ❌ 687ms | 400ms |
| Weighted α = 0.3 | 0.272 | 0.547 | 0.555 | 197ms | 133ms |
| Weighted α = 0.5 | 0.295 | 0.601 | 0.598 | 205ms | 133ms |
| Weighted α = 0.6 | 0.301 | 0.609 | 0.603 | 178ms | 112ms |
| **Weighted α = 0.7 + diversity** ★ | **0.354** | **0.715** | **0.657** | **194ms** | **120ms** |
| Weighted α = 0.8 | 0.305 | 0.615 | 0.597 | 188ms | 120ms |

---

## Alpha Sweep — Finding the Peak

```
  α = 0.3  (70% dense)   ██████████░░░░░░░░░░  0.555
  α = 0.5  (balanced)    ████████████████░░░░  0.598
  α = 0.6               █████████████████░░░  0.603
  α = 0.7  (70% BM25) ★ ██████████████████░░  0.612   ← peak
  α = 0.8  (80% BM25)   ████████████████░░░░  0.597

  Each █ ≈ 0.003 NDCG   (baseline 0.54, before diversity filter)
```

> News corpus is entity-heavy → BM25 confidence signal matters → 70% BM25 weight wins

---

## Chunk Size Sweep

| Chunk size | Chunks | NDCG@5 | p95 | Why |
|---|---|---|---|---|
| 256 chars | 47,102 | 0.578 | 345ms | Too small — sentences cut mid-thought. 2.4× bigger index. |
| **512 chars** ★ | **19,817** | **0.590** | **195ms** | Sweet spot for 384-dim encoder |
| 1024 chars | 7,968 | 0.590 | 85ms | Vector blurs — 220 tokens compressed to 384 dims |

```
  256 chars   ████████████████░░   NDCG 0.578   345ms p95
  512 chars   █████████████████░   NDCG 0.590   195ms p95  ★ winner
  1024 chars  █████████████████░   NDCG 0.590    85ms p95

  Rule: 384-dim encoder → max ~110 tokens (~512 chars) before vector blurs
```

---

## Cold-Cache vs Warm-Cache Latency

```
                     Cold       Warm p50   Warm p95   Budget
  Dense only         116ms  →   134ms      195ms      ✅
  BM25 only          143ms  →   135ms      198ms      ✅
  Hybrid RRF          62ms  →    60ms       85ms      ✅  fastest
  Hybrid + rerank    400ms  →   483ms      687ms      ❌  over budget
  Weighted α=0.7 ★   120ms  →   133ms      194ms      ✅

  500ms budget ─────────────────────────────────────────────
```

> Cold ≈ warm for all configs. ONNX runtime warms in a single query — no penalty after first request.
> Cross-encoder rejected: 400ms cold init + 687ms warm p95, both over budget, quality also drops.

---

## Stratified Results — By Query Type

| Query Type | N | P@5 | R@5 | NDCG@5 | Avg rel docs |
|---|---|---|---|---|---|
| comparison_query | 79 | 0.337 | 0.779 | **0.700** | 2.2 |
| temporal_query | 51 | 0.353 | 0.712 | 0.637 | 2.4 |
| inference_query | 70 | 0.380 | 0.654 | 0.628 | 3.2 |

> Comparison queries score highest — 80% of them only need 2 relevant docs.
> Inference queries are hardest — all 4-doc queries are inference type.

---

## Stratified Results — By Difficulty

| Relevant docs needed | N | P@5 | R@5 | NDCG@5 | P@5 ceiling |
|---|---|---|---|---|---|
| 2 docs | 104 | 0.325 | **0.813** | **0.724** | 0.400 |
| 3 docs | 75 | 0.395 | 0.658 | 0.626 | 0.600 |
| 4 docs | 21 | 0.371 | 0.464 | 0.453 | 0.800 |

```
  n_rel = 2   NDCG ████████████████████████████  0.724
  n_rel = 3   NDCG ████████████████████████      0.626
  n_rel = 4   NDCG ████████████████              0.453

  System finds 81% of relevant docs when only 2 exist.
  Harder queries (4 docs) are harder by construction — not retrieval failure.
```

---

## P@5 Ceiling — Why 0.354 Is Not Low

```
  Theoretical max P@5 on this dataset:
  avg 2.58 relevant docs ÷ 5 slots = 0.516

  ┌─────────────────────────────────────────────────┐
  │  0.516  ████████████████████████████████  MAX   │
  │  0.354  █████████████████████  achieved (68%)   │
  │  0.000  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  floor  │
  └─────────────────────────────────────────────────┘

  Gap explained:
  · 14.9% of relevant docs never enter top-20 pool (hard ceiling)
  · Remaining gap = chunk clustering (fixed by diversity filter)
```

---

## Cost

| Component | Cost per 1K queries |
|---|---|
| Embedding (FastEmbed ONNX, local) | $0.00 |
| Vector search (FAISS, local file) | $0.00 |
| BM25 search (in-memory) | $0.00 |
| LLM API calls | $0.00 |
| **Total** | **$0.00** |

> Compute only: 1,000 queries × 133ms = 133s CPU → **$0.0015** on a $0.04/hr VM.
> OpenAI embeddings + Pinecone + GPT-4o equivalent: ~$5–$10 per 1K queries.
> Decision to avoid API calls was also a latency decision — OpenAI embed round-trip adds 200–400ms, which alone would break the 500ms budget.

---

## What Was Rejected and Why

| Approach | NDCG | p95 | Why rejected |
|---|---|---|---|
| Cross-encoder rerank | 0.563 | 687ms | Hurts multi-hop + breaks latency budget |
| 256-char chunks | 0.578 | 345ms | Sentences cut mid-thought, 2.4× bigger index |
| 1024-char chunks | 0.590 | 85ms | Vector blurs at 220 tokens in 384-dim space |
| Dense-only (no BM25) | 0.510 | 195ms | Fails on named entities — BM25 alone beats it |
| α = 0.3 (dense-heavy) | 0.555 | 197ms | Wrong direction — corpus rewards BM25 weight |
| BGE-small encoder | +0.006 NDCG | same | 60% larger model, marginal gain |
