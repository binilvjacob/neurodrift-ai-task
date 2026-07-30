# CLAUDE.md

Multi-tenant RAG pipeline — take-home task. This file exists so context survives a lost session.

## Original brief

Build a RAG pipeline: document ingestion (PDF/DOCX/TXT/XLSX) -> chunking -> HF embeddings ->
Pinecone/Qdrant storage with per-tenant ("hotel") namespaces -> retrieval -> HF LLM answer
grounded in retrieved context. Config via env vars, modular code, any open HF models.

## Decisions already made — do not re-litigate

- Python 3.11, FastAPI, pydantic-settings for config.
- Embeddings: `BAAI/bge-small-en-v1.5` via sentence-transformers, local CPU, 384-dim. Query
  instruction prefix applied at retrieval time only (`embed_query`), never at indexing
  (`embed_documents`) — this is a BGE-specific requirement, not a stylistic choice.
- Vector store: Pinecone serverless, single index, one namespace per `tenant_id`. "Hotel" in the
  brief is a generic tenant — nothing hotel-specific in the code.
- LLM: open instruct model via the Hugging Face router (`router.huggingface.co/v1`),
  OpenAI-compatible client, behind an `LLMProvider` ABC so the model swaps via env var alone
  (`hf_llm_model`, optional `hf_llm_provider` suffix).
- All config through environment variables — no hardcoded keys, model names, or index names.
- Provider interfaces (ABCs) for embeddings (`EmbeddingProvider`), vector store (`VectorStore`),
  and LLM (`LLMProvider`), each with exactly one production implementation today.

## Non-negotiable requirements

- **Similarity threshold gate**: if the top match scores below `SIMILARITY_THRESHOLD`, return a
  refusal WITHOUT calling the LLM at all. Implemented in `Retriever.retrieve` — checks
  `raw_matches[0].score` before building any source list or touching the LLM.
- Every query response returns source chunks with scores and metadata (`SourceChunk`), even on
  the refusal path (empty list).
- **Tenant isolation test** against an in-memory fake (not yet written), plus an **integration
  test** against real Pinecone with two namespaces, behind a `@pytest.mark.integration` marker
  that skips without credentials (marker declared in `pyproject.toml`, no test file yet).
- **Eval harness**: fixtures with in-scope questions (expected source docs) and out-of-scope
  questions (expected refusal). Reports hit-rate@k and refusal accuracy, plus a sweep mode across
  candidate thresholds so `SIMILARITY_THRESHOLD` is chosen from data, not guessed. Not yet built.
- Loaders preserve location, threaded through to source chunks via `Locator`:
  PDF page number, XLSX sheet + row range, DOCX nearest heading, TXT line range.
- Excel serializes **row-wise** with the header repeated per row (`col: val | col: val`) — never
  flattened into prose. See `load_excel` in `app/ingestion/loaders.py`.
- Chunk size defined in tokens (`CHUNK_SIZE_TOKENS`), validated against bge-small's 512-token
  limit. Silent truncation is unacceptable — `Chunker.__init__` raises if `chunk_size_tokens`
  exceeds the tokenizer's `model_max_length`. **Known gap**: this check only fires on first use
  (lazy `@lru_cache` dependency), not at process boot — needs a FastAPI startup hook to be a true
  startup-time check. See "Known gaps" below.
- Route handlers are `def`, not `async def` — embedding, Pinecone, and LLM calls are all
  blocking; FastAPI threadpools them. All three routes (`health`, `ingest`, `query`) follow this.
- `document_id` is `sha256(file_bytes)[:16]` — re-ingestion of identical bytes is idempotent by
  construction (same document_id -> same chunk_ids -> same vector ids -> upsert overwrites in
  place). `tenant_id` is regex-validated (`^[a-zA-Z0-9_-]{1,64}$`, see `TENANT_ID_PATTERN` in
  `app/config.py`) both at the FastAPI path-param layer and again inside
  `PineconeVectorStore` before it ever reaches a namespace string.
- Structured per-request logging: tenant, question, chunk ids with scores, grounded flag,
  latency. **Not yet implemented — zero logging calls exist in `app/` currently.**

## Out of scope — do not build

Auth, rate limiting, async ingestion jobs, reranking, custom UI. FastAPI `/docs` is the
interface. These are documented as deliberate limitations in the README (not yet written).

## Implementation status (as of last review)

### Done and correct
- `app/config.py` — env-driven settings, tenant regex.
- `app/ingestion/loaders.py` — all four source types, locators preserved per format.
- `app/ingestion/chunker.py` — token-window chunking on the embedding model's own tokenizer,
  offset-mapping-based locator merging across chunk boundaries.
- `app/embeddings/bge.py` — query prefix only on `embed_query`.
- `app/vectorstore/pinecone_store.py` — namespace-per-tenant, tenant_id re-validated internally,
  batched upsert, `delete_document` by id-prefix (implemented but no route calls it yet — fine,
  not required by the brief).
- `app/ingestion/pipeline.py` — content-hash `document_id`, builds Pinecone metadata with the
  `Locator` JSON-serialized (Pinecone metadata must be flat).
- `app/retrieval/retriever.py` — threshold gate exactly as specified.
- `app/llm/hf_inference.py` — HF router via OpenAI-compatible client.
- `app/rag/pipeline.py` — refusal path short-circuits before the LLM; grounded path always
  attaches sources.
- `app/api/routes_ingest.py`, `routes_query.py` — sync `def` handlers, tenant_id path validation.
- `scripts/verify_chunking.py` + `scripts/_make_test_fixtures.py` — manual smoke-test tooling
  (not automated tests); fixtures live in `scratch/` (`hotel_policy.pdf`, `room_rates.xlsx`).

### Missing / stubbed
- `tests/` — only an empty `__init__.py`. No tenant-isolation test, no Pinecone integration test,
  no `conftest.py`.
- `eval/` — only an empty `__init__.py`. No fixtures, no hit-rate@k, no refusal-accuracy report,
  no threshold sweep.
- No structured logging anywhere in `app/`.
- No README.
- No `.env.example`.
- No FastAPI startup hook to eagerly validate config (chunk size vs tokenizer limit, Pinecone
  creds, HF token) before accepting traffic — currently these only surface on first request.

## Known repo-hygiene issue (fix before adding real credentials)

There is no `.gitignore`, and `.env` is tracked in git and pushed to `origin/develop`. As of the
last check, `PINECONE_API_KEY` and `HF_TOKEN` are both empty in every commit so far — nothing has
actually leaked. But the file will leak real keys the moment they're filled in and committed.
Fix: add `.gitignore` (must include `.env`, `__pycache__/`, `*.pyc`), `git rm --cached .env`, add
`.env.example` with placeholder values, commit before filling in real credentials.
(`__pycache__/*.pyc` files are also currently tracked for every module — same root cause.)

## Remaining work plan (ordered, ~12h budget, demoable-first)

1. Git hygiene: `.gitignore`, untrack `.env`, add `.env.example`. (0.5h)
2. Live smoke test with real Pinecone + HF credentials: upload `scratch/` fixtures, run one
   in-scope and one out-of-scope query via `/docs`. Do this early to surface integration
   surprises immediately. (0.5h)
3. FastAPI startup hook to eagerly construct the dependency graph (fail fast on bad config). (1h)
4. Structured per-request logging (tenant, question, chunk ids+scores, grounded flag, latency). (1.5h)
5. Tenant-isolation test against an in-memory fake `EmbeddingProvider`/`VectorStore`. (2h)
6. Pinecone integration test, two namespaces, `@pytest.mark.integration`, skips without creds. (1.5h)
7. Eval harness: in-scope + out-of-scope fixtures, hit-rate@k, refusal accuracy, threshold sweep. (3h)
8. README: setup, env vars, how to run tests/eval, documented out-of-scope limitations. (1h)
9. Buffer: rerun eval with chosen threshold, final end-to-end pass. (~1h)

If time runs short, cut step 6 (integration test) before any other — tenant isolation is still
proven by the in-memory fake test in step 5, so no non-negotiable requirement is dropped.

## Environment variables (see `app/config.py` for defaults)

`EMBEDDING_MODEL_NAME`, `EMBEDDING_DEVICE`, `EMBEDDING_QUERY_INSTRUCTION`,
`PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_CLOUD`, `PINECONE_REGION`,
`HF_TOKEN`, `HF_LLM_MODEL`, `HF_LLM_PROVIDER`, `HF_BASE_URL`, `HF_REQUEST_TIMEOUT_S`,
`TOP_K`, `SIMILARITY_THRESHOLD`, `CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS`.
