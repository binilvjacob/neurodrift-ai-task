---
title: Multi-tenant RAG Pipeline
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Multi-tenant RAG Pipeline

A Retrieval-Augmented Generation pipeline that answers questions from a per-tenant knowledge
base, refuses to answer when it isn't confident the answer is in that knowledge base, and always
returns the source chunks it based its answer on.

"Tenant" is a generic stand-in for whatever multi-tenant unit you have — a hotel, a customer, a
workspace. Nothing in the code is hotel-specific.

## How it works

1. **Ingest** — upload a PDF/DOCX/TXT/XLSX file for a tenant. It's split into structural segments
   (pages, paragraphs, rows) with their location preserved, chunked by token count, embedded, and
   upserted into that tenant's own Pinecone namespace.
2. **Query** — a question is embedded and matched against that tenant's namespace only. If the top
   match's similarity score is below `SIMILARITY_THRESHOLD`, the pipeline refuses to answer and
   never calls the LLM. Otherwise the matched chunks are sent to the LLM as context, and the
   response always includes those chunks with their scores and locations.

## Setup

Requires Python 3.11+ (tested on 3.12; torch has no macOS x86_64 wheels beyond 2.2.2, see
`requirements.txt`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env`:

| Variable | Description |
|---|---|
| `EMBEDDING_MODEL_NAME` | HF sentence-transformers model id (default `BAAI/bge-small-en-v1.5`, 384-dim) |
| `EMBEDDING_DEVICE` | `cpu` or `cuda` |
| `EMBEDDING_QUERY_INSTRUCTION` | BGE's required query prefix — applied at query time only, never at indexing |
| `PINECONE_API_KEY` | required |
| `PINECONE_INDEX_NAME` | created automatically on first run if it doesn't exist |
| `PINECONE_CLOUD` / `PINECONE_REGION` | serverless spec, only used if the index needs creating |
| `HF_TOKEN` | Hugging Face token for the Inference Providers router |
| `HF_LLM_MODEL` | e.g. `openai/gpt-oss-20b` — any chat-completions-compatible model on the router |
| `HF_LLM_PROVIDER` | optional provider suffix (`model:provider`), leave blank for the router's default |
| `HF_BASE_URL` | `https://router.huggingface.co/v1` |
| `TOP_K` | chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | refusal gate — see `eval/` below for how this was chosen |
| `CHUNK_SIZE_TOKENS` / `CHUNK_OVERLAP_TOKENS` | validated at startup against the embedding model's max sequence length; the process refuses to start if the chunk size doesn't fit |

Run the server:

```bash
uvicorn app.main:app --reload
```

Startup eagerly builds the embedder, chunker, Pinecone client, and LLM client before accepting any
requests — a bad API key or an oversized `CHUNK_SIZE_TOKENS` crashes the process immediately
instead of surfacing on a user's first request. This means startup takes on the order of tens of
seconds (loading the embedding model, connecting to Pinecone) rather than being instant.

Interactive API docs: `http://127.0.0.1:8000/docs`.

## API

- `POST /tenants/{tenant_id}/documents` — multipart file upload (`.pdf`, `.docx`, `.txt`, `.xlsx`).
  `document_id` is a content hash, so re-uploading identical bytes is idempotent.
- `POST /tenants/{tenant_id}/query` — `{"question": "...", "top_k": 5}` (top_k optional). Returns
  the answer, a `grounded` flag, `sources` (chunk id, filename, text, score, location), and a
  `refusal_reason` when ungrounded.
- `GET /health`

`tenant_id` must match `^[a-zA-Z0-9_-]{1,64}$`, enforced both at the API layer and again inside the
vector store before it ever becomes a Pinecone namespace.

## Tests

```bash
pytest                    # unit tests + integration test (if credentials are set)
pytest -m "not integration"   # unit tests only, no live Pinecone calls
```

- `tests/test_tenant_isolation.py` — tenant isolation proven against an in-memory fake vector
  store and embedder, no external services required.
- `tests/test_pinecone_integration.py` — the same isolation guarantee proven against two real
  Pinecone namespaces. Marked `@pytest.mark.integration`; skips automatically (rather than
  failing) when `PINECONE_API_KEY`/`PINECONE_INDEX_NAME` aren't set. Cleans up its own namespaces
  after running.

## Eval harness

```bash
python -m eval.run_eval            # hit-rate@k and refusal accuracy at the configured threshold
python -m eval.run_eval --sweep    # + sweeps candidate thresholds and recommends one from data
```

Fixtures reuse the sample documents in `scratch/` (a hotel policy PDF, a room rates spreadsheet):
10 in-scope questions with an expected source document, and 6 out-of-scope questions expected to
be refused. `hit-rate@k` is threshold-independent (did the right document appear in the raw top-k
results); the sweep re-applies the refusal gate to those same cached results at each candidate
threshold, so it costs one retrieval pass per question regardless of how many thresholds are
tried.

The last sweep run recommended `SIMILARITY_THRESHOLD=0.50` from a 0.20–0.70 candidate range,
matching the configured default — i.e. the default is corroborated by data, not just guessed. Two
honest caveats: the fixture set is small (2 documents, reusing existing test data — a real
deployment should eval against its actual corpus), and Pinecone serverless scores can drift
slightly between a query issued immediately after ingestion and one issued once the index has
settled, which is a property of the vector store, not of this code.

## Deliberate limitations (out of scope)

Auth, rate limiting, async ingestion jobs, reranking, and a custom UI were intentionally not
built. `/docs` is the interface. These are gaps to be aware of before any real deployment, not
oversights:

- **No auth** — every tenant's endpoints are open to anyone who can reach the server.
- **No rate limiting** — nothing prevents abuse of the LLM or embedding calls.
- **Synchronous ingestion** — large files block the request until fully chunked, embedded, and
  upserted (acceptable for this scope; route handlers are sync `def`s that FastAPI runs in its
  threadpool, so the process itself stays responsive to other requests).
- **No reranking** — retrieval is a single dense-vector similarity search; no second-stage
  reranker on top of the initial top-k.
- **No UI** — FastAPI's generated `/docs` page is the only interface.

## Logging

Every ingest and query request emits one structured JSON log line (`tenant_id`, `question` or
`document_id`/filename, chunk ids with scores, the `grounded` flag, and latency in ms), on a
dedicated `rag` logger namespace kept separate from uvicorn's own logging config.

## Deploying (Hugging Face Spaces)

The `Dockerfile` builds a self-contained image (the embedding model and tokenizer are
downloaded once at *build* time, not on every cold start) and listens on port 7860, matching
Spaces' Docker SDK default. This repo's `README.md` frontmatter (the block at the very top) is
already configured for `sdk: docker`.

1. Create a new Space at huggingface.co/new-space, SDK = **Docker**.
2. Push this repo to the Space's git remote (`git remote add space <space-git-url>`, then
   `git push space develop:main`).
3. In the Space's **Settings → Variables and secrets**, add every variable from `.env.example`
   as a secret — `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `HF_TOKEN`, etc.
4. The Space builds and boots automatically. `/docs` is reachable at the Space's public URL once
   it's up.

Two things worth knowing: free-tier Spaces sleep after inactivity, so the first request after a
period of idleness pays both the Space's own wake-up time and this app's ~20–30s eager startup
(building the embedder, chunker, Pinecone client, and LLM client before accepting traffic — see
above). And there's no auth in front of any of this (see "Deliberate limitations"), so a publicly
reachable Space is reachable by anyone with the URL, not just intended evaluators.
