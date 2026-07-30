"""Retrieval eval: hit-rate@k, refusal accuracy, and an optional threshold sweep.

Usage:
    python -m eval.run_eval
    python -m eval.run_eval --sweep
    python -m eval.run_eval --sweep --thresholds 0.3,0.4,0.5,0.6
    python -m eval.run_eval --skip-ingest   # reuse already-ingested fixture documents
"""

import argparse

from app.config import get_settings
from app.embeddings.bge import BGEEmbeddingProvider
from app.ingestion.chunker import Chunker
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.retriever import Retriever
from app.vectorstore.pinecone_store import PineconeVectorStore
from eval.harness import (
    best_threshold,
    collect_results,
    hit_rate_at_k,
    ingest_fixtures,
    refusal_accuracy,
    sweep_thresholds,
)

DEFAULT_TENANT = "eval-fixtures"
DEFAULT_THRESHOLD_CANDIDATES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def _build_components():
    settings = get_settings()
    embedder = BGEEmbeddingProvider(
        model_name=settings.embedding_model_name,
        device=settings.embedding_device,
        query_instruction=settings.embedding_query_instruction,
    )
    chunker = Chunker(
        tokenizer_name=settings.embedding_model_name,
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )
    vector_store = PineconeVectorStore(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
        dimension=embedder.dimension,
    )
    ingestion_pipeline = IngestionPipeline(chunker=chunker, embedder=embedder, vector_store=vector_store)
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=settings.top_k,
        score_threshold=settings.similarity_threshold,
    )
    return settings, ingestion_pipeline, retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tenant", default=DEFAULT_TENANT, help=f"eval tenant namespace (default: {DEFAULT_TENANT})")
    parser.add_argument("--top-k", type=int, default=None, help="defaults to TOP_K from settings")
    parser.add_argument("--sweep", action="store_true", help="sweep candidate thresholds instead of the configured one")
    parser.add_argument("--thresholds", default=None, help="comma-separated candidate thresholds for --sweep")
    parser.add_argument("--skip-ingest", action="store_true", help="reuse already-ingested fixture documents")
    args = parser.parse_args()

    settings, ingestion_pipeline, retriever = _build_components()
    top_k = args.top_k or settings.top_k

    if not args.skip_ingest:
        print(f"Ingesting fixture documents into tenant {args.tenant!r}...")
        ingest_fixtures(args.tenant, ingestion_pipeline)

    results = collect_results(args.tenant, retriever, top_k)
    print(f"\nRan retrieval for {len(results)} fixture questions (top_k={top_k}).\n")

    for result in results:
        top_score = result.matches[0].score if result.matches else None
        expected = result.expected_filename or "(out-of-scope, expect refusal)"
        score_display = f"{top_score:.3f}" if top_score is not None else "n/a"
        print(f"  [{expected:35s}] top_score={score_display:>6s}  {result.question}")

    hit_rate = hit_rate_at_k(results)
    print(f"\nhit-rate@{top_k}: {hit_rate:.1%}")

    if args.sweep:
        candidates = [float(t) for t in args.thresholds.split(",")] if args.thresholds else DEFAULT_THRESHOLD_CANDIDATES
        sweep_report = sweep_thresholds(results, candidates)
        print("\nThreshold sweep (refusal accuracy):")
        for row in sweep_report:
            print(f"  threshold={row['threshold']:.2f}  ->  refusal_accuracy={row['refusal_accuracy']:.1%}")
        chosen = best_threshold(sweep_report)
        print(
            f"\nRecommended SIMILARITY_THRESHOLD from data: {chosen:.2f} "
            f"(currently configured: {settings.similarity_threshold})"
        )
    else:
        accuracy = refusal_accuracy(results, settings.similarity_threshold)
        print(f"refusal accuracy @ threshold={settings.similarity_threshold}: {accuracy:.1%}")


if __name__ == "__main__":
    main()
