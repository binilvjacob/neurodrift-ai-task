from dataclasses import dataclass

from app.ingestion.pipeline import IngestionPipeline
from app.models.schemas import SourceChunk
from app.retrieval.retriever import Retriever, passes_threshold
from eval.fixtures import DOCUMENTS, IN_SCOPE_QUESTIONS, OUT_OF_SCOPE_QUESTIONS


@dataclass
class QuestionResult:
    question: str
    expected_filename: str | None  # None marks an out-of-scope question (refusal expected)
    matches: list[SourceChunk]


def ingest_fixtures(tenant_id: str, ingestion_pipeline: IngestionPipeline) -> None:
    for doc in DOCUMENTS:
        ingestion_pipeline.ingest_file(tenant_id, doc["path"], doc["path"].name, doc["source_type"])


def collect_results(tenant_id: str, retriever: Retriever, top_k: int) -> list[QuestionResult]:
    """Runs retrieval once per fixture question (in-scope and out-of-scope) and caches the
    raw ranked matches, so hit-rate@k and a threshold sweep can both be computed afterward
    without re-embedding or re-querying the vector store.
    """
    results = []
    for item in IN_SCOPE_QUESTIONS:
        matches = retriever.search(tenant_id, item["question"], top_k)
        results.append(QuestionResult(item["question"], item["expected_filename"], matches))
    for item in OUT_OF_SCOPE_QUESTIONS:
        matches = retriever.search(tenant_id, item["question"], top_k)
        results.append(QuestionResult(item["question"], None, matches))
    return results


def hit_rate_at_k(results: list[QuestionResult]) -> float:
    """Fraction of in-scope questions whose expected document appears among the raw top-k
    matches. Threshold-independent by design — this measures retrieval quality alone.
    """
    in_scope = [r for r in results if r.expected_filename is not None]
    if not in_scope:
        return 0.0
    hits = sum(1 for r in in_scope if any(m.filename == r.expected_filename for m in r.matches))
    return hits / len(in_scope)


def refusal_accuracy(results: list[QuestionResult], threshold: float) -> float:
    """Fraction of ALL fixture questions where the grounded/refused decision at this
    threshold matches what was expected (in-scope -> grounded, out-of-scope -> refused).
    """
    if not results:
        return 0.0
    correct = sum(
        1
        for r in results
        if passes_threshold(r.matches, threshold) == (r.expected_filename is not None)
    )
    return correct / len(results)


def sweep_thresholds(results: list[QuestionResult], candidates: list[float]) -> list[dict]:
    return [{"threshold": t, "refusal_accuracy": refusal_accuracy(results, t)} for t in candidates]


def best_threshold(sweep_report: list[dict]) -> float:
    """Picks the median of the thresholds tied for the highest refusal accuracy, rather
    than an arbitrary extreme, so the recommendation is robust to small fixture-set noise.
    """
    best_accuracy = max(row["refusal_accuracy"] for row in sweep_report)
    tied = sorted(row["threshold"] for row in sweep_report if row["refusal_accuracy"] == best_accuracy)
    return tied[len(tied) // 2]
