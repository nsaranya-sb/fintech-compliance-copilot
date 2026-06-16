"""Evaluation harness for the PCI DSS Compliance Copilot.

Reads test scenarios from eval/eval_scenarios.json, runs each through
the RAG pipeline, scores retrieval recall and classification accuracy,
prints a results table, and saves to a timestamped CSV.

Scoring uses hierarchy-aware matching: expected "3.5" is satisfied by
retrieved "3.5.1" (child) or vice versa (one is a dotted-prefix of the other).
Exact-match results are also logged for comparison.

Usage:
    python -m evals.run_eval
"""

import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load env vars before any src imports
load_dotenv(override=True)

from src.config import get_settings
from src.embeddings.embedding_service import EmbeddingService
from src.models import ComplianceQueryRequest
from src.rag.engine import RAGEngine
from src.vectorstore.chroma_store import ChromaVectorStore

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
SCENARIOS_PATH = Path("evals/eval_scenarios.json")
RESULTS_DIR = Path("evals/results")

# Classification normalization map
CLASSIFICATION_MAP = {
    "compliant": "Compliant",
    "warning": "Warning",
    "non-compliant": "Non-Compliant",
    "noncompliant": "Non-Compliant",
    "🟢 compliant": "Compliant",
    "🟡 warning": "Warning",
    "🔴 non-compliant": "Non-Compliant",
}


def normalize_classification(raw: str) -> str:
    """Normalize classification string for comparison."""
    return CLASSIFICATION_MAP.get(raw.lower().strip(), raw.strip())


def is_hierarchy_match(expected: str, retrieved: str) -> bool:
    """Check if two requirement numbers are in the same dotted-prefix family.

    Returns True if one is a prefix of the other in dotted notation.
    e.g., "3.5" matches "3.5.1", "3.5.1.1"; "3.6.1" matches "3.6.1.1".
    Also matches exact equality.

    Args:
        expected: The expected requirement number (e.g., "3.5").
        retrieved: A retrieved requirement number (e.g., "3.5.1").

    Returns:
        True if one is a dotted-prefix of the other.
    """
    if expected == retrieved:
        return True
    # Check if one is a prefix of the other with a dot boundary
    # "3.5" is a prefix of "3.5.1" but NOT of "3.55"
    if retrieved.startswith(expected + "."):
        return True
    if expected.startswith(retrieved + "."):
        return True
    return False


def score_retrieval(expected_reqs: set[str], retrieved_reqs: set[str]) -> dict:
    """Score retrieval with both exact and hierarchy-aware matching.

    Args:
        expected_reqs: Set of expected requirement numbers.
        retrieved_reqs: Set of retrieved requirement numbers.

    Returns:
        Dict with exact and hierarchy-aware hits, misses, and recall.
    """
    # Exact matching
    exact_hits = expected_reqs & retrieved_reqs
    exact_misses = expected_reqs - retrieved_reqs
    exact_recall = len(exact_hits) / len(expected_reqs) if expected_reqs else 1.0

    # Hierarchy-aware matching
    hierarchy_hits = set()
    hierarchy_misses = set()

    for exp_req in expected_reqs:
        found = False
        for ret_req in retrieved_reqs:
            if is_hierarchy_match(exp_req, ret_req):
                found = True
                break
        if found:
            hierarchy_hits.add(exp_req)
        else:
            hierarchy_misses.add(exp_req)

    hierarchy_recall = len(hierarchy_hits) / len(expected_reqs) if expected_reqs else 1.0

    return {
        "exact_hits": exact_hits,
        "exact_misses": exact_misses,
        "exact_recall": exact_recall,
        "hierarchy_hits": hierarchy_hits,
        "hierarchy_misses": hierarchy_misses,
        "hierarchy_recall": hierarchy_recall,
    }


def run_evaluation():
    """Run the full evaluation suite."""
    # Load scenarios
    if not SCENARIOS_PATH.exists():
        print(f"ERROR: Scenarios file not found: {SCENARIOS_PATH}")
        sys.exit(1)

    with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    print(f"\n{'='*80}")
    print(f"  PCI DSS Compliance Copilot — Evaluation Harness")
    print(f"  Scenarios: {len(scenarios)} | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    # Initialize pipeline
    settings = get_settings()

    if settings.OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    if settings.ANTHROPIC_API_KEY:
        os.environ["ANTHROPIC_API_KEY"] = settings.ANTHROPIC_API_KEY

    embedding_service = EmbeddingService(
        model=settings.EMBEDDING_MODEL,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        api_key=settings.OPENAI_API_KEY,
    )
    vector_store = ChromaVectorStore(
        persist_directory=settings.VECTORDB_PATH,
        collection_name=settings.COLLECTION_NAME,
    )
    rag_engine = RAGEngine(
        vector_store=vector_store,
        embedding_service=embedding_service,
        api_key=settings.ANTHROPIC_API_KEY,
        use_query_decomposition=settings.USE_QUERY_DECOMPOSITION,
    )

    # Run scenarios
    results = []

    for scenario in scenarios:
        scenario_id = scenario["id"]
        query_text = scenario["scenario"]
        expected_reqs = set(scenario["expected_requirements"])
        expected_class = normalize_classification(scenario["expected_classification"])

        print(f"  Running: {scenario_id}...", end=" ", flush=True)
        start = time.time()

        try:
            request = ComplianceQueryRequest(query=query_text)
            response = rag_engine.process_query(request)
            elapsed = time.time() - start

            # Extract requirement numbers from retrieved chunk IDs
            retrieved_reqs = set()
            for chunk_id in response.retrieved_chunk_ids:
                try:
                    result = vector_store._collection.get(
                        ids=[chunk_id],
                        include=["metadatas"],
                    )
                    if result["metadatas"] and result["metadatas"][0]:
                        req_num = result["metadatas"][0].get("requirement_number", "")
                        if req_num:
                            retrieved_reqs.add(req_num)
                except Exception:
                    pass

            # Score retrieval (exact + hierarchy-aware)
            scores = score_retrieval(expected_reqs, retrieved_reqs)

            # Use hierarchy-aware recall for pass/fail determination
            recall = scores["hierarchy_recall"]

            # Score classification
            actual_class = normalize_classification(response.risk_classification.value)
            class_match = actual_class == expected_class

            # Determine pass/fail (hierarchy recall >= 0.5 AND classification match)
            passed = recall >= 0.5 and class_match

            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} ({elapsed:.1f}s)")

            results.append({
                "id": scenario_id,
                "hierarchy_recall_pct": round(scores["hierarchy_recall"] * 100, 1),
                "exact_recall_pct": round(scores["exact_recall"] * 100, 1),
                "hierarchy_hits": sorted(scores["hierarchy_hits"]),
                "hierarchy_misses": sorted(scores["hierarchy_misses"]),
                "exact_hits": sorted(scores["exact_hits"]),
                "exact_misses": sorted(scores["exact_misses"]),
                "retrieved_reqs": sorted(retrieved_reqs),
                "expected_class": expected_class,
                "actual_class": actual_class,
                "class_match": class_match,
                "passed": passed,
                "elapsed_s": round(elapsed, 1),
                "error": None,
            })

        except Exception as e:
            elapsed = time.time() - start
            print(f"💥 ERROR ({elapsed:.1f}s): {str(e)[:80]}")
            results.append({
                "id": scenario_id,
                "hierarchy_recall_pct": 0.0,
                "exact_recall_pct": 0.0,
                "hierarchy_hits": [],
                "hierarchy_misses": sorted(expected_reqs),
                "exact_hits": [],
                "exact_misses": sorted(expected_reqs),
                "retrieved_reqs": [],
                "expected_class": expected_class,
                "actual_class": "ERROR",
                "class_match": False,
                "passed": False,
                "elapsed_s": round(elapsed, 1),
                "error": str(e)[:200],
            })

    # Print results table
    print(f"\n{'='*80}")
    print(f"  RESULTS (hierarchy-aware recall | exact recall)")
    print(f"{'='*80}")
    print(
        f"  {'ID':<32} {'H-Recall':>8} {'E-Recall':>8} "
        f"{'H-Hits':>8} {'Expected':>15} {'Actual':>15} {'Status':>6}"
    )
    print(f"  {'-'*32} {'-'*8} {'-'*8} {'-'*8} {'-'*15} {'-'*15} {'-'*6}")

    for r in results:
        total_expected = len(r["hierarchy_hits"]) + len(r["hierarchy_misses"])
        hits_str = f"{len(r['hierarchy_hits'])}/{total_expected}"
        print(
            f"  {r['id']:<32} {r['hierarchy_recall_pct']:>7.1f}% {r['exact_recall_pct']:>7.1f}% "
            f"{hits_str:>8} {r['expected_class']:>15} {r['actual_class']:>15} "
            f"{'PASS' if r['passed'] else 'FAIL':>6}"
        )
        # Log hierarchy vs exact differences
        if r["hierarchy_hits"] != r.get("exact_hits", []):
            extra = set(r["hierarchy_hits"]) - set(r["exact_hits"])
            if extra:
                print(f"    ↳ hierarchy bonus: {sorted(extra)} (matched via prefix)")

    # Aggregate metrics
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    avg_hierarchy_recall = sum(r["hierarchy_recall_pct"] for r in results) / total if total else 0
    avg_exact_recall = sum(r["exact_recall_pct"] for r in results) / total if total else 0
    class_accuracy = sum(1 for r in results if r["class_match"]) / total if total else 0
    errors = sum(1 for r in results if r["error"])

    print(f"\n  {'─'*55}")
    print(f"  Hierarchy-Aware Retrieval Recall:  {avg_hierarchy_recall:.1f}%")
    print(f"  Exact-Match Retrieval Recall:      {avg_exact_recall:.1f}%")
    print(f"  Classification Accuracy:           {class_accuracy*100:.1f}%")
    print(f"  Passed:                            {passed_count}/{total}")
    print(f"  Errors:                            {errors}/{total}")
    print(f"{'='*80}\n")

    # Save to CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"eval_{timestamp}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "hierarchy_recall_pct", "exact_recall_pct",
                "hierarchy_hits", "hierarchy_misses",
                "exact_hits", "exact_misses", "retrieved_reqs",
                "expected_class", "actual_class", "class_match", "passed",
                "elapsed_s", "error",
            ],
        )
        writer.writeheader()
        for r in results:
            row = r.copy()
            row["hierarchy_hits"] = "|".join(row["hierarchy_hits"])
            row["hierarchy_misses"] = "|".join(row["hierarchy_misses"])
            row["exact_hits"] = "|".join(row["exact_hits"])
            row["exact_misses"] = "|".join(row["exact_misses"])
            row["retrieved_reqs"] = "|".join(row["retrieved_reqs"])
            writer.writerow(row)

    print(f"  Results saved to: {csv_path}")


if __name__ == "__main__":
    run_evaluation()
