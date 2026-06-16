"""Evaluation harness for the PCI DSS Compliance Copilot.

Reads test scenarios from eval/eval_scenarios.json, runs each through
the RAG pipeline, scores retrieval recall and classification accuracy,
prints a results table, and saves to a timestamped CSV.

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


def extract_requirement_numbers(chunk_ids_or_chunks, engine_response) -> set[str]:
    """Extract requirement numbers from the response's retrieved chunk IDs.

    Parses requirement numbers from chunk IDs or from the chunks themselves
    via the vector store.
    """
    # The response has retrieved_chunk_ids — we need to map back to requirement numbers
    # We'll get them from the engine's last retrieval instead
    return set()


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
            # Chunk IDs have format: "file.pdf::chunkN" — we need the actual req numbers
            # We'll query the vector store to get chunk metadata
            retrieved_reqs = set()
            for chunk_id in response.retrieved_chunk_ids:
                # Parse from the chunks that were used — we need to get them from the store
                # The chunk IDs are in the response; look them up
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

            # Score retrieval recall
            hits = expected_reqs & retrieved_reqs
            misses = expected_reqs - retrieved_reqs
            recall = len(hits) / len(expected_reqs) if expected_reqs else 1.0

            # Score classification
            actual_class = normalize_classification(response.risk_classification.value)
            class_match = actual_class == expected_class

            # Determine pass/fail (recall >= 0.5 AND classification match)
            passed = recall >= 0.5 and class_match

            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} ({elapsed:.1f}s)")

            results.append({
                "id": scenario_id,
                "recall_pct": round(recall * 100, 1),
                "hits": sorted(hits),
                "misses": sorted(misses),
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
                "recall_pct": 0.0,
                "hits": [],
                "misses": sorted(expected_reqs),
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
    print(f"  RESULTS")
    print(f"{'='*80}")
    print(f"  {'ID':<35} {'Recall':>7} {'Hits/Miss':>12} {'Expected':>15} {'Actual':>15} {'Status':>8}")
    print(f"  {'-'*35} {'-'*7} {'-'*12} {'-'*15} {'-'*15} {'-'*8}")

    for r in results:
        hits_str = f"{len(r['hits'])}/{len(r['hits'])+len(r['misses'])}"
        print(
            f"  {r['id']:<35} {r['recall_pct']:>6.1f}% {hits_str:>12} "
            f"{r['expected_class']:>15} {r['actual_class']:>15} "
            f"{'PASS' if r['passed'] else 'FAIL':>8}"
        )

    # Aggregate metrics
    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    avg_recall = sum(r["recall_pct"] for r in results) / total if total else 0
    class_accuracy = sum(1 for r in results if r["class_match"]) / total if total else 0
    errors = sum(1 for r in results if r["error"])

    print(f"\n  {'─'*50}")
    print(f"  Overall Retrieval Recall:    {avg_recall:.1f}%")
    print(f"  Classification Accuracy:     {class_accuracy*100:.1f}%")
    print(f"  Passed:                      {passed_count}/{total}")
    print(f"  Errors:                      {errors}/{total}")
    print(f"{'='*80}\n")

    # Save to CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"eval_{timestamp}.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id", "recall_pct", "hits", "misses", "retrieved_reqs",
                "expected_class", "actual_class", "class_match", "passed",
                "elapsed_s", "error",
            ],
        )
        writer.writeheader()
        for r in results:
            row = r.copy()
            row["hits"] = "|".join(row["hits"])
            row["misses"] = "|".join(row["misses"])
            row["retrieved_reqs"] = "|".join(row["retrieved_reqs"])
            writer.writerow(row)

    print(f"  Results saved to: {csv_path}")


if __name__ == "__main__":
    run_evaluation()
