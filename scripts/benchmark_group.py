"""Reproducible K3 group retrieval benchmark on the UEH public corpus."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingest import build_knowledge_base
from src import (
    FixedSizeChunker,
    LocalEmbedder,
    MockEmbedder,
    PolicySectionChunker,
    RecursiveChunker,
    SentenceChunker,
)


@dataclass(frozen=True)
class BenchmarkCase:
    query: str
    expected_doc_id: str
    gold_answer: str
    metadata_filter: dict[str, str] | None = None


CASES = [
    BenchmarkCase(
        "Sau khi đăng ký học phần, sinh viên kiểm tra tổng học phí ở đâu?",
        "ueh-course-registration",
        "Chọn In phiếu đóng tiền để kiểm tra danh sách học phần và tổng học phí.",
    ),
    BenchmarkCase(
        "Sau khi đóng học phí, cần đối chiếu những thông tin nào trên Portal?",
        "ueh-tuition-payment",
        "Đối chiếu học phí đã nộp và thời khóa biểu đã được cập nhật.",
    ),
    BenchmarkCase(
        "Một cuốn sách thư viện UEH được mượn và gia hạn trong bao lâu?",
        "ueh-library-borrowing",
        "Mượn tiêu chuẩn 20 ngày và được gia hạn một lần thêm 20 ngày.",
    ),
    BenchmarkCase(
        "Những sinh viên nào được ưu tiên khi đăng ký ký túc xá UEH?",
        "ueh-dormitory-registration",
        "Có hộ nghèo/cận nghèo và người có thành tích nổi bật kèm minh chứng.",
    ),
    BenchmarkCase(
        "Học bổng hỗ trợ học tập mở đăng ký mấy lần mỗi năm?",
        "ueh-scholarship-policy",
        "Hai lần mỗi năm, vào học kỳ đầu và học kỳ cuối.",
        {"audience": "student"},
    ),
]


STRATEGIES = {
    "fixed_350_overlap_50": FixedSizeChunker(chunk_size=350, overlap=50),
    "sentence_2": SentenceChunker(max_sentences_per_chunk=2),
    "recursive_350": RecursiveChunker(chunk_size=350),
    "policy_section_350": PolicySectionChunker(chunk_size=350),
    "policy_section_500": PolicySectionChunker(chunk_size=500),
}


def run(embedding_fn: Callable[[str], list[float]]) -> dict:
    summary: dict = {}
    for strategy_name, chunker in STRATEGIES.items():
        store = build_knowledge_base(
            "data/k3_university",
            embedding_fn=embedding_fn,
            chunker=chunker,
            collection_name=f"group_{strategy_name}",
        )
        cases = []
        for case in CASES:
            if case.metadata_filter:
                results = store.search_with_filter(case.query, 3, case.metadata_filter)
            else:
                results = store.search(case.query, 3)
            retrieved_ids = [result["metadata"].get("doc_id") for result in results]
            rank = next(
                (index for index, doc_id in enumerate(retrieved_ids, start=1) if doc_id == case.expected_doc_id),
                None,
            )
            cases.append(
                {
                    "query": case.query,
                    "expected_doc_id": case.expected_doc_id,
                    "rank": rank,
                    "top_score": round(results[0]["score"], 6) if results else None,
                    "top_doc_id": retrieved_ids[0] if retrieved_ids else None,
                    "gold_answer": case.gold_answer,
                }
            )
        hits = sum(case["rank"] is not None for case in cases)
        reciprocal_rank = sum(1 / case["rank"] for case in cases if case["rank"]) / len(cases)
        summary[strategy_name] = {
            "chunk_count": store.get_collection_size(),
            "recall_at_3": hits / len(cases),
            "mrr": round(reciprocal_rank, 4),
            "cases": cases,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("local", "mock"), default="local")
    args = parser.parse_args()
    embedder = LocalEmbedder() if args.provider == "local" else MockEmbedder()
    output = {"embedding_backend": embedder._backend_name, "strategies": run(embedder)}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
