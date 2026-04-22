from collections import Counter

from tabulate import tabulate


def get_stats_from_results(results: list[dict]) -> Counter[str]:
    stats: Counter[str] = Counter()
    gold_papers, gold_chunks = set(), set()
    for res in results:
        gold_papers.add(res["gold_paper"])
        gold_chunks.add(f"{res['gold_paper']}_{res['gold_chunk']}")
        stats["queries"] += 1
        if res["paper_found"]:
            rank = res["corr_paper_rank"]
            stats[f"corr_paper@{rank}"] += 1
        if res["chunk_found"]:
            rank = res["corr_chunk_rank"]
            stats[f"corr_chunk@{rank}"] += 1

    stats["gold_papers"] = len(gold_papers)
    stats["gold_chunks"] = len(gold_chunks)
    return stats


def print_overall_stats(stats: Counter[str], args) -> None:
    rows = []
    print(
        f"Results for {args.output_file=}, {args.query_field=}, {args.search_type=}, "
        f"HYBRID_WEIGHTS={getattr(args, 'hybrid_weights', {'dense': 0.3, 'full_text': 0.7})}, {args.rerank=}"
    )
    print(f"Total queries: {stats['queries']}\n")
    for i in range(1, args.k + 1):
        stats[f"total_corr_paper@{i}"] = (
            stats[f"total_corr_paper@{i - 1}"] + stats[f"corr_paper@{i}"]
        )
        paper_recall_at_i = stats[f"total_corr_paper@{i}"] / stats["queries"]

        stats[f"total_corr_chunk@{i}"] = (
            stats[f"total_corr_chunk@{i - 1}"] + stats[f"corr_chunk@{i}"]
        )
        chunk_recall_at_i = stats[f"total_corr_chunk@{i}"] / stats["queries"]

        if i in (1, 3, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
            rows.append(
                [f"{i}", f"{paper_recall_at_i:.2%}", f"{chunk_recall_at_i:.2%}"]
            )

    print(tabulate(rows, headers=["k", "paper R @ k", "chunk R @ k"]))
