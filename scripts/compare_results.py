import json
import sys

from scipy.stats import spearmanr

DOC_ONLY = False


def load_results(fn, k):
    query_to_res = {}
    query_to_gold = {}
    query_to_res_to_rank = {}
    with open(fn) as f:
        for line in f:
            d = json.loads(line)
            q = d["query"]
            query_to_res[q] = []
            query_to_res_to_rank[q] = {}
            for i, res in enumerate(d["results"][:k]):
                if DOC_ONLY:
                    chunk = res["document_id"]
                else:
                    chunk = (res["document_id"], res["chunk_number"])
                query_to_res[q].append(chunk)
                query_to_res_to_rank[q][chunk] = i + 1
            query_to_gold[q] = (d["gold_paper"], d["gold_chunk"])

    return query_to_res, query_to_gold, query_to_res_to_rank


def compare_results(res1, res2):
    ious = []
    global_i = 0
    global_u = 0
    for q, chunks1 in res1.items():
        if q not in res2:
            print('WARNING: query "{q}" in file 1 missing from file 2, skipping')
            continue
        set1 = set(chunks1)
        set2 = set(res2[q])
        i = len(set1 & set2)
        u = len(set1 | set2)
        ious.append(i / u)
        global_i += i
        global_u += u

    print(f"total queries: {len(ious)}")
    print(f"global IOU: {global_i} / {global_u} = {global_i / global_u:.2f}")
    print(f"average IOU: {sum(ious) / len(ious):.2f}")


def compare_rankings(r_to_rank1, r_to_rank2):
    ranks1, ranks2, overlaps = [], [], []
    for q, r_to_r1 in r_to_rank1.items():
        if q not in r_to_rank2:
            print('WARNING: query "{q}" in file 1 missing from file 2, skipping')
            continue
        overlap = 0
        for chunk, rank in r_to_r1.items():
            if chunk in r_to_rank2[q]:
                overlap += 1
                ranks1.append(rank)
                ranks2.append(r_to_rank2[q][chunk])

        overlaps.append(overlap)

    print(f"avg overlap: {sum(overlaps) / len(overlaps):.2f}")
    spearman, _ = spearmanr(ranks1, ranks2)
    print(f"Spearman correlation: {spearman:.2f}")


def main():
    k = int(sys.argv[3])
    res1, gold1, r_to_rank1 = load_results(sys.argv[1], k)
    res2, gold2, r_to_rank2 = load_results(sys.argv[2], k)
    assert gold1 == gold2, "gold annotation doesn't match"
    compare_rankings(r_to_rank1, r_to_rank2)
    compare_results(res1, res2)


if __name__ == "__main__":
    main()
