#!/usr/bin/env python3
"""KMeans on customer texts. Review the printed examples; do not ship raw ids."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=Path("data/pairs_cluster.csv"))
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    rows = list(csv.DictReader(args.pairs.open(encoding="utf-8")))
    texts = [r["customer_text"] for r in rows]
    vec = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=3, stop_words="english")
    x = vec.fit_transform(texts)
    km = MiniBatchKMeans(n_clusters=args.k, random_state=7, batch_size=512, n_init=10)
    labels = km.fit_predict(x)
    terms = vec.get_feature_names_out()
    order = km.cluster_centers_.argsort()[:, ::-1]
    print(Counter(labels))
    for i in range(args.k):
        top = ", ".join(terms[j] for j in order[i, :12])
        print(f"\ncluster {i} n={(labels == i).sum()}\n  {top}")
        shown = 0
        for text, lab in zip(texts, labels):
            if lab == i:
                print("  -", text[:160])
                shown += 1
            if shown >= 4:
                break


if __name__ == "__main__":
    main()
