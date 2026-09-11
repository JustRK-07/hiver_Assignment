# Data

`pairs_index.csv` / `pairs_cluster.csv` are seeded subsamples of British Airways threads from the Kaggle Customer Support on Twitter corpus (`thoughtvector/customer-support-on-twitter`).

Rebuild from a local `twcs.csv`:

```bash
PYTHONPATH=. python data/extract_brand.py --src /path/to/twcs.csv
```

Do not commit the 517MB full dump.
