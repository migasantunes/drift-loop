# Dataset

## Choice: Yambda-50M

[Yambda](https://huggingface.co/datasets/yandex/yambda) is a music-streaming
interaction dataset published by Yandex in 2025. drift-loop uses its smallest
version.

| | Yambda-50M (from the dataset card, checked 16 Nov 2026) |
|---|---|
| Users | 10,000 |
| Items | 934,057 |
| Listens | 46,467,212 |
| Likes | 881,456 |
| Dislikes | 107,776 |

Why this one:

- **Time is intact.** Timestamps have a single global ordering, which the replay needs.
- **Long enough to drift.** Yandex describes the data as collected over roughly ten
  months. The dataset card does not state the span; M1 measures it.
- **Catalogue churn.** Over 900 thousand tracks for 10 thousand users means new
  items keep arriving, which is the kind of drift a recommender really faces.
- **Audio embeddings** for most tracks give the M2 item tower something to use for
  tracks with no interactions.
- **Organic flag.** Each event says whether it came from the platform's recommender
  or from the user's own navigation.
- **Fits on a laptop.**

Considered and rejected: OTTO (about five weeks, too short to see drift), MovieLens
(ratings entered in bursts, timestamps are not consumption time; kept as a fallback
because everyone knows it), RetailRocket (small, and its licence terms would need checking).

## Files and columns

Layout on the Hub: `flat/50m/` with `listens.parquet`, `likes.parquet`,
`dislikes.parquet`, `unlikes.parquet`, `undislikes.parquet`, `multi_event.parquet`
and `embeddings.parquet`.

| Column | Type | Notes |
|---|---|---|
| `uid` | uint32 | User |
| `item_id` | uint32 | Track |
| `timestamp` | uint32 | Time since dataset start in **5-second units** |
| `is_organic` | uint8 | See open questions |
| `played_ratio_pct` | uint16 | Listens only |
| `track_length_seconds` | uint32 | Listens only |

## Mapping to the canonical schema

Implemented in `src/driftloop/data/yambda.py`.

- `user_id` = `uid`, `item_id` = `item_id`.
- `ts` = `timestamp * 5` (seconds since dataset start).
- A listen counts as a positive interaction when `played_ratio_pct >= 50`.
  `weight` = 1.0.
- Likes, dislikes and their reversals are not used in M0. Whether likes become a
  second positive signal is an M1 decision.

## Getting the data

```bash
uv sync --extra data
uv run driftloop yambda-download --size 50m --out data/yambda
```

Data lives under `data/`, which is git-ignored. Nothing from the dataset is ever
committed, including samples.

## Open questions for M1

1. **Meaning of `is_organic`.** The card's field table says 0 = recommender-driven
   and 1 = organic, while its FAQ words it the other way round. Settle it by
   comparing the organic share in the data with the figure reported in the Yambda
   paper, and record the answer here.
2. **Licence wording.** The card is tagged Apache 2.0 and also says the dataset is
   published for scientific and research purposes. This project is non-commercial
   research and does not redistribute data, which satisfies both readings; re-read
   the card before publishing results.
3. **Real time span and events per day**, to choose warm-up and window length.
