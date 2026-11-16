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
- **Long enough to drift.** The Yambda paper describes an observation period of
  about eleven months. The dataset card does not state the span; M1 measures it.
- **Catalogue churn.** Over 900 thousand tracks for 10 thousand users means new
  items keep arriving, which is the kind of drift a recommender really faces.
- **Audio embeddings** give the M2 item tower something to use for tracks with no
  interactions. They cover most tracks in the full dataset; coverage of the 50M
  subset is not stated and is measured in M1.
- **Organic flag.** Each event says whether it came from the platform's recommender
  or from the user's own navigation.
- **Fits on a laptop.**

Considered and rejected: OTTO (about five weeks, too short to see drift), MovieLens
(ratings entered in bursts, timestamps are not consumption time; kept as a fallback
because everyone knows it), RetailRocket (small, and its licence terms would need checking).

## Files and columns

Layout on the Hub: `flat/50m/` with `listens.parquet`, `likes.parquet`,
`dislikes.parquet`, `unlikes.parquet`, `undislikes.parquet` and
`multi_event.parquet`. `embeddings.parquet` and the album and artist mapping files
sit at the repository root, shared by all sizes.

| Column | Type | Notes |
|---|---|---|
| `uid` | uint32 | User |
| `item_id` | uint32 | Track |
| `timestamp` | uint32 | Time since dataset start; unit unconfirmed, see open question 1 |
| `is_organic` | uint8 | See open questions |
| `played_ratio_pct` | uint16 | Listens only |
| `track_length_seconds` | uint32 | Listens only |

## Mapping to the canonical schema

Implemented in `src/driftloop/data/yambda.py`.

- `user_id` = `uid`, `item_id` = `item_id`.
- `ts` = `timestamp * timestamp_unit_seconds`. The default unit is 1 (the raw value
  is already seconds); see open question 1.
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

1. **Timestamp unit.** The card says "delta times, binned into 5s units"; the
   paper's formula floors the elapsed seconds to a multiple of 5, which means the
   stored value is already in seconds. Getting this wrong silently makes every
   window five times too short or too long. Settle it first: for about eleven
   months the largest raw timestamp is near 2.9e7 if the unit is seconds and near
   5.7e6 if it is five seconds. Then set `timestamp_unit_seconds` and record it here.
2. **Meaning of `is_organic`.** The card's field table says 0 = recommender-driven
   and 1 = organic, while its FAQ words it the other way round. Settle it by
   comparing the organic share in the data with the paper, which reports a little
   under half of listens as recommendation-driven, and record the answer here.
3. **Licence wording.** The card is tagged Apache 2.0 and also says the dataset is
   published for scientific and research purposes. This project is non-commercial
   research and does not redistribute data, which satisfies both readings; re-read
   the card before publishing results.
4. **Real time span and events per day**, to choose warm-up and window length.
