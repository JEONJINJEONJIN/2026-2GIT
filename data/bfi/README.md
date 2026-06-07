# BFI-44 Data Notes

`bfi44_scoring.csv` stores BFI-44 scoring metadata for v4 Big Five
speech-measurement experiments.

Columns:

- `item_id`: stable local item id.
- `trait`: one of `agreeableness`, `conscientiousness`, `neuroticism`,
  `openness`, or `extraversion`.
- `reverse_scored`: whether the 1-5 Likert answer should be reversed before
  normalization.
- `text`: item wording. This is intentionally empty until redistribution rights
  for the exact BFI-44 wording are confirmed.

For local experiments, copy `bfi44_item_text.local.example.csv` to
`bfi44_item_text.local.csv` and fill the `text` column from an approved source.
The `.local.csv` file is git-ignored because item wording may not be safe to
redistribute.

The v4 experiment configs read:

```text
data/bfi/bfi44_scoring.csv
data/bfi/bfi44_item_text.local.csv
```

Do not run `scripts/12_run_bigfive_pilot.py` until all local item text is
populated. The readiness checker and runner fail fast if any item text is empty.
