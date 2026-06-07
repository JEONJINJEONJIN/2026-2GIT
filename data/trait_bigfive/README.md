# TRAIT Big Five Data Notes

This directory is reserved for the normalized TRAIT Big Five dataset used by
the v4 speech-action consistency experiments.

Expected files after conversion:

- `scenarios.jsonl`: normalized scenario/action rows.
- `splits.json`: audit index of scenario ids by split and trait.
- `contrastive_pairs/*.jsonl`: high-vs-low pair files for trait vector
  extraction.

Raw TRAIT exports are not committed here. Convert an approved local export with:

```powershell
.venv\Scripts\python.exe scripts\11_convert_trait_bigfive.py --input path\to\trait_export.jsonl
```

Then build vector-extraction pairs with:

```powershell
.venv\Scripts\python.exe scripts\09_build_trait_contrastive_pairs.py
```

The converter expects the TRAIT dataset-card fields:

- `personality`
- `question`
- `response_high1`
- `response_high2`
- `response_low1`
- `response_low2`
