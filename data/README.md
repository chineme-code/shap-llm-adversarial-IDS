# Data

The datasets are not committed to this repository (they are large and are
redistributed from their original sources). Download them and place them here.

## IDS2025 (primary)
- **File expected:** `data/IDS2025.xlsx`
- **Source:** Mendeley Data, "IDS2025 (Balanced Intrusion Detection Evaluation
  Dataset)", Panigrahi & Borah (2025), DOI 10.17632/pkskt3fv3v.1
- A refined, class-rebalanced derivative of CICIDS2017. 91,830 rows, 80 columns,
  7 classes. Label column is `newLabel`.

## UNSW-NB15 (generalization test)
- **Files expected:** `data/UNSW_NB15_training-set.csv`,
  `data/UNSW_NB15_testing-set.csv`
- **Source:** UNSW Canberra, Australian Centre for Cyber Security
  (Moustafa & Slay, 2015). Also mirrored on Kaggle and Figshare.
- Use the pre-split partition files, not the raw four-part archive.
- Note: in the public release the file *contents* are swapped relative to their
  names (the larger 175,341-row partition is the training set). `preprocess.load_unsw`
  handles this automatically by assigning `train` = the larger partition.

## Generated file
- `data/X_test_sample.csv` is written by `src/run_ids2025.py` (the first 50 test
  rows) so the dashboard has data to display. It is safe to keep in the repo.
