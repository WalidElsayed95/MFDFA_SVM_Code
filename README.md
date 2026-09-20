# MFDFA–SVM earthquake/icequake reproducibility code

This archive contains the code and derived data used to estimate multifractal
features, generate the study figures, and evaluate nine independent
station-specific support-vector classifiers. It accompanies the manuscript
*Multifractal signatures reveal source-dependent differences between tectonic
earthquakes and icequakes*.

## Contents

- `mfdfa.py`: MFDFA spectrum estimation and scalar feature definitions.
- `extract_features.py`: raw-waveform feature-extraction driver.
- `nested_svm.py`: station-specific nested stratified 10 x 10 SVM procedure.
- `evaluate_models.py`: metrics, confusion matrices, validation, and provenance.
- `verify_release.py`: read-only integrity and numerical checks.
- `figure_*.py`: code used to generate the corresponding manuscript figures.
- `data/mfdfa_features.csv`: archived 28,022-row derived feature table.
- `data/selected_records.csv`: selected-record manifest.
- `results/`: complete out-of-fold predictions and saved evaluation outputs.
- `supplementary/`: supplementary feature distributions and statistics.

## Analysis design

The 28,022 station–waveform records comprise 13,749 earthquake and 14,273
icequake records from nine stations. A separate SVM is fitted for each station.
The outer ten folds estimate held-out performance; a ten-fold stratified grid
search is performed inside each outer-training partition. `StandardScaler` and
`SVC(class_weight="balanced")` are contained in one pipeline. Stations are not
pooled during training or testing.

The archived outputs give a station-macro accuracy of 94.28%, station-macro
balanced accuracy of 94.28%, macro precision of 93.56%, and macro F1 of 93.87%.

## Installation

Python 3.11 was used for the archived run. From the release directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The Figure 2 map has separate system-level GMT requirements; see
`environment_map.yml`.

## Verify the archived release

```bash
python verify_release.py
```

This checks the source syntax, SHA-256 of the derived table, correspondence
with the selection manifest, 28,022 unique out-of-fold predictions, nine
stations, 90 outer folds, saved macro metrics, and a numerical MFDFA smoke test.

## Reproduce the classifier outputs

The supplied derived feature table is sufficient to reproduce the model
evaluation:

```bash
python evaluate_models.py
```

Existing outputs are reused when their input hash and validation structure
match. To force retraining, run:

```bash
python nested_svm.py --no-reuse
python evaluate_models.py
```

## Regenerate features from raw arrays

The original waveform arrays are not redistributed in this software archive.
To regenerate `data/mfdfa_features.csv`, place these source files in
`data/raw/`:

- `all_waveforms.npy`: shape `(43363, 3001)` in the supplied source format;
  columns 0–2999 contain waveform samples and the last column contains the
  station identifier.
- `all_labels.npy`: one binary label per source row (`0` earthquake,
  `1` icequake).

`serial_no` in `data/selected_records.csv` indexes the source arrays. Preserve
the archived feature table before regeneration, then run:

```bash
python extract_features.py
```

The source NPY array contains 3,000 samples per row and does not itself encode
the sampling frequency. The script does not add filtering, resampling,
imputation, or zero filling; it processes the supplied samples selected by the
manifest.

## Generate figures

```bash
python figure_01_singularity_spectrum.py
python figure_03_receiver_counts.py
python figure_05_workflow.py
python figure_07_feature_distributions.py
```

Figure 2 additionally requires the source earthquake and icequake catalog CSVs,
PyGMT/GMT, and access to the GMT remote relief grid:

```bash
python figure_02_event_station_map.py \
  --earthquakes /path/to/earthquakes_catalog.csv \
  --icequakes /path/to/icequakes_catalog.csv
```

## Reproducibility notes

- `serial_no` is a unique feature-row identifier, not a verified physical-event
  identifier across stations.
- The nine station models are independent; no waveform from another station is
  included in a given station model.
- The summed confusion matrix is descriptive only and is not the output of a
  pooled global classifier.
- Package file hashes are listed in `SHA256SUMS.txt`.

## Citation and license

The original code, derived feature data, evaluation results, and generated
figures in this repository are released under the MIT License; see `LICENSE`
(Copyright (c) 2026 Walid E. AboElnasr). This license does not alter the terms
of third-party source material or software dependencies. The original waveform
arrays are not redistributed. Citation metadata are provided in `CITATION.cff`.
