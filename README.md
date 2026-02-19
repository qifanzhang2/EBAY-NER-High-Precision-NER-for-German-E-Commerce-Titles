# EBAY-NER — High-Precision NER for German E-Commerce Titles

Competition-grade named-entity recognition (NER) for German listing titles: extract **brands**, **product types**, and **attribute values**. This solution is tuned for **F0.2** (**precision ≫ recall**) where false positives are far more expensive than misses.

> Design principle: **selective extraction**. If the model isn’t confident, it stays quiet.

---

## Results (from the challenge run)

- **4th / ~100 teams** — eBay University Machine Learning Challenge
- **F0.2 ≈ 0.93** (baseline ≈ 0.88)
- **Precision ≈ 0.94** with minimal recall loss via a second-stage verifier

---

## What’s in this repo

```text
.
├── Ebay_save.py                 # end-to-end pipeline (train → calibrate → verify → predict)
├── Gaz_save.py                  # build gazetteers from the labeled train TSV
├── Tagged_Titles_Train.tsv       # token-level labeled training data
├── README.md
└── Annexure_updated.pdf          # write-up / appendix
```

> Note: `Listing_Titles.tsv` (the unlabeled titles for final prediction) is expected by default,
> but is not committed here. Place it next to the scripts or point to it via `TITLES_TSV`.

---

## Method overview (faithful to the code)

### Stage 1 — Sequence tagger (token-level BIO tagging)

- Encoder: **`xlm-roberta-large`**
- Head: **linear classifier + CRF** (Conditional Random Field) decoding
- **Category-aware label masking**: only aspects valid for a category are allowed
- Optional feature fusion:
  - **CharCNN** token features (helps with part numbers / weird formats)
  - **Per-aspect gazetteer logit bias** (pushes known values toward the right labels)

### Ensemble + strict aggregation

- **5-fold CV ensemble**
- Decode each fold → extract spans → aggregate across folds
- A span survives only if it passes **vote** + **confidence** thresholds (category + aspect aware)
- Hard caps per title per aspect prevent “entity spam”

### Calibration (contest-metric targeted)

- Thresholds are tuned on OOF (out-of-fold) predictions to maximize the **challenge metric**
- Includes safeguards so tuning doesn’t “win by deleting everything”

### Stage 2 — Precision gate (verifier)

A **`HistGradientBoostingClassifier`** runs *after* strict aggregation to filter borderline spans using
entity-level features (span length/shape, confidence stats, gazetteer signals, token patterns, etc.).
It is intentionally conservative: the point is to **delete suspicious predictions**.

The verifier is also **safety-checked**: if it doesn’t improve OOF score enough, the pipeline can fall back to stage-1 only.

---

## Quickstart

### 1) Install dependencies

Python 3.9+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

pip install -U pip
pip install torch transformers torchcrf scikit-learn pandas numpy
```

> GPU strongly recommended. CPU will work but will be slow.

---

## Data formats

### `Tagged_Titles_Train.tsv` (token-level supervision)

This is read as a TSV grouped by **Record Number**. The script expects:
- column 0: record id
- column 1: category id (1 or 2)
- column 3: token
- column 4: aspect label (or `O` / continuation marker depending on the provided format)

(Exactly matching the challenge’s provided training dump.)

### `Listing_Titles.tsv` (unlabeled)

Expected columns:
- record id
- category id
- title text

If your file uses `Category Id`, the code renames it automatically.

---

## Run end-to-end (train → calibrate → verify → predict)

Default behavior:
- trains (or loads) 5 fold checkpoints into `checkpoints_proplus/`
- calibrates thresholds on OOF
- trains the verifier on OOF (if enabled)
- runs prediction on records `5001..30000`
- writes `predictions.tsv`

```bash
python Ebay_save.py
```

Outputs:
- `predictions.tsv` (submission format)
- `checkpoints_proplus/fold*_best.pt` (cached fold checkpoints)

---

## Build gazetteers (recommended)

The tagger can consume gazetteer files for extra precision.
`Gaz_save.py` mines labeled spans from the training data and builds:
- `gazetteer_hard.json`
- `gazetteer_soft.json`
- `gazetteer_coverage.json`

```bash
python Gaz_save.py --train Tagged_Titles_Train.tsv
```

Then run the main pipeline (it will pick them up by default).

---

## Repro / Debug knobs (environment variables)

These are read directly in `Ebay_save.py`:

- **Paths**
  - `TRAIN_TSV` (default: `Tagged_Titles_Train.tsv`)
  - `TITLES_TSV` (default: `Listing_Titles.tsv`)
  - `GAZ_FILE` / `GAZ_SOFT_FILE`
  - `SUBMIT_OUT` (default: `predictions.tsv`)
- **Speed**
  - `BATCH_SIZE` (default: 8)
  - `TORCH_COMPILE=1` (use `torch.compile` when available)
- **Precision package**
  - `HIGH_PRECISION=1` (default on)
  - `USE_VERIFIER=1` (default on)
  - `VER_PREC_FLOOR` (default tuned for high precision mode)
  - `USE_CHARCNN=1`, `CHAR_MAXLEN`
  - `USE_ASP_GAZ=1`
- **Validation**
  - `STRICT_CV=1` to run strict CV reporting

Example (fast-ish dev run, verifier off):

```bash
STRICT_CV=1 USE_VERIFIER=0 BATCH_SIZE=16 python Ebay_save.py
```

---

## Why this works (in one paragraph)

German e-commerce titles are noisy, abbreviatory, and packed with alphanumerics. This solution treats extraction as a **risk-controlled decision**: a strong tagger proposes candidates, an ensemble+calibrator makes “only-if-confident” decisions, and a verifier deletes anything that smells like a false positive. That’s how you optimize F0.2 without playing whack-a-mole with rules.

---

## Notes / caveats

- This repo prioritizes **reproducibility of the challenge pipeline**, not library-style cleanliness.
  `Ebay_save.py` is intentionally all-in-one so it can be run in a single command.
- If you want to productionize it, the natural refactor is: `data/`, `model/`, `calib/`, `verifier/`, `predict/`.

---

## License

For academic / portfolio use. If you reuse parts, please attribute.
