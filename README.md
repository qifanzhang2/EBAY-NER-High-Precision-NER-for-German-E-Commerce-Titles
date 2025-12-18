# EBAY-NER — High-Precision NER for German E-Commerce Titles

Competition-grade named-entity recognition (NER) for German listing titles: extract **brands**, **product types**, and **attributes**. Built for **F0.2** (precision >> recall), where false positives are far more costly than misses.

## Results
- **4th / ~100 teams** — eBay University Machine Learning Challenge  
- **F0.2 ≈ 0.93** (baseline ≈ 0.88)  
- **Precision ≈ 0.94** with minimal recall loss via a second-stage verifier  

## Approach (high level)
1. **Sequence tagger:** `xlm-roberta-large` + **CRF** for BIO tagging  
2. **Ensembling + calibration:** 5-fold CV ensemble, category-aware calibration  
3. **Verifier (“precision gate”):** `HistGradientBoostingClassifier` filters borderline spans using entity-level features (confidence stats, span length, gazetteer hits)

## Repo layout
```text
ebay-ner/
├── Ebay_save.py              # end-to-end pipeline: tagger → calibration → verifier
├── Tagged_Titles_Train.tsv   # labeled German titles (train/val)
├── Listing_Titles.tsv        # unlabeled titles (final predictions)
├── gazetteer_hard.json       # curated brand & product lexicons
└── README.md
```

## Tech stack
Python • PyTorch • Hugging Face Transformers • scikit-learn • pandas/NumPy • W&B • Git

## Evaluation
- Metric: **F0.2** (heavily penalizes false positives)  
- Validation: held-out splits by product category + error analysis by entity/category  

## Future work
Domain adaptation (other languages), active learning for long-tail entities, confidence-aware abstention for safer deployment.
