# Email Campaign Recommender System

A two-tower retrieval model that predicts which subscribers are most likely to open a given email campaign.

## Overview

The system combines collaborative filtering (ALS) with content-based signals (sentence transformer embeddings) to rank subscribers for each campaign. It is designed to handle cold-start campaigns — new campaigns the model has never seen during training.

## Architecture

**Baseline — ALS (Alternating Least Squares)**
Factorises a sparse subscriber × campaign interaction matrix into low-rank embeddings. These learned factors also serve as warm-start weights for the two-tower model.

**Two-Tower Model**
- **User tower:** ALS user embedding + normalised subscriber engagement stats (open rate, click rate, etc.) → fused 50-dim L2-normalised vector
- **Item tower:** ALS item embedding + sentence-transformer text embedding (projected from 384 → 50 dim) → fused 50-dim L2-normalised vector
- **Loss:** In-batch negatives retrieval loss (temperature = 0.1)

**Two-phase training:**
1. Phase 1 (8 epochs) — warms up the text projection and fusion layers while ALS embeddings are frozen
2. Phase 2 (6 epochs) — fine-tunes all variables jointly; ALS embeddings updated at a lower learning rate (0.003) to preserve their collaborative signal

## Data

Three input sources (loaded from Google Drive):
- `amplify_email_sends.csv` — individual send records (opened, clicked, converted, bounced flags)
- `amplify_email_subscribers.csv` — subscriber-level static features
- `amplify_email_campaigns.json` — campaign metadata including subject and HTML body

Interaction weights: delivered-not-opened = 0, opened = 3, clicked = 8, converted = 18.

## Pipeline

| Step | Description |
|------|-------------|
| 0 | Install & import dependencies |
| 1 | Load data from Google Drive |
| 2 | Clean sends, strip HTML from campaigns, build numeric subscriber features |
| 3 | Temporal train/test split (last 20% of sends by time → test) |
| 4 | Build weighted interaction pairs (positives + explicit negatives) |
| 5 | Train ALS baseline |
| 6 | Encode campaign text with `BAAI/bge-small-en-v1.5` sentence transformer |
| 7 | Define user and item towers |
| 8 | Build `tf.data` pipelines |
| 9 | Two-phase training of the two-tower model |
| 10 | Evaluate Recall@K and NDCG@K (cold-start + popularity baseline) |
| 11 | Multi-seed evaluation (5 seeds) to measure result variance |

## Evaluation

Models are compared against a **popularity baseline** (globally most-active subscribers) using **Recall@K** and **NDCG@K** at K ∈ {10, 25, 50, 100, 200}.

The primary evaluation scenario is **cold-start** — campaigns that appeared only in the test set (~99% of test campaigns). The two-tower model handles cold-start via text embeddings; ALS falls back to the mean item factor.

## Dependencies

```
tensorflow / tf-keras
tensorflow-recommenders
implicit
sentence-transformers
pandas, numpy, scipy, scikit-learn
matplotlib
```

## Notes

- Runs on Google Colab with Google Drive mounted at `/content/drive`
- Text embeddings cached to `/content/drive/MyDrive/models`
- Output plots saved to `/content/drive/MyDrive/models/k_sensitivity_plots.png`
