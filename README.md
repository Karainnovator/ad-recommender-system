# Evaluating Personalized Advertising Recommender Systems

A technical and ethical analysis of popularity-based vs. personalized ad recommendation.

**Authors:** Manno Elwasty, Kadir Kara, Haseeb Shamsul, Wesley van der Engh
**Course:** Recommender Systems, University of Amsterdam

## Project Structure

```
ad-recommender-system/
├── data/
│   ├── generate_data.py                   # Synthetic dataset generator
│   └── social_media_ad_engagement.csv     # Real Kaggle dataset (place here)
├── src/
│   ├── popularity_recommender.py          # System 1: popularity-based
│   ├── personalized_recommender.py        # System 2: LightGBM personalized
│   ├── evaluation.py                      # Precision@K, Recall@K, AUC-ROC
│   └── fairness.py                        # Gini, demographic parity, exposure
├── results/                               # Generated CSV results
├── main.py                                # End-to-end pipeline
└── requirements.txt
```

## Dataset

This project uses the [Social Media Ad Engagement Dataset](https://www.kaggle.com/datasets/ziya07/social-media-ad-engagement-dataset) from Kaggle.

Download the CSV and place it at `data/social_media_ad_engagement.csv`. If the file is not present, the pipeline automatically generates a synthetic dataset with the same schema.

**Features used for prediction:**
| Feature | Type | Description |
|---------|------|-------------|
| `age` | User | Age of the user |
| `gender` | User | Male, Female, Other |
| `location` | User | Geographic region |
| `device_type` | User | Mobile, Desktop, Tablet |
| `ad_type` | Ad | Image, Video, Text, Carousel |
| `ad_category` | Ad | Technology, Fashion, Food, etc. |
| `ad_duration` | Ad | Length in seconds |
| `engaged` | Target | Binary: 1 = engaged, 0 = not |

> Note: Columns like `clicks`, `likes`, `shares`, `view_time` are **not** used as input features because they represent engagement outcomes — using them would cause data leakage.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# With real Kaggle data (recommended)
python main.py data/social_media_ad_engagement.csv

# With synthetic data (if no CSV available)
python main.py
```

## Two Recommender Systems

### 1. Popularity-Based (Baseline)
Recommends the same top-K highest-engagement ads to **all** users. Simple, transparent, but concentrates exposure on already popular content (Matthew effect).

### 2. Personalized (LightGBM)
Predicts `P(engagement | user, ad)` using user and ad features. Recommends per-user top-K by predicted engagement probability. More accurate but raises bias and fairness concerns.

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| Precision@K | Fraction of top-K recommendations that are relevant |
| Recall@K | Fraction of relevant items found in top-K |
| AUC-ROC | Model's ability to rank engaged above non-engaged |
| Gini coefficient | Concentration of recommendations across ad categories |
| Exposure distribution | What ad categories each demographic group sees |

## Key Research Questions

1. How do popularity-based and personalized recommenders compare on accuracy?
2. Do these systems produce differential exposure across demographic groups?
3. What are the trade-offs between engagement optimization and fair information distribution?
