# Evaluating Personalized Advertising Recommender Systems

A technical and ethical analysis of popularity-based vs. personalized ad recommendation.

**Authors:** Manno Elwasty, Kadir Kara, Haseeb Shamsul
**Course:** Introduction to Modeling Systems Dynamics, University of Amsterdam

## Project Structure

```
ad-recommender-system/
├── data/
│   ├── generate_data.py          # Synthetic dataset generator
│   ├── users.csv                 # Generated: 5,000 users
│   ├── ads.csv                   # Generated: 200 advertisements
│   └── interactions.csv          # Generated: 100,000 interactions
├── src/
│   ├── popularity_recommender.py # System 1: popularity-based
│   ├── personalized_recommender.py # System 2: LightGBM personalized
│   ├── evaluation.py             # Precision@K, Recall@K, AUC-ROC
│   └── fairness.py               # Demographic parity, exposure analysis
├── notebooks/
│   └── analysis.ipynb            # Full analysis with visualizations
├── results/                      # Generated evaluation results
├── main.py                       # End-to-end pipeline
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
# Full pipeline: data generation → training → evaluation → fairness analysis
python main.py

# Or generate data separately
python data/generate_data.py
```

## Two Recommender Systems

### 1. Popularity-Based (Baseline)
Recommends the same top-K highest-engagement ads to **all** users. Simple, transparent, but concentrates exposure on already-popular content (Matthew effect).

### 2. Personalized (LightGBM)
Predicts `P(engagement | user, ad)` using user features (age, gender, location, device) and ad features (type, category, duration). Recommends per-user top-K by predicted engagement. More accurate but raises bias/fairness concerns.

## Evaluation

| Metric | What it measures |
|--------|-----------------|
| Precision@K | Fraction of top-K recommendations that are relevant |
| Recall@K | Fraction of relevant items found in top-K |
| AUC-ROC | Model's ability to distinguish engaged vs. not engaged |
| Gini coefficient | Concentration of recommendations across categories |
| Demographic parity | Equal exposure distribution across demographic groups |
