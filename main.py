"""
Main pipeline: Evaluating Personalized Advertising Recommender Systems.

Runs the full experiment end-to-end:
  1. Load data (real Kaggle CSV or generate synthetic)
  2. Train/test split (80/20, stratified)
  3. Train & evaluate popularity-based recommender
  4. Train & evaluate personalized recommender (LightGBM)
  5. Compare accuracy (Precision@K, Recall@K, AUC-ROC)
  6. Analyze fairness (exposure distribution, Gini, demographic parity)
  7. Save results to CSV

Usage:
  python main.py                              # uses synthetic data
  python main.py data/social_media_ad_engagement.csv  # uses real Kaggle data
"""
import sys
import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))

from data.generate_data import generate_dataset
from src.popularity_recommender import PopularityRecommender
from src.personalized_recommender import PersonalizedRecommender
from src.evaluation import (
    get_relevant_items, evaluate_recommendations,
    compute_summary, compute_prediction_metrics,
)
from src.fairness import (
    exposure_by_group, category_concentration, accuracy_by_group,
)


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────

def load_kaggle_data(csv_path: str):
    """
    Load the real Kaggle dataset and extract users / ads / interactions.

    The Kaggle CSV has one row per interaction with all columns merged.
    We split it into the three logical tables our pipeline expects.
    """
    df = pd.read_csv(csv_path)

    # Standardise column names (handle minor variations)
    col_map = {
        "device": "device_type",
        "category": "ad_category",
        "duration": "ad_duration",
        "type": "ad_type",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    user_cols = ["user_id", "age", "gender", "location", "device_type"]
    ad_cols = ["ad_id", "ad_type", "ad_category", "ad_duration"]

    users = df[user_cols].drop_duplicates("user_id").reset_index(drop=True)
    ads = df[ad_cols].drop_duplicates("ad_id").reset_index(drop=True)

    keep = user_cols + ad_cols + ["engaged"]
    keep = [c for c in keep if c in df.columns]
    interactions = df[keep].copy()

    return users, ads, interactions


def load_data(csv_path=None):
    """Load real data if a path is given, otherwise generate synthetic data."""
    if csv_path and os.path.exists(csv_path):
        print(f"  Loading real dataset: {csv_path}")
        return load_kaggle_data(csv_path)
    else:
        print("  No real dataset found — generating synthetic data")
        return generate_dataset()


# ──────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────

def main():
    K_VALUES = [5, 10, 20]
    K = 10  # primary K for fairness analysis

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/social_media_ad_engagement.csv"

    # ── 1. DATA ──────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    users, ads, interactions = load_data(csv_path)
    print(f"  Users:           {len(users):,}")
    print(f"  Ads:             {len(ads):,}")
    print(f"  Interactions:    {len(interactions):,}")
    print(f"  Engagement rate: {interactions['engaged'].mean():.1%}")

    # ── 2. TRAIN/TEST SPLIT ─────────────────────────────────
    print(f"\nSTEP 2: Train/test split (80/20, stratified)")
    train, test = train_test_split(
        interactions, test_size=0.2, random_state=42,
        stratify=interactions["engaged"],
    )
    print(f"  Train: {len(train):,}  |  Test: {len(test):,}")

    relevant = get_relevant_items(test)
    test_uids = test["user_id"].unique()
    print(f"  Test users: {len(test_uids):,}  |  With engagements: {len(relevant):,}")

    # ── 3. POPULARITY RECOMMENDER ────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 3: Popularity-Based Recommender")
    print("=" * 60)
    pop = PopularityRecommender(min_impressions=5)
    pop.fit(train)
    pop_recs = pop.recommend_all(test_uids, k=max(K_VALUES))
    pop_eval = evaluate_recommendations(pop_recs, relevant, K_VALUES)
    pop_summary = compute_summary(pop_eval, K_VALUES)

    print("\n  Accuracy:")
    for m, v in pop_summary.items():
        print(f"    {m}: {v:.4f}")

    # ── 4. PERSONALIZED RECOMMENDER ──────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 4: Personalized Recommender (LightGBM)")
    print("=" * 60)
    pers = PersonalizedRecommender()

    t0 = time.time()
    pers.fit(train)
    print(f"  Training time: {time.time() - t0:.1f}s")

    # Raw prediction quality on test set
    test_preds = pers.predict(test)
    pred_metrics = compute_prediction_metrics(test["engaged"].values, test_preds)
    print(f"  AUC-ROC:  {pred_metrics['AUC-ROC']:.4f}")
    print(f"  Log-loss: {pred_metrics['Log-loss']:.4f}")

    # Generate per-user recommendations (sample for speed)
    n_sample = min(500, len(test_uids))
    sample_uids = np.random.default_rng(42).choice(test_uids, n_sample, replace=False)
    sample_users = users[users["user_id"].isin(sample_uids)]

    print(f"\n  Generating top-{max(K_VALUES)} for {n_sample} users...")
    t0 = time.time()
    pers_recs, _ = pers.recommend_all(sample_users, ads, k=max(K_VALUES))
    print(f"  Done in {time.time() - t0:.1f}s")

    pers_eval = evaluate_recommendations(pers_recs, relevant, K_VALUES)
    pers_summary = compute_summary(pers_eval, K_VALUES)

    print("\n  Accuracy:")
    for m, v in pers_summary.items():
        print(f"    {m}: {v:.4f}")

    # ── 5. COMPARISON ────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 5: Accuracy Comparison")
    print("=" * 60)

    pop_eval_s = pop_eval[pop_eval["user_id"].isin(sample_uids)]
    pop_sum_s = compute_summary(pop_eval_s, K_VALUES)

    print(f"\n  {'Metric':<15} {'Popularity':>12} {'Personalized':>14} {'Delta':>10}")
    print(f"  {'-' * 55}")
    for k in K_VALUES:
        for mt in ["Precision", "Recall"]:
            key = f"{mt}@{k}"
            pv, rv = pop_sum_s[key], pers_summary[key]
            d = ((rv - pv) / max(pv, 1e-8)) * 100
            print(f"  {key:<15} {pv:>12.4f} {rv:>14.4f} {d:>+9.1f}%")

    print(f"\n  AUC-ROC (personalized): {pred_metrics['AUC-ROC']:.4f}")

    # ── 6. FAIRNESS ANALYSIS ─────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 6: Fairness Analysis")
    print("=" * 60)

    users_s = users[users["user_id"].isin(sample_uids)].copy()
    users_s["age_group"] = pd.cut(users_s["age"], bins=[17, 25, 35, 45, 65],
                                   labels=["18-25", "26-35", "36-45", "46-65"])

    for name, recs, ev in [
        ("Popularity", {u: pop_recs[u] for u in sample_uids}, pop_eval_s),
        ("Personalized", pers_recs, pers_eval),
    ]:
        print(f"\n  ── {name} {'─' * (40 - len(name))}")

        # Category concentration
        conc = category_concentration(recs, ads, K)
        print(f"  Gini coefficient:      {conc['gini']:.3f}")
        print(f"  Categories represented: {conc['n_categories_represented']}")
        print(f"  Top category share:    {conc['top_category_share']:.1%}")

        # Accuracy by gender
        ev_m = ev.merge(users_s[["user_id", "gender"]], on="user_id")
        g_acc = ev_m.groupby("gender")[f"precision@{K}"].mean()
        print(f"\n  Precision@{K} by gender:")
        for g, v in g_acc.items():
            print(f"    {g}: {v:.4f}")

        # Accuracy by age group
        ev_a = ev.merge(users_s[["user_id", "age_group"]], on="user_id")
        a_acc = ev_a.groupby("age_group", observed=True)[f"precision@{K}"].mean()
        print(f"\n  Precision@{K} by age group:")
        for g, v in a_acc.items():
            print(f"    {g}: {v:.4f}")

        # Exposure by gender
        exp = exposure_by_group(recs, users_s, ads, "gender", K)
        if not exp.empty:
            print(f"\n  Ad category exposure by gender:")
            for grp in exp.index:
                top3 = exp.loc[grp].nlargest(3)
                cats = ", ".join(f"{c}: {v:.1%}" for c, v in top3.items())
                print(f"    {grp}: {cats}")

    # ── 7. SAVE ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 7: Saving results")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)
    pop_eval.to_csv("results/popularity_evaluation.csv", index=False)
    pers_eval.to_csv("results/personalized_evaluation.csv", index=False)

    rows = []
    for k in K_VALUES:
        for mt in ["Precision", "Recall"]:
            key = f"{mt}@{k}"
            rows.append({"Metric": key, "Popularity": pop_sum_s[key],
                          "Personalized": pers_summary[key]})
    pd.DataFrame(rows).to_csv("results/comparison.csv", index=False)

    print("  Saved to results/")
    print("\nDone!")


if __name__ == "__main__":
    main()
