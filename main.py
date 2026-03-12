"""
Main pipeline: generate data, train both recommenders, evaluate, and analyze fairness.
Run: python main.py
"""
import sys
import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from data.generate_data import generate_dataset
from src.popularity_recommender import PopularityRecommender
from src.personalized_recommender import PersonalizedRecommender
from src.evaluation import (
    get_relevant_items, evaluate_recommendations, compute_summary, compute_prediction_metrics
)
from src.fairness import fairness_report, category_concentration, exposure_by_group


def main():
    K_VALUES = [5, 10, 20]
    K_MAIN = 10  # primary K for fairness analysis

    # ================================================================
    # 1. GENERATE / LOAD DATA
    # ================================================================
    print("=" * 60)
    print("STEP 1: Generating dataset")
    print("=" * 60)
    users, ads, interactions = generate_dataset(
        n_users=5000, n_ads=200, n_interactions=100_000, seed=42
    )
    print(f"  Users:        {len(users):,}")
    print(f"  Ads:          {len(ads):,}")
    print(f"  Interactions: {len(interactions):,}")
    print(f"  Engagement rate: {interactions['engaged'].mean():.1%}")

    # Save data
    users.to_csv("data/users.csv", index=False)
    ads.to_csv("data/ads.csv", index=False)
    interactions.to_csv("data/interactions.csv", index=False)

    # ================================================================
    # 2. TRAIN/TEST SPLIT
    # ================================================================
    print(f"\nSTEP 2: Train/test split (80/20, stratified)")
    train, test = train_test_split(
        interactions, test_size=0.2, random_state=42, stratify=interactions["engaged"]
    )
    print(f"  Train: {len(train):,}  |  Test: {len(test):,}")
    print(f"  Train engagement: {train['engaged'].mean():.1%}  |  Test: {test['engaged'].mean():.1%}")

    relevant = get_relevant_items(test)
    test_user_ids = test["user_id"].unique()
    print(f"  Test users: {len(test_user_ids):,}")
    print(f"  Users with engagements: {len(relevant):,}")

    # ================================================================
    # 3. POPULARITY-BASED RECOMMENDER
    # ================================================================
    print(f"\n{'=' * 60}")
    print("STEP 3: Popularity-Based Recommender")
    print("=" * 60)
    pop_rec = PopularityRecommender(min_impressions=5)
    pop_rec.fit(train)

    pop_recs = pop_rec.recommend_all(test_user_ids, k=max(K_VALUES))
    pop_eval = evaluate_recommendations(pop_recs, relevant, K_VALUES)
    pop_summary = compute_summary(pop_eval, K_VALUES)

    print("\n  Performance:")
    for metric, value in pop_summary.items():
        print(f"    {metric}: {value:.4f}")

    # ================================================================
    # 4. PERSONALIZED RECOMMENDER
    # ================================================================
    print(f"\n{'=' * 60}")
    print("STEP 4: Personalized Recommender (LightGBM)")
    print("=" * 60)
    pers_rec = PersonalizedRecommender()

    t0 = time.time()
    pers_rec.fit(train)
    print(f"  Training time: {time.time() - t0:.1f}s")

    # Predict on test set for AUC-ROC
    test_preds = pers_rec.predict(test)
    pred_metrics = compute_prediction_metrics(test["engaged"].values, test_preds)
    print(f"\n  Prediction metrics:")
    for metric, value in pred_metrics.items():
        print(f"    {metric}: {value:.4f}")

    # Generate recommendations for test users (sample for speed)
    sample_size = min(500, len(test_user_ids))
    sample_users = np.random.default_rng(42).choice(test_user_ids, sample_size, replace=False)
    sample_user_df = users[users["user_id"].isin(sample_users)]

    print(f"\n  Generating recommendations for {sample_size} users...")
    t0 = time.time()
    pers_recs, pers_scores = pers_rec.recommend_all(sample_user_df, ads, k=max(K_VALUES))
    print(f"  Recommendation time: {time.time() - t0:.1f}s")

    pers_eval = evaluate_recommendations(pers_recs, relevant, K_VALUES)
    pers_summary = compute_summary(pers_eval, K_VALUES)

    print("\n  Performance:")
    for metric, value in pers_summary.items():
        print(f"    {metric}: {value:.4f}")

    # ================================================================
    # 5. COMPARISON
    # ================================================================
    print(f"\n{'=' * 60}")
    print("STEP 5: Comparison")
    print("=" * 60)

    # Filter pop_eval to same users for fair comparison
    pop_eval_sample = pop_eval[pop_eval["user_id"].isin(sample_users)]
    pop_summary_sample = compute_summary(pop_eval_sample, K_VALUES)

    print(f"\n  {'Metric':<15} {'Popularity':>12} {'Personalized':>14} {'Delta':>10}")
    print(f"  {'-'*55}")
    for k in K_VALUES:
        for metric_type in ["Precision", "Recall"]:
            key = f"{metric_type}@{k}"
            pop_val = pop_summary_sample[key]
            pers_val = pers_summary[key]
            delta = ((pers_val - pop_val) / max(pop_val, 1e-8)) * 100
            print(f"  {key:<15} {pop_val:>12.4f} {pers_val:>14.4f} {delta:>+9.1f}%")

    print(f"\n  AUC-ROC (personalized): {pred_metrics['AUC-ROC']:.4f}")
    print(f"  Log-loss (personalized): {pred_metrics['Log-loss']:.4f}")

    # ================================================================
    # 6. FAIRNESS ANALYSIS
    # ================================================================
    print(f"\n{'=' * 60}")
    print("STEP 6: Fairness Analysis")
    print("=" * 60)

    for system_name, recs, eval_df in [
        ("Popularity", {uid: pop_recs[uid] for uid in sample_users}, pop_eval_sample),
        ("Personalized", pers_recs, pers_eval),
    ]:
        print(f"\n  --- {system_name} ---")

        # Category concentration
        conc = category_concentration(recs, ads, K_MAIN)
        print(f"  Gini coefficient: {conc['gini']:.3f}")
        print(f"  Categories represented: {conc['n_categories_represented']}")
        print(f"  Top category share: {conc['top_category_share']:.1%}")

        # Accuracy by gender
        eval_merged = eval_df.merge(users[["user_id", "gender", "age"]], on="user_id")
        gender_acc = eval_merged.groupby("gender")[f"precision@{K_MAIN}"].mean()
        print(f"\n  Precision@{K_MAIN} by gender:")
        for g, v in gender_acc.items():
            print(f"    {g}: {v:.4f}")

        # Accuracy by age group
        eval_merged["age_group"] = pd.cut(eval_merged["age"], bins=[17, 25, 35, 45, 65],
                                           labels=["18-25", "26-35", "36-45", "46-65"])
        age_acc = eval_merged.groupby("age_group", observed=True)[f"precision@{K_MAIN}"].mean()
        print(f"\n  Precision@{K_MAIN} by age group:")
        for g, v in age_acc.items():
            print(f"    {g}: {v:.4f}")

        # Exposure by gender
        users_sample = users[users["user_id"].isin(sample_users)]
        exp = exposure_by_group(recs, users_sample, ads, "gender", K_MAIN)
        if not exp.empty:
            print(f"\n  Exposure distribution by gender (top categories):")
            for gender in exp.index:
                top3 = exp.loc[gender].nlargest(3)
                cats = ", ".join([f"{c}: {v:.1%}" for c, v in top3.items()])
                print(f"    {gender}: {cats}")

    # ================================================================
    # 7. SAVE RESULTS
    # ================================================================
    print(f"\n{'=' * 60}")
    print("STEP 7: Saving results")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)
    pop_eval.to_csv("results/popularity_evaluation.csv", index=False)
    pers_eval.to_csv("results/personalized_evaluation.csv", index=False)

    # Summary comparison
    comparison = []
    for k in K_VALUES:
        for mt in ["Precision", "Recall"]:
            key = f"{mt}@{k}"
            comparison.append({
                "Metric": key,
                "Popularity": pop_summary_sample[key],
                "Personalized": pers_summary[key],
            })
    pd.DataFrame(comparison).to_csv("results/comparison.csv", index=False)

    print("  Saved to results/")
    print("\nDone!")


if __name__ == "__main__":
    main()
