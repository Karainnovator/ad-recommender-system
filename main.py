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
    exposure_by_group, category_concentration,
    exposure_divergence, accuracy_equity,
    provider_fairness, fairness_scorecard, compare_systems,
    print_fairness_report, print_comparison_report,
)
from src.bias import run_bias_audit, print_bias_report, calibration_summary
from src.fair_reranker import FairReranker


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
    CANDIDATE_K = 50  # larger pool for fair re-ranking

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
    pop_recs = pop.recommend_all(test_uids, k=CANDIDATE_K)
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

    print(f"\n  Generating top-{CANDIDATE_K} candidates for {n_sample} users...")
    t0 = time.time()
    pers_candidates, pers_scores = pers.recommend_all(sample_users, ads, k=CANDIDATE_K)
    print(f"  Done in {time.time() - t0:.1f}s")

    # Slice to top-K_VALUES for original evaluation (same as before)
    pers_recs = {uid: recs[:max(K_VALUES)] for uid, recs in pers_candidates.items()}

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

    scorecards = {}

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

        # Provider fairness (ad-side)
        pf = provider_fairness(recs, ads, K)
        if pf:
            print(f"\n  Provider fairness:")
            print(f"    Catalog coverage: {pf['catalog_coverage']}%")
            print(f"    HHI: {pf['hhi']:.4f} ({pf['hhi_verdict']})")
            print(f"    Ads with zero exposure: {pf['zero_exposure_count']} ({pf['zero_exposure_pct']}%)")

        # Exposure divergence between groups
        for col in ["gender", "age_group"]:
            div_df = exposure_divergence(recs, users_s, ads, col, K)
            if not div_df.empty:
                print(f"\n  Exposure divergence ({col}):")
                for _, row in div_df.iterrows():
                    print(f"    {row['group_a']} vs {row['group_b']}: "
                          f"JSD={row['jsd']:.4f}")

        # Accuracy equity (statistical + practical significance)
        for col in ["gender", "age_group"]:
            eq = accuracy_equity(ev, users_s, col, K)
            print(f"\n  Accuracy equity ({col}):")
            print(f"    Gap ratio: {eq['gap_ratio']:.3f} "
                  f"(best: {eq['best_served']}, worst: {eq['worst_served']})")
            print(f"    Effect size: ε²={eq['epsilon_squared']:.4f} "
                  f"({eq['effect_size']}), p={eq['kruskal_p_value']:.4f}")

        # Exposure by gender
        exp = exposure_by_group(recs, users_s, ads, "gender", K)
        if not exp.empty:
            print(f"\n  Ad category exposure by gender:")
            for grp in exp.index:
                top3 = exp.loc[grp].nlargest(3)
                cats = ", ".join(f"{c}: {v:.1%}" for c, v in top3.items())
                print(f"    {grp}: {cats}")

        # Fairness scorecard
        sc = fairness_scorecard(recs, ev, users_s, ads, relevant, K)
        scorecards[name] = sc
        print_fairness_report(sc, system_name=name)

    # Cross-system comparison
    if len(scorecards) == 2:
        comparison = compare_systems(scorecards)
        print_comparison_report(comparison, list(scorecards.keys()))

    # ── 7. BIAS AUDIT ─────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 7: Bias Audit")
    print("=" * 60)

    for name, recs, ev in [
        ("Popularity", {u: pop_recs[u] for u in sample_uids}, pop_eval_s),
        ("Personalized", pers_recs, pers_eval),
    ]:
        audit = run_bias_audit(
            recommendations=recs,
            relevant_items=relevant,
            interactions=train,
            users=users_s,
            ads=ads,
            eval_df=ev,
            k=K,
        )
        print_bias_report(audit, system_name=name)

    # Calibration bias (personalized model only)
    # test already has gender from data generation; merge only if missing
    if "gender" in test.columns:
        gender_col = test["gender"].values
    else:
        gender_col = test.merge(users[["user_id", "gender"]], on="user_id")["gender"].values
    cal = calibration_summary(
        y_true=test["engaged"].values,
        y_pred=test_preds,
        groups=gender_col,
    )
    if not cal.empty:
        print(f"\n  Calibration (ECE) by gender — personalized model:")
        for _, row in cal.iterrows():
            print(f"    {row['group']}: ECE={row['ECE']:.4f}")

    # ── 8. FAIR RE-RANKING ──────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 8: Fair Re-ranking (accuracy vs fairness trade-off)")
    print("=" * 60)

    # Build candidate pools for re-ranking
    pop_candidates_s = {u: pop_recs[u] for u in sample_uids}

    # Test multiple lambda values for the personalized recommender
    lambdas = [1.0, 0.7, 0.5, 0.3]
    print(f"\n  MMR diversity re-ranking (personalized model)")
    print(f"  Candidates per user: {CANDIDATE_K} → select top-{max(K_VALUES)}")
    print(f"\n  {'λ':<6} {'Prec@10':>8} {'Rec@10':>8} {'Gini':>6} "
          f"{'HHI':>6} {'Coverage':>9} {'MaxJSD(g)':>10} {'Stereo':>7}")
    print(f"  {'-' * 65}")

    fair_variants = {}
    for lam in lambdas:
        reranker = FairReranker(ads, strategy="mmr", lambda_param=lam)
        fair_recs = reranker.rerank_all(
            pers_candidates, k=max(K_VALUES), scores_dict=pers_scores
        )
        fair_eval = evaluate_recommendations(fair_recs, relevant, K_VALUES)
        fair_sum = compute_summary(fair_eval, K_VALUES)

        # Key fairness metrics
        conc = category_concentration(fair_recs, ads, K)
        pf = provider_fairness(fair_recs, ads, K)
        div_df = exposure_divergence(fair_recs, users_s, ads, "gender", K)
        max_jsd_g = div_df["jsd"].max() if not div_df.empty else 0.0

        # Stereotyping count
        from src.bias import bias_amplification
        ba = bias_amplification(fair_recs, train, users_s, ads, "gender", K)
        n_stereo = len(ba[ba["is_stereotyping"]]) if not ba.empty else 0

        label = f"{lam:.1f}"
        if lam == 1.0:
            label += " (orig)"
        print(f"  {label:<6} {fair_sum['Precision@10']:>8.4f} {fair_sum['Recall@10']:>8.4f} "
              f"{conc['gini']:>6.3f} {pf['hhi']:>6.4f} "
              f"{pf['catalog_coverage']:>8.1f}% {max_jsd_g:>10.4f} "
              f"{n_stereo:>5}/24")

        fair_variants[f"λ={lam}"] = {
            "recs": fair_recs, "eval": fair_eval, "summary": fair_sum,
        }

    # Also test category cap
    cap_reranker = FairReranker(ads, strategy="category_cap", category_cap=0.3)
    cap_recs = cap_reranker.rerank_all(
        pers_candidates, k=max(K_VALUES), scores_dict=pers_scores
    )
    cap_eval = evaluate_recommendations(cap_recs, relevant, K_VALUES)
    cap_sum = compute_summary(cap_eval, K_VALUES)
    cap_conc = category_concentration(cap_recs, ads, K)
    cap_pf = provider_fairness(cap_recs, ads, K)
    cap_div = exposure_divergence(cap_recs, users_s, ads, "gender", K)
    cap_jsd = cap_div["jsd"].max() if not cap_div.empty else 0.0
    cap_ba = bias_amplification(cap_recs, train, users_s, ads, "gender", K)
    cap_stereo = len(cap_ba[cap_ba["is_stereotyping"]]) if not cap_ba.empty else 0

    print(f"  {'cap30':<6} {cap_sum['Precision@10']:>8.4f} {cap_sum['Recall@10']:>8.4f} "
          f"{cap_conc['gini']:>6.3f} {cap_pf['hhi']:>6.4f} "
          f"{cap_pf['catalog_coverage']:>8.1f}% {cap_jsd:>10.4f} "
          f"{cap_stereo:>5}/24")

    # Show the best fair variant's full scorecard + bias audit
    best_lam = 0.5
    best_fair = fair_variants[f"λ={best_lam}"]
    print(f"\n  ── Detailed analysis: Personalized + MMR (λ={best_lam}) ──")

    # Fairness scorecard comparison: original vs fair
    fair_sc = fairness_scorecard(
        best_fair["recs"], best_fair["eval"], users_s, ads, relevant, K
    )
    print_fairness_report(fair_sc, system_name=f"Personalized + MMR (λ={best_lam})")

    # Side-by-side: original personalized vs fair
    all_scorecards = {
        "Popularity": scorecards["Popularity"],
        "Personalized": scorecards["Personalized"],
        f"Pers+MMR(λ={best_lam})": fair_sc,
    }
    three_way = compare_systems(all_scorecards)
    print_comparison_report(three_way, list(all_scorecards.keys()))

    # Bias audit on the fair version
    fair_audit = run_bias_audit(
        recommendations=best_fair["recs"],
        relevant_items=relevant,
        interactions=train,
        users=users_s,
        ads=ads,
        eval_df=best_fair["eval"],
        k=K,
    )
    print_bias_report(fair_audit, system_name=f"Personalized + MMR (λ={best_lam})")

    # ── 9. SAVE ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("STEP 9: Saving results")
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
