"""
Bias detection and measurement for recommender systems.

Detects where and how much the system introduces or amplifies systematic bias:

  - equal_opportunity:       TPR parity across demographic groups
  - calibration_bias:        prediction calibration per group (ECE)
  - diversity_by_group:      per-group recommendation diversity (entropy)
  - bias_amplification:      data vs. recommendation rates with stereotyping detection
  - intersectional_bias:     compound disadvantage at attribute intersections

Removed from prior version (with justification):
  - disparate_impact (80% rule): designed for binary hire/no-hire decisions in
    employment law, not applicable to multi-category recommendation distributions
    where differential exposure can reflect legitimate preference differences
  - demographic_parity: redundant with exposure_divergence in fairness.py;
    JSD is a strictly better measure of distributional differences
"""
import numpy as np
import pandas as pd
from scipy.stats import entropy


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Add age_group column if missing."""
    if "age_group" not in df.columns and "age" in df.columns:
        df = df.copy()
        df["age_group"] = pd.cut(
            df["age"], bins=[17, 25, 35, 45, 65],
            labels=["18-25", "26-35", "36-45", "46-65"],
        )
    return df


def _build_rec_df(recommendations: dict, users: pd.DataFrame,
                  ads: pd.DataFrame, group_col: str, k: int) -> pd.DataFrame:
    """Flatten recommendations into a DataFrame with group and category."""
    rows = []
    for uid, rec_list in recommendations.items():
        user_match = users[users["user_id"] == uid]
        if user_match.empty:
            continue
        group_val = user_match.iloc[0][group_col]
        for ad_id in rec_list[:k]:
            ad_match = ads[ads["ad_id"] == ad_id]
            if ad_match.empty:
                continue
            rows.append({
                "user_id": uid,
                "group": group_val,
                "ad_id": ad_id,
                "category": ad_match.iloc[0]["ad_category"],
            })
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# 1. Equal Opportunity (TPR parity)
# ──────────────────────────────────────────────────────────────

def equal_opportunity(recommendations: dict, relevant_items: dict,
                      users: pd.DataFrame, group_col: str,
                      k: int = 10) -> pd.DataFrame:
    """
    True-positive rate per demographic group.

    TPR = |recommended ∩ relevant| / |relevant| for each user,
    then averaged per group. This is equivalent to Recall@K by group,
    but framed in the ML fairness literature as Equal Opportunity.

    A fair system has equal TPR across groups — meaning it finds
    relevant items equally well regardless of demographics.
    """
    users = _add_age_group(users)
    rows = []
    for uid, rec_list in recommendations.items():
        rel = relevant_items.get(uid, set())
        if not rel:
            continue
        user_match = users[users["user_id"] == uid]
        if user_match.empty:
            continue
        tpr = len(set(rec_list[:k]) & rel) / len(rel)
        rows.append({
            "group": user_match.iloc[0][group_col],
            "tpr": tpr,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    summary = df.groupby("group")["tpr"].agg(["mean", "std", "count"])
    summary.columns = ["tpr_mean", "tpr_std", "n_users"]
    summary["tpr_gap"] = summary["tpr_mean"].max() - summary["tpr_mean"]
    return summary


# ──────────────────────────────────────────────────────────────
# 2. Calibration Bias (personalized model only)
# ──────────────────────────────────────────────────────────────

def calibration_by_group(y_true: np.ndarray, y_pred: np.ndarray,
                         groups: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Per-group calibration: compare predicted vs actual engagement rates.

    For each group, bin predictions into deciles and compute:
      - mean predicted probability
      - actual engagement rate
      - calibration error (absolute difference)

    A biased model may be well-calibrated overall but miscalibrated
    for minority groups.
    """
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "group": groups,
    })

    rows = []
    for group_val, gdf in df.groupby("group"):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        gdf = gdf.copy()
        gdf["bin"] = pd.cut(gdf["y_pred"], bins=bin_edges, include_lowest=True)

        for bin_label, bdf in gdf.groupby("bin", observed=True):
            if len(bdf) < 5:
                continue
            rows.append({
                "group": group_val,
                "bin": str(bin_label),
                "n_samples": len(bdf),
                "mean_predicted": bdf["y_pred"].mean(),
                "actual_rate": bdf["y_true"].mean(),
                "calibration_error": abs(bdf["y_pred"].mean() - bdf["y_true"].mean()),
            })

    return pd.DataFrame(rows)


def calibration_summary(y_true: np.ndarray, y_pred: np.ndarray,
                        groups: np.ndarray) -> pd.DataFrame:
    """
    Expected Calibration Error (ECE) per demographic group.

    ECE = weighted average of |predicted - actual| across bins.
    """
    detail = calibration_by_group(y_true, y_pred, groups)
    if detail.empty:
        return pd.DataFrame()

    def ece(gdf):
        total = gdf["n_samples"].sum()
        if total == 0:
            return 0.0
        return (gdf["calibration_error"] * gdf["n_samples"]).sum() / total

    result = detail.groupby("group").apply(ece).reset_index()
    result.columns = ["group", "ECE"]
    result = result.sort_values("ECE", ascending=False)
    return result


# ──────────────────────────────────────────────────────────────
# 3. Diversity by Group
# ──────────────────────────────────────────────────────────────

def diversity_by_group(recommendations: dict, users: pd.DataFrame,
                       ads: pd.DataFrame, group_col: str,
                       k: int = 10) -> pd.DataFrame:
    """
    Per-group recommendation diversity using Shannon entropy and coverage.

    - entropy:  higher = more diverse category distribution
    - coverage: fraction of total ad catalog the group is exposed to
    - n_categories: how many distinct categories appear

    Groups with low diversity are being pigeonholed into narrow content.
    """
    users = _add_age_group(users)
    rec_df = _build_rec_df(recommendations, users, ads, group_col, k)
    if rec_df.empty:
        return pd.DataFrame()

    total_ads = ads["ad_id"].nunique()
    all_categories = ads["ad_category"].nunique()

    rows = []
    for group_val, gdf in rec_df.groupby("group"):
        cat_counts = gdf["category"].value_counts(normalize=True)
        rows.append({
            "group": group_val,
            "entropy": entropy(cat_counts.values, base=2),
            "max_entropy": np.log2(all_categories),
            "normalized_entropy": entropy(cat_counts.values, base=2) / np.log2(all_categories),
            "n_categories": gdf["category"].nunique(),
            "catalog_coverage": gdf["ad_id"].nunique() / total_ads,
            "n_users": gdf["user_id"].nunique(),
        })

    return pd.DataFrame(rows).sort_values("entropy")


# ──────────────────────────────────────────────────────────────
# 4. Bias Amplification (with stereotyping detection)
# ──────────────────────────────────────────────────────────────

def bias_amplification(recommendations: dict, interactions: pd.DataFrame,
                       users: pd.DataFrame, ads: pd.DataFrame,
                       group_col: str, k: int = 10) -> pd.DataFrame:
    """
    Compare engagement rates in the data vs. recommendation exposure,
    with stereotyping detection.

    For each (group, category):
      - data_rate:       group's engagement share with this category
      - overall_rate:    all groups' engagement share (for context)
      - rec_rate:        recommendation share from the system
      - amplification:   rec_rate - data_rate (how much the system changes it)
      - is_stereotyping: True when the system reinforces an existing above-average
                         association (filter bubble) or deepens an existing
                         below-average exclusion

    Stereotyping means the system pushes groups further toward what they already
    lean toward. This is distinct from bias amplification in general — a system
    could amplify a pattern without stereotyping (e.g., showing more of a
    category that the group doesn't particularly lean toward).
    """
    users = _add_age_group(users)

    # Data-level rates
    inter = interactions.copy()
    if group_col not in inter.columns:
        inter = inter.merge(users[["user_id", group_col]], on="user_id")
    if "ad_category" not in inter.columns:
        inter = inter.merge(ads[["ad_id", "ad_category"]], on="ad_id")
    inter = _add_age_group(inter)

    engaged = inter[inter["engaged"] == 1]
    data_rates = pd.crosstab(
        engaged[group_col], engaged["ad_category"], normalize="index"
    )

    # Overall category engagement rates (across all groups)
    overall_rates = engaged["ad_category"].value_counts(normalize=True)

    # Recommendation-level rates
    rec_df = _build_rec_df(recommendations, users, ads, group_col, k)
    if rec_df.empty:
        return pd.DataFrame()

    rec_rates = pd.crosstab(rec_df["group"], rec_df["category"], normalize="index")

    # Align
    all_cats = sorted(set(data_rates.columns) | set(rec_rates.columns))
    data_rates = data_rates.reindex(columns=all_cats, fill_value=0)
    rec_rates = rec_rates.reindex(columns=all_cats, fill_value=0)
    overall_rates = overall_rates.reindex(all_cats, fill_value=0)

    common_groups = sorted(set(data_rates.index) & set(rec_rates.index))
    data_rates = data_rates.loc[common_groups]
    rec_rates = rec_rates.loc[common_groups]

    rows = []
    for group in common_groups:
        for cat in all_cats:
            d_rate = data_rates.loc[group, cat]
            r_rate = rec_rates.loc[group, cat]
            o_rate = overall_rates[cat]
            amplification = r_rate - d_rate

            # Stereotyping detection:
            # Group already leans toward this category (above overall average)
            # AND the system pushes them further in that direction.
            # OR: group already leans away AND system pushes them further away.
            above_avg = d_rate > o_rate + 0.005  # small tolerance for noise
            below_avg = d_rate < o_rate - 0.005
            is_stereotyping = (
                (amplification > 0.02 and above_avg) or
                (amplification < -0.02 and below_avg)
            )

            rows.append({
                "group": group,
                "category": cat,
                "data_rate": round(d_rate, 4),
                "overall_rate": round(o_rate, 4),
                "rec_rate": round(r_rate, 4),
                "amplification": round(amplification, 4),
                "amplification_pct": round(amplification * 100, 2),
                "is_stereotyping": is_stereotyping,
            })

    df = pd.DataFrame(rows)
    return df.sort_values("amplification", key=abs, ascending=False)


# ──────────────────────────────────────────────────────────────
# 5. Intersectional Bias (gender × age_group)
# ──────────────────────────────────────────────────────────────

def intersectional_bias(eval_df: pd.DataFrame, users: pd.DataFrame,
                        k: int = 10, min_group_size: int = 5) -> pd.DataFrame:
    """
    Analyze accuracy at intersections of gender × age_group.

    Detects compound disadvantage: when membership in two groups
    simultaneously produces worse outcomes than either group alone.

    For each intersection:
      - precision_mean:           actual precision@K
      - gender_avg / age_avg:     single-dimension averages
      - worst_single_dim:         min(gender_avg, age_avg)
      - compound_penalty:         actual - worst_single_dim
                                  (negative = compound disadvantage)

    A negative compound_penalty means the intersection is served worse
    than predicted from either dimension individually. This can reveal
    hidden bias against specific demographic combinations.

    Intersections with fewer than min_group_size users are excluded
    to avoid noise from tiny samples.
    """
    users = _add_age_group(users)
    merged = eval_df.merge(
        users[["user_id", "gender", "age_group"]], on="user_id"
    )

    prec_col = f"precision@{k}"
    if prec_col not in merged.columns:
        return pd.DataFrame()

    # Single-dimension averages
    gender_avg = merged.groupby("gender")[prec_col].mean()
    age_avg = merged.groupby("age_group", observed=True)[prec_col].mean()

    rows = []
    for (gender, age_group), gdf in merged.groupby(["gender", "age_group"], observed=True):
        if len(gdf) < min_group_size:
            continue

        actual = gdf[prec_col].mean()
        g_avg = gender_avg.get(gender, actual)
        a_avg = age_avg.get(age_group, actual)
        worst_single = min(g_avg, a_avg)
        compound_penalty = actual - worst_single

        rows.append({
            "gender": gender,
            "age_group": str(age_group),
            "n_users": len(gdf),
            "precision_mean": round(actual, 4),
            "gender_avg": round(g_avg, 4),
            "age_avg": round(a_avg, 4),
            "worst_single_dim": round(worst_single, 4),
            "compound_penalty": round(compound_penalty, 4),
            "has_compound_disadvantage": compound_penalty < -0.002,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("compound_penalty")


# ──────────────────────────────────────────────────────────────
# 6. Full Bias Audit
# ──────────────────────────────────────────────────────────────

def run_bias_audit(recommendations: dict, relevant_items: dict,
                   interactions: pd.DataFrame, users: pd.DataFrame,
                   ads: pd.DataFrame, eval_df: pd.DataFrame = None,
                   k: int = 10, group_cols: list = None) -> dict:
    """
    Run all bias metrics for a single recommender system.

    Args:
        recommendations: {user_id: [ad_id, ...]}
        relevant_items:  {user_id: {ad_id, ...}} ground truth
        interactions:    training data
        users:           user demographics
        ads:             ad features
        eval_df:         per-user evaluation results (for intersectional analysis)
        k:               top-K for analysis
        group_cols:      protected attributes to audit (default: gender, age_group)

    Returns:
        Nested dict: {group_col: {metric_name: result_df}}
        Plus "intersectional" key if eval_df is provided.
    """
    if group_cols is None:
        group_cols = ["gender", "age_group"]

    users = _add_age_group(users)
    results = {}

    for col in group_cols:
        results[col] = {
            "equal_opportunity": equal_opportunity(
                recommendations, relevant_items, users, col, k
            ),
            "diversity": diversity_by_group(
                recommendations, users, ads, col, k
            ),
            "bias_amplification": bias_amplification(
                recommendations, interactions, users, ads, col, k
            ),
        }

    # Intersectional analysis (gender × age_group)
    if eval_df is not None:
        results["intersectional"] = intersectional_bias(eval_df, users, k)

    return results


def print_bias_report(audit: dict, system_name: str = "Recommender"):
    """Print a human-readable summary of the bias audit."""
    print(f"\n{'=' * 60}")
    print(f"BIAS AUDIT: {system_name}")
    print(f"{'=' * 60}")

    for group_col, metrics in audit.items():
        # Intersectional results handled separately
        if group_col == "intersectional":
            df = metrics
            if df is None or df.empty:
                continue
            print(f"\n  Intersectional analysis (gender × age_group)")
            print(f"  {'─' * 50}")

            disadvantaged = df[df["has_compound_disadvantage"]]
            if len(disadvantaged) > 0:
                print(f"  {len(disadvantaged)} intersection(s) with compound disadvantage:")
                for _, row in disadvantaged.iterrows():
                    print(f"    {row['gender']} {row['age_group']} "
                          f"(n={row['n_users']}): "
                          f"precision={row['precision_mean']:.4f} "
                          f"vs worst single dim={row['worst_single_dim']:.4f} "
                          f"(penalty={row['compound_penalty']:.4f})")
            else:
                print(f"  No compound disadvantage detected")

            best = df.iloc[-1]
            worst = df.iloc[0]
            print(f"\n  Best:  {best['gender']} {best['age_group']} "
                  f"(precision={best['precision_mean']:.4f}, n={best['n_users']})")
            print(f"  Worst: {worst['gender']} {worst['age_group']} "
                  f"(precision={worst['precision_mean']:.4f}, n={worst['n_users']})")
            continue

        print(f"\n  Protected attribute: {group_col}")
        print(f"  {'─' * 50}")

        # Equal opportunity
        eo = metrics["equal_opportunity"]
        if not eo.empty:
            print(f"\n  Equal Opportunity (TPR by group):")
            for group in eo.index:
                row = eo.loc[group]
                gap_str = f"  (gap: {row['tpr_gap']:.4f})" if row["tpr_gap"] > 0 else ""
                print(f"    {group}: TPR={row['tpr_mean']:.4f} "
                      f"(n={int(row['n_users'])}){gap_str}")

        # Diversity
        div = metrics["diversity"]
        if not div.empty:
            print(f"\n  Diversity (normalized entropy):")
            for _, row in div.iterrows():
                print(f"    {row['group']}: {row['normalized_entropy']:.3f} "
                      f"({row['n_categories']} categories, "
                      f"{row['catalog_coverage']:.1%} catalog)")

        # Bias amplification with stereotyping
        ba = metrics["bias_amplification"]
        if not ba.empty:
            stereotyping = ba[ba["is_stereotyping"]]
            total_pairs = len(ba)
            print(f"\n  Bias amplification: "
                  f"{len(stereotyping)}/{total_pairs} pairs show stereotyping")

            top = ba.head(5)
            for _, row in top.iterrows():
                flag = " [STEREOTYPING]" if row["is_stereotyping"] else ""
                direction = "amplified" if row["amplification"] > 0 else "dampened"
                print(f"    {row['group']}×{row['category']}: "
                      f"data={row['data_rate']:.1%} → rec={row['rec_rate']:.1%} "
                      f"({row['amplification_pct']:+.1f}pp {direction}){flag}")
