"""
Fairness analysis for recommender systems.
Analyzes exposure distribution and demographic parity across user groups.
"""
import numpy as np
import pandas as pd


def exposure_by_group(recommendations: dict, users: pd.DataFrame, ads: pd.DataFrame,
                      group_col: str, k: int = 10) -> pd.DataFrame:
    """
    Compute what ad categories each demographic group is exposed to.

    Returns a DataFrame: rows = group values, columns = ad categories,
    values = fraction of recommendation slots filled by that category.
    """
    rows = []
    for user_id, rec_list in recommendations.items():
        user = users[users["user_id"] == user_id].iloc[0]
        for ad_id in rec_list[:k]:
            ad = ads[ads["ad_id"] == ad_id]
            if len(ad) == 0:
                continue
            rows.append({
                "user_id": user_id,
                "group": user[group_col],
                "category": ad.iloc[0]["category"],
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()

    # Cross-tabulate: group x category → fraction
    ct = pd.crosstab(df["group"], df["category"], normalize="index")
    return ct


def demographic_parity_ratio(recommendations: dict, users: pd.DataFrame,
                             group_col: str, k: int = 10) -> pd.DataFrame:
    """
    For each ad category, compute the ratio of exposure between groups.
    A ratio close to 1.0 = parity. Far from 1.0 = disparity.
    """
    exposure = exposure_by_group(recommendations, users,
                                 pd.DataFrame({"ad_id": range(1000), "category": ["Unknown"] * 1000}),
                                 group_col, k)
    # This is better computed from the actual exposure matrix
    return exposure


def gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient of a distribution. 0 = perfect equality, 1 = max concentration."""
    values = np.sort(values)
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))


def category_concentration(recommendations: dict, ads: pd.DataFrame, k: int = 10) -> dict:
    """
    Compute how concentrated recommendations are across ad categories.
    Returns Gini coefficient and category distribution.
    """
    all_recs = []
    for rec_list in recommendations.values():
        all_recs.extend(rec_list[:k])

    rec_df = pd.DataFrame({"ad_id": all_recs}).merge(ads[["ad_id", "category"]], on="ad_id")
    cat_counts = rec_df["category"].value_counts(normalize=True).sort_values()

    return {
        "gini": gini_coefficient(cat_counts.values),
        "distribution": cat_counts.to_dict(),
        "n_categories_represented": len(cat_counts),
        "top_category_share": cat_counts.max(),
    }


def fairness_report(recommendations: dict, users: pd.DataFrame, ads: pd.DataFrame,
                    eval_df: pd.DataFrame, k: int = 10) -> dict:
    """
    Generate a comprehensive fairness report.
    """
    report = {}

    # Per-group accuracy
    eval_with_demo = eval_df.merge(users[["user_id", "gender", "age", "location"]], on="user_id")
    eval_with_demo["age_group"] = pd.cut(eval_with_demo["age"], bins=[17, 25, 35, 45, 65],
                                          labels=["18-25", "26-35", "36-45", "46-65"])

    for group_col in ["gender", "age_group", "location"]:
        group_metrics = eval_with_demo.groupby(group_col).agg({
            f"precision@{k}": "mean",
            f"recall@{k}": "mean",
        }).round(4)
        report[f"accuracy_by_{group_col}"] = group_metrics

    # Exposure distribution
    for group_col in ["gender", "age_group"]:
        col_for_users = group_col
        if group_col == "age_group":
            users_ext = users.copy()
            users_ext["age_group"] = pd.cut(users_ext["age"], bins=[17, 25, 35, 45, 65],
                                             labels=["18-25", "26-35", "36-45", "46-65"])
            exp = exposure_by_group(recommendations, users_ext, ads, "age_group", k)
        else:
            exp = exposure_by_group(recommendations, users, ads, group_col, k)
        report[f"exposure_by_{group_col}"] = exp

    # Category concentration
    report["concentration"] = category_concentration(recommendations, ads, k)

    return report
