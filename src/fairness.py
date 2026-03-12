"""
Fairness analysis for recommender systems.

Measures how recommendations are distributed across demographic groups:
  - Exposure distribution: what ad categories each group sees
  - Gini coefficient:      how concentrated recommendations are
  - Demographic parity:    whether groups receive equal treatment

These metrics directly address the project's research question:
"Do personalized recommendations systematically perform differently
 across demographic groups?"
"""
import numpy as np
import pandas as pd


def exposure_by_group(recommendations: dict, users: pd.DataFrame,
                      ads: pd.DataFrame, group_col: str, k: int = 10) -> pd.DataFrame:
    """
    Compute what ad categories each demographic group is exposed to.

    Returns:
        DataFrame where rows = group values, columns = ad categories,
        values = fraction of that group's recommendation slots.
    """
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
                "group": group_val,
                "category": ad_match.iloc[0]["ad_category"],
            })

    if not rows:
        return pd.DataFrame()

    return pd.crosstab(
        pd.DataFrame(rows)["group"],
        pd.DataFrame(rows)["category"],
        normalize="index",
    )


def gini_coefficient(values: np.ndarray) -> float:
    """
    Gini coefficient of a distribution.
    0 = perfect equality (uniform), 1 = maximum concentration.
    """
    values = np.sort(values)
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values))


def category_concentration(recommendations: dict, ads: pd.DataFrame, k: int = 10) -> dict:
    """
    How concentrated are recommendations across ad categories?

    Returns dict with:
      - gini:                    Gini coefficient
      - distribution:            {category: share}
      - n_categories_represented: how many categories appear
      - top_category_share:      share of the most recommended category
    """
    all_recs = []
    for rec_list in recommendations.values():
        all_recs.extend(rec_list[:k])

    rec_df = pd.DataFrame({"ad_id": all_recs}).merge(
        ads[["ad_id", "ad_category"]], on="ad_id"
    )
    counts = rec_df["ad_category"].value_counts(normalize=True).sort_values()

    return {
        "gini": gini_coefficient(counts.values),
        "distribution": counts.to_dict(),
        "n_categories_represented": len(counts),
        "top_category_share": counts.max(),
    }


def accuracy_by_group(eval_df: pd.DataFrame, users: pd.DataFrame,
                       group_col: str, k: int = 10) -> pd.DataFrame:
    """
    Precision@K and Recall@K broken down by demographic group.
    Reveals whether one system is systematically better/worse for certain groups.
    """
    merged = eval_df.merge(users, on="user_id")

    if group_col == "age_group" and "age_group" not in merged.columns:
        merged["age_group"] = pd.cut(
            merged["age"], bins=[17, 25, 35, 45, 65],
            labels=["18-25", "26-35", "36-45", "46-65"]
        )

    return merged.groupby(group_col, observed=True).agg({
        f"precision@{k}": "mean",
        f"recall@{k}": "mean",
    }).round(4)
