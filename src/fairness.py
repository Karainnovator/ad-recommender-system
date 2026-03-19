"""
Fairness evaluation for recommender systems.

Evaluates whether the system meets fairness standards and compares
fairness across systems:

  Core metrics:
  - exposure_divergence:     JSD between groups' category distributions
  - accuracy_equity:         statistical + practical significance of accuracy gaps
  - provider_fairness:       ad-side fairness (HHI, coverage, zero-exposure)

  Utilities:
  - exposure_by_group:       what ad categories each group sees
  - gini_coefficient:        inequality measure for any distribution
  - category_concentration:  aggregate category stats

  Evaluation:
  - fairness_scorecard:      pass/warn/fail with justified thresholds
  - compare_systems:         side-by-side fairness comparison

Removed from prior version (with justification):
  - representation_ratio: always ~1.0 because both systems generate exactly K
    recommendations for every user. Only meaningful if a system skips users.
  - accuracy_by_group: subsumed by accuracy_equity, which adds statistical
    testing and effect size. Raw per-group means are available via
    accuracy_equity()["per_group"].
  - Separate "accuracy gap significance" scorecard criterion: gave PASS when
    p > 0.05, which rewards low statistical power rather than detecting
    fairness. Replaced with effect size (epsilon-squared) integrated into
    the accuracy equity assessment.
"""
import numpy as np
import pandas as pd
from scipy.stats import entropy, kruskal
from scipy.spatial.distance import jensenshannon


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


# ──────────────────────────────────────────────────────────────
# 1. Exposure by Group
# ──────────────────────────────────────────────────────────────

def exposure_by_group(recommendations: dict, users: pd.DataFrame,
                      ads: pd.DataFrame, group_col: str, k: int = 10) -> pd.DataFrame:
    """
    Compute what ad categories each demographic group is exposed to.

    Returns:
        DataFrame where rows = group values, columns = ad categories,
        values = fraction of that group's recommendation slots.
    """
    users = _add_age_group(users)
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


# ──────────────────────────────────────────────────────────────
# 2. Gini Coefficient
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# 3. Category Concentration
# ──────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────
# 4. Exposure Divergence
# ──────────────────────────────────────────────────────────────

def exposure_divergence(recommendations: dict, users: pd.DataFrame,
                        ads: pd.DataFrame, group_col: str,
                        k: int = 10) -> pd.DataFrame:
    """
    Measure how differently each pair of groups experiences the recommender
    using Jensen-Shannon divergence on their category exposure distributions.

    JSD = 0: identical exposure.  JSD = 1: completely disjoint exposure.

    Also computes KL divergence from each group to the overall distribution,
    showing which group deviates most from the average.
    """
    exposure = exposure_by_group(recommendations, users, ads, group_col, k)
    if exposure.empty:
        return pd.DataFrame()

    exposure = exposure.fillna(0)
    groups = list(exposure.index)

    # Overall average distribution
    avg_dist = exposure.mean(axis=0).values
    avg_safe = avg_dist + 1e-12
    avg_safe = avg_safe / avg_safe.sum()

    # Pairwise JSD
    pair_rows = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            p = exposure.loc[groups[i]].values
            q = exposure.loc[groups[j]].values
            # scipy jensenshannon returns the distance (sqrt of divergence)
            jsd = jensenshannon(p, q, base=2) ** 2  # divergence, not distance
            pair_rows.append({
                "group_a": groups[i],
                "group_b": groups[j],
                "jsd": round(jsd, 4),
            })

    # Per-group KL divergence from average
    group_kl = {}
    for g in groups:
        p = exposure.loc[g].values + 1e-12
        p = p / p.sum()
        group_kl[g] = round(entropy(p, avg_safe, base=2), 4)

    for row in pair_rows:
        row["kl_a_to_avg"] = group_kl[row["group_a"]]
        row["kl_b_to_avg"] = group_kl[row["group_b"]]

    return pd.DataFrame(pair_rows).sort_values("jsd", ascending=False)


# ──────────────────────────────────────────────────────────────
# 5. Accuracy Equity
# ──────────────────────────────────────────────────────────────

def accuracy_equity(eval_df: pd.DataFrame, users: pd.DataFrame,
                    group_col: str, k: int = 10) -> dict:
    """
    Statistical AND practical analysis of accuracy differences across groups.

    Returns:
      - max_gap:          absolute precision@K difference between best/worst group
      - gap_ratio:        min/max group precision (1.0 = perfect equity)
      - kruskal_p_value:  Kruskal-Wallis p-value (are differences statistically real?)
      - epsilon_squared:  effect size (practical significance, independent of sample size)
                          < 0.01 negligible, 0.01-0.06 small, 0.06-0.14 medium, >= 0.14 large
      - effect_size:      human-readable interpretation of epsilon_squared
      - per_group:        DataFrame with mean, std, n_users per group
      - worst_served:     group with lowest precision
      - best_served:      group with highest precision

    Why both statistical and practical significance?
    A large gap_ratio (e.g., 0.36) can be statistically non-significant
    if the disadvantaged group is small (low power). Conversely, a tiny
    gap can be "significant" with enough data. Effect size (epsilon-squared)
    avoids both traps — it measures how much group membership explains
    variance in accuracy, regardless of sample size.
    """
    users = _add_age_group(users)
    merged = eval_df.merge(users, on="user_id")

    if group_col == "age_group" and "age_group" not in merged.columns:
        merged["age_group"] = pd.cut(
            merged["age"], bins=[17, 25, 35, 45, 65],
            labels=["18-25", "26-35", "36-45", "46-65"]
        )

    prec_col = f"precision@{k}"
    grouped = merged.groupby(group_col, observed=True)[prec_col]

    per_group = grouped.agg(["mean", "std", "count"]).round(4)
    per_group.columns = ["mean", "std", "n_users"]

    means = per_group["mean"]
    max_gap = means.max() - means.min()
    gap_ratio = means.min() / means.max() if means.max() > 0 else 1.0

    # Kruskal-Wallis test + epsilon-squared effect size
    group_samples = [g[prec_col].values for _, g in merged.groupby(group_col, observed=True)]
    group_samples = [s for s in group_samples if len(s) > 0]

    if len(group_samples) >= 2:
        stat, p_value = kruskal(*group_samples)
        n_total = sum(len(s) for s in group_samples)
        epsilon_sq = stat / (n_total - 1) if n_total > 1 else 0.0
    else:
        stat, p_value, epsilon_sq = 0.0, 1.0, 0.0

    # Interpret effect size
    if epsilon_sq < 0.01:
        effect_size = "negligible"
    elif epsilon_sq < 0.06:
        effect_size = "small"
    elif epsilon_sq < 0.14:
        effect_size = "medium"
    else:
        effect_size = "large"

    return {
        "max_gap": round(max_gap, 4),
        "gap_ratio": round(gap_ratio, 4),
        "kruskal_statistic": round(stat, 4),
        "kruskal_p_value": round(p_value, 4),
        "epsilon_squared": round(epsilon_sq, 4),
        "effect_size": effect_size,
        "per_group": per_group,
        "worst_served": means.idxmin(),
        "best_served": means.idxmax(),
    }


# ──────────────────────────────────────────────────────────────
# 6. Provider Fairness
# ──────────────────────────────────────────────────────────────

def provider_fairness(recommendations: dict, ads: pd.DataFrame,
                      k: int = 10) -> dict:
    """
    Ad-side fairness: are all ad categories (and individual ads) getting
    a fair chance at being recommended?

    Metrics:
      - HHI (Herfindahl-Hirschman Index): standard market concentration measure
        from economics.  < 0.15 competitive, 0.15-0.25 moderate, > 0.25 concentrated.
      - catalog_coverage: % of total ads that appear in at least one recommendation
      - category_equity: per-category share vs uniform ideal
    """
    all_recs = []
    for rec_list in recommendations.values():
        all_recs.extend(rec_list[:k])

    if not all_recs:
        return {}

    rec_df = pd.DataFrame({"ad_id": all_recs}).merge(
        ads[["ad_id", "ad_category"]], on="ad_id"
    )

    # Category-level
    cat_counts = rec_df["ad_category"].value_counts(normalize=True)
    n_cats = ads["ad_category"].nunique()
    ideal_share = 1.0 / n_cats

    category_equity = []
    for cat in sorted(ads["ad_category"].unique()):
        share = cat_counts.get(cat, 0.0)
        category_equity.append({
            "category": cat,
            "share": round(share, 4),
            "ideal_share": round(ideal_share, 4),
            "deviation": round(share - ideal_share, 4),
            "ratio_to_ideal": round(share / ideal_share, 4) if ideal_share > 0 else 0.0,
        })

    # Ad-level
    ad_counts = rec_df["ad_id"].value_counts()
    all_ad_ids = set(ads["ad_id"])
    recommended_ads = set(ad_counts.index)
    zero_ads = all_ad_ids - recommended_ads

    # HHI
    hhi = (cat_counts ** 2).sum()

    return {
        "category_gini": round(gini_coefficient(cat_counts.values), 4),
        "ad_gini": round(gini_coefficient(ad_counts.values), 4),
        "zero_exposure_count": len(zero_ads),
        "zero_exposure_pct": round(len(zero_ads) / len(all_ad_ids) * 100, 1),
        "catalog_coverage": round(len(recommended_ads) / len(all_ad_ids) * 100, 1),
        "hhi": round(hhi, 4),
        "hhi_verdict": "concentrated" if hhi > 0.25 else "moderate" if hhi > 0.15 else "competitive",
        "category_equity": pd.DataFrame(category_equity).sort_values("deviation"),
    }


# ──────────────────────────────────────────────────────────────
# 7. Fairness Scorecard
# ──────────────────────────────────────────────────────────────

def fairness_scorecard(recommendations: dict, eval_df: pd.DataFrame,
                       users: pd.DataFrame, ads: pd.DataFrame,
                       relevant_items: dict, k: int = 10,
                       group_cols: list = None) -> pd.DataFrame:
    """
    Aggregate pass/warn/fail scorecard across fairness criteria.

    Threshold justifications:

    Exposure divergence (JSD, bounded [0, 1]):
      PASS < 0.10  Groups see largely similar distributions.
      WARN < 0.30  Noticeable differences but not extreme.
      FAIL >= 0.30 Groups experience substantially different recommenders.
      Source: JSD > 0.3 means the average group is more than halfway to a
      completely different distribution. Values in academic fairness literature
      typically flag divergence > 0.1 as concerning.

    Accuracy equity (gap_ratio = min/max group precision):
      PASS if gap_ratio > 0.80 OR max_gap < 0.005
           (The OR clause handles the case where precision values are tiny —
            a gap_ratio of 0.5 when the gap is 0.001 isn't meaningful.)
      WARN if gap_ratio > 0.50
      FAIL otherwise
      Source: analogous to the four-fifths guideline, adapted for continuous
      metrics rather than binary selection decisions.

    Accuracy effect size (epsilon-squared from Kruskal-Wallis):
      PASS if negligible (<0.01) or small (<0.06)
      WARN if medium (<0.14)
      FAIL if large (>=0.14)
      Source: standard non-parametric effect size interpretation.
      This replaces the old p-value criterion which rewarded small samples.

    Provider fairness — HHI:
      PASS < 0.15, WARN < 0.25, FAIL >= 0.25
      Source: U.S. DOJ/FTC horizontal merger guidelines.

    Provider fairness — catalog coverage:
      PASS > 25%, WARN > 10%, FAIL <= 10%
      Source: at K=10 with N_ads, a perfectly diverse system would show ~5%
      per user but ~25%+ across all users. Below 10% means extreme concentration.
      (Previous threshold of 50% was structurally unreachable for many systems.)
    """
    if group_cols is None:
        group_cols = ["gender", "age_group"]

    users = _add_age_group(users)
    rows = []

    for col in group_cols:
        # Exposure divergence
        div_df = exposure_divergence(recommendations, users, ads, col, k)
        if not div_df.empty:
            max_jsd = div_df["jsd"].max()
            rows.append({
                "criterion": f"Exposure divergence ({col})",
                "value": round(max_jsd, 4),
                "threshold": "PASS < 0.10 | WARN < 0.30 | FAIL >= 0.30",
                "verdict": "PASS" if max_jsd < 0.10 else "WARN" if max_jsd < 0.30 else "FAIL",
            })

        # Accuracy equity (combined practical + statistical significance)
        eq = accuracy_equity(eval_df, users, col, k)
        trivial_gap = eq["max_gap"] < 0.005
        ratio_ok = eq["gap_ratio"] > 0.80

        if trivial_gap or ratio_ok:
            verdict = "PASS"
        elif eq["gap_ratio"] > 0.50:
            verdict = "WARN"
        else:
            verdict = "FAIL"

        rows.append({
            "criterion": f"Accuracy equity ({col})",
            "value": f"ratio={eq['gap_ratio']:.2f}, gap={eq['max_gap']:.4f}",
            "threshold": "PASS if ratio > 0.80 OR gap < 0.005 | WARN if ratio > 0.50",
            "verdict": verdict,
        })

        # Effect size (replaces old p-value criterion)
        rows.append({
            "criterion": f"Accuracy effect size ({col})",
            "value": f"ε²={eq['epsilon_squared']:.4f} ({eq['effect_size']})",
            "threshold": "PASS < 0.06 | WARN < 0.14 | FAIL >= 0.14",
            "verdict": ("PASS" if eq["epsilon_squared"] < 0.06
                        else "WARN" if eq["epsilon_squared"] < 0.14
                        else "FAIL"),
        })

    # Provider fairness (group-independent)
    pf = provider_fairness(recommendations, ads, k)
    if pf:
        rows.append({
            "criterion": "Category concentration (HHI)",
            "value": pf["hhi"],
            "threshold": "PASS < 0.15 | WARN < 0.25 | FAIL >= 0.25",
            "verdict": "PASS" if pf["hhi"] < 0.15 else "WARN" if pf["hhi"] < 0.25 else "FAIL",
        })
        rows.append({
            "criterion": "Catalog coverage",
            "value": f"{pf['catalog_coverage']}%",
            "threshold": "PASS > 25% | WARN > 10% | FAIL <= 10%",
            "verdict": "PASS" if pf["catalog_coverage"] > 25 else "WARN" if pf["catalog_coverage"] > 10 else "FAIL",
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# 8. Cross-System Comparison
# ──────────────────────────────────────────────────────────────

def compare_systems(scorecards: dict) -> pd.DataFrame:
    """
    Side-by-side fairness comparison of multiple recommender systems.

    Args:
        scorecards: {system_name: scorecard_df} from fairness_scorecard()

    Returns:
        DataFrame with one row per criterion, one verdict column per system,
        plus a 'fairer' column showing which system is fairer per criterion.
    """
    system_names = list(scorecards.keys())
    if len(system_names) < 2:
        return pd.DataFrame()

    first = scorecards[system_names[0]]
    result = first[["criterion", "threshold"]].copy()

    verdict_order = {"PASS": 2, "WARN": 1, "FAIL": 0}

    for name in system_names:
        sc = scorecards[name]
        result[f"{name}_value"] = sc["value"].values
        result[f"{name}_verdict"] = sc["verdict"].values

    winners = []
    for _, row in result.iterrows():
        scores = {name: verdict_order.get(row[f"{name}_verdict"], -1)
                  for name in system_names}
        max_score = max(scores.values())
        best = [n for n, s in scores.items() if s == max_score]
        if len(best) == len(system_names):
            winners.append("tie")
        else:
            winners.append(" / ".join(best))
    result["fairer"] = winners

    return result


# ──────────────────────────────────────────────────────────────
# 9. Print Utilities
# ──────────────────────────────────────────────────────────────

def print_fairness_report(scorecard: pd.DataFrame, system_name: str = "Recommender"):
    """Print a formatted fairness scorecard."""
    print(f"\n{'=' * 60}")
    print(f"FAIRNESS SCORECARD: {system_name}")
    print(f"{'=' * 60}")

    n_pass = (scorecard["verdict"] == "PASS").sum()
    n_warn = (scorecard["verdict"] == "WARN").sum()
    n_fail = (scorecard["verdict"] == "FAIL").sum()
    total = len(scorecard)

    print(f"  Overall: {n_pass}/{total} PASS, {n_warn} WARN, {n_fail} FAIL")
    print()

    for _, row in scorecard.iterrows():
        icon = {"PASS": "[OK]", "WARN": "[!!]", "FAIL": "[XX]"}[row["verdict"]]
        print(f"  {icon} {row['criterion']}")
        print(f"       value={row['value']}")
        print(f"       {row['threshold']}")

    if n_fail > 0:
        print(f"\n  >> {n_fail} criteria FAILED — system does not meet fairness standards")
    elif n_warn > 0:
        print(f"\n  >> All criteria pass, but {n_warn} warnings need attention")
    else:
        print(f"\n  >> All criteria PASSED")


def print_comparison_report(comparison: pd.DataFrame, system_names: list):
    """Print a formatted side-by-side fairness comparison."""
    print(f"\n{'=' * 60}")
    print(f"FAIRNESS COMPARISON: {' vs '.join(system_names)}")
    print(f"{'=' * 60}")

    wins = {name: 0 for name in system_names}
    for _, row in comparison.iterrows():
        fairer = row["fairer"]
        if fairer != "tie":
            for name in system_names:
                if name in fairer:
                    wins[name] += 1

        verdicts = " | ".join(
            f"{name}: {row[f'{name}_verdict']}" for name in system_names
        )
        print(f"\n  {row['criterion']}")
        print(f"    {verdicts}  >> {row['fairer']}")

    print(f"\n  {'─' * 50}")
    print(f"  Wins: {' | '.join(f'{n}: {w}' for n, w in wins.items())}")
    overall = max(wins, key=wins.get)
    if all(v == list(wins.values())[0] for v in wins.values()):
        print(f"  Overall: TIE")
    else:
        print(f"  Overall fairer system: {overall}")
