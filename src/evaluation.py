"""
Evaluation metrics for recommender systems.

Accuracy metrics:
  - Precision@K:  fraction of top-K that the user actually engaged with
  - Recall@K:     fraction of engaged items that appear in top-K
  - AUC-ROC:      model's ability to rank engaged above non-engaged
  - Log-loss:     calibration quality of predicted probabilities
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss


def get_relevant_items(test_interactions: pd.DataFrame) -> dict:
    """
    Build ground truth: for each user, the set of ad_ids they engaged with.

    Args:
        test_interactions: test split with 'user_id', 'ad_id', 'engaged'

    Returns:
        dict {user_id: set of ad_ids with engaged == 1}
    """
    engaged = test_interactions[test_interactions["engaged"] == 1]
    return engaged.groupby("user_id")["ad_id"].apply(set).to_dict()


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """What fraction of the top-K recommendations are relevant?"""
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant) / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """What fraction of all relevant items appear in the top-K?"""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    return len(set(top_k) & relevant) / len(relevant)


def evaluate_recommendations(recommendations: dict, relevant_items: dict,
                              k_values=(5, 10, 20)) -> pd.DataFrame:
    """
    Compute Precision@K and Recall@K for every user.

    Args:
        recommendations: {user_id: [ad_id, ...]} ranked by predicted relevance
        relevant_items:  {user_id: {ad_id, ...}} ground truth
        k_values:        tuple of K values to evaluate

    Returns:
        DataFrame with one row per user, columns per metric per K
    """
    rows = []
    for uid, rec_list in recommendations.items():
        rel = relevant_items.get(uid, set())
        row = {"user_id": uid, "n_relevant": len(rel)}
        for k in k_values:
            row[f"precision@{k}"] = precision_at_k(rec_list, rel, k)
            row[f"recall@{k}"] = recall_at_k(rec_list, rel, k)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_summary(eval_df: pd.DataFrame, k_values=(5, 10, 20)) -> dict:
    """Average Precision@K and Recall@K across all users."""
    return {
        f"{m}@{k}": eval_df[f"{m.lower()}@{k}"].mean()
        for k in k_values for m in ["Precision", "Recall"]
    }


def compute_prediction_metrics(y_true, y_pred_proba) -> dict:
    """AUC-ROC and Log-loss for the personalized model's raw predictions."""
    metrics = {}
    try:
        metrics["AUC-ROC"] = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        metrics["AUC-ROC"] = float("nan")
    try:
        metrics["Log-loss"] = log_loss(y_true, y_pred_proba)
    except ValueError:
        metrics["Log-loss"] = float("nan")
    return metrics
