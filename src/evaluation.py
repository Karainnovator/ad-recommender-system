"""
Evaluation metrics for recommender systems.
Precision@K, Recall@K, AUC-ROC, and per-group analysis.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss


def get_relevant_items(test_interactions: pd.DataFrame) -> dict:
    """Get the set of ads each user actually engaged with in the test set."""
    engaged = test_interactions[test_interactions["engaged"] == 1]
    return engaged.groupby("user_id")["ad_id"].apply(set).to_dict()


def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    """Precision@K: fraction of top-K that are relevant."""
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant) / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    """Recall@K: fraction of relevant items found in top-K."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    return len(set(top_k) & relevant) / len(relevant)


def evaluate_recommendations(recommendations: dict, relevant_items: dict, k_values=[5, 10, 20]):
    """
    Evaluate a recommender system.

    Args:
        recommendations: {user_id: [ad_id, ...]} ordered by predicted relevance
        relevant_items: {user_id: {ad_id, ...}} ground truth engagements
        k_values: list of K values to evaluate

    Returns:
        DataFrame with per-user metrics and summary stats
    """
    results = []

    for user_id, rec_list in recommendations.items():
        rel = relevant_items.get(user_id, set())
        row = {"user_id": user_id, "n_relevant": len(rel)}
        for k in k_values:
            row[f"precision@{k}"] = precision_at_k(rec_list, rel, k)
            row[f"recall@{k}"] = recall_at_k(rec_list, rel, k)
        results.append(row)

    return pd.DataFrame(results)


def compute_summary(eval_df: pd.DataFrame, k_values=[5, 10, 20]) -> dict:
    """Compute mean metrics across all users."""
    summary = {}
    for k in k_values:
        summary[f"Precision@{k}"] = eval_df[f"precision@{k}"].mean()
        summary[f"Recall@{k}"] = eval_df[f"recall@{k}"].mean()
    return summary


def compute_prediction_metrics(y_true, y_pred_proba):
    """Compute AUC-ROC and Log-loss for the personalized model."""
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
