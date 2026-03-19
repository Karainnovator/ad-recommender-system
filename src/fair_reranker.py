"""
Fair re-ranking layer for recommender systems.

Applied as post-processing: takes candidate recommendations and re-ranks
them to satisfy fairness constraints while preserving relevance.

Two strategies:
  - mmr:          MMR-style greedy selection balancing relevance with category diversity
  - category_cap: hard limit — no single category takes more than X% of K slots

Usage:
    reranker = FairReranker(ads, strategy="mmr", lambda_param=0.5)
    fair_recs = reranker.rerank_all(candidate_recs, k=10, scores_dict=scores)
"""
import numpy as np
import pandas as pd


class FairReranker:
    """
    Post-processing fairness layer that re-ranks recommendations
    to balance relevance with diversity/fairness constraints.
    """

    def __init__(self, ads: pd.DataFrame, strategy: str = "mmr",
                 lambda_param: float = 0.5, category_cap: float = 0.3):
        """
        Args:
            ads:            DataFrame with ad_id, ad_category columns
            strategy:       "mmr" or "category_cap"
            lambda_param:   relevance-diversity tradeoff for MMR
                            1.0 = pure relevance, 0.0 = pure diversity
            category_cap:   max fraction of K slots per category (for category_cap)
        """
        self.strategy = strategy
        self.lambda_param = lambda_param
        self.category_cap = category_cap

        # Pre-build fast lookup: ad_id -> category
        self._cat_lookup = dict(zip(ads["ad_id"], ads["ad_category"]))

    def _get_category(self, ad_id):
        return self._cat_lookup.get(ad_id, "unknown")

    # ──────────────────────────────────────────────────────────
    # MMR (Maximal Marginal Relevance) diversity re-ranking
    # ──────────────────────────────────────────────────────────

    def _mmr_rerank(self, candidates: list, scores: list, k: int) -> list:
        """
        Greedy MMR selection.

        At each step, pick the candidate that maximizes:
            score = lambda * normalized_relevance + (1 - lambda) * diversity

        diversity(ad, selected) = 1 / (1 + count of same category in selected)
        This gives full score to new categories and diminishing returns to repeats.
        """
        if not candidates:
            return []

        max_score = max(scores) if scores else 1.0
        if max_score == 0:
            max_score = 1.0

        selected = []
        selected_cats = {}  # category -> count
        remaining = set(range(len(candidates)))

        for _ in range(min(k, len(candidates))):
            best_idx = None
            best_mmr = -float("inf")

            for idx in remaining:
                cat = self._get_category(candidates[idx])
                cat_count = selected_cats.get(cat, 0)

                rel = scores[idx] / max_score
                div = 1.0 / (1.0 + cat_count)

                mmr = self.lambda_param * rel + (1 - self.lambda_param) * div

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            if best_idx is None:
                break

            cat = self._get_category(candidates[best_idx])
            selected.append(candidates[best_idx])
            selected_cats[cat] = selected_cats.get(cat, 0) + 1
            remaining.remove(best_idx)

        return selected

    # ──────────────────────────────────────────────────────────
    # Category cap re-ranking
    # ──────────────────────────────────────────────────────────

    def _cap_rerank(self, candidates: list, scores: list, k: int) -> list:
        """
        Hard cap: no category gets more than category_cap fraction of K slots.

        Iterates candidates in relevance order, skipping any that would
        exceed the cap. Falls back to filling remaining slots if needed.
        """
        max_per_cat = max(1, int(np.ceil(k * self.category_cap)))
        selected = []
        cat_counts = {}
        skipped = []

        # Sort by score descending
        order = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)

        for idx in order:
            ad_id = candidates[idx]
            cat = self._get_category(ad_id)

            if cat_counts.get(cat, 0) < max_per_cat:
                selected.append(ad_id)
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
            else:
                skipped.append(ad_id)

            if len(selected) >= k:
                break

        # Fallback: if not enough ads passed the cap, fill from skipped
        if len(selected) < k:
            for ad_id in skipped:
                selected.append(ad_id)
                if len(selected) >= k:
                    break

        return selected

    # ──────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────

    def rerank(self, candidates: list, k: int = 10,
               scores: list = None) -> list:
        """
        Re-rank a single user's candidate list.

        Args:
            candidates: list of ad_ids, ordered by original relevance
            k:          number to select
            scores:     relevance scores (same length as candidates).
                        If None, uses rank-based scores.

        Returns:
            List of k ad_ids, re-ranked for fairness.
        """
        if not candidates:
            return []

        if scores is None:
            n = len(candidates)
            scores = [(n - i) / n for i in range(n)]

        if self.strategy == "mmr":
            return self._mmr_rerank(candidates, scores, k)
        elif self.strategy == "category_cap":
            return self._cap_rerank(candidates, scores, k)
        else:
            return candidates[:k]

    def rerank_all(self, recommendations: dict, k: int = 10,
                   scores_dict: dict = None) -> dict:
        """
        Re-rank recommendations for all users.

        Args:
            recommendations: {user_id: [ad_id, ...]} candidates (longer than k)
            k:               number to select per user
            scores_dict:     {user_id: [score, ...]} optional relevance scores

        Returns:
            {user_id: [ad_id, ...]} re-ranked, length k
        """
        result = {}
        for uid, candidates in recommendations.items():
            user_scores = scores_dict.get(uid) if scores_dict else None
            result[uid] = self.rerank(candidates, k, user_scores)
        return result
