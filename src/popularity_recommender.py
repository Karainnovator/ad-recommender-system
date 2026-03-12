"""
Popularity-Based Recommender System.
Recommends the K ads with highest historical engagement rate to ALL users.
"""
import pandas as pd
import numpy as np


class PopularityRecommender:
    def __init__(self, min_impressions=5):
        self.min_impressions = min_impressions
        self.popular_ads = None

    def fit(self, interactions: pd.DataFrame):
        """Compute popularity scores from training data."""
        ad_stats = interactions.groupby("ad_id").agg(
            impressions=("engaged", "count"),
            engagements=("engaged", "sum"),
        ).reset_index()

        # Filter out ads with too few impressions
        ad_stats = ad_stats[ad_stats["impressions"] >= self.min_impressions]
        ad_stats["engagement_rate"] = ad_stats["engagements"] / ad_stats["impressions"]
        ad_stats = ad_stats.sort_values("engagement_rate", ascending=False)

        self.popular_ads = ad_stats
        return self

    def recommend(self, user_id, k=10):
        """Return top-K ad_ids for any user (same for everyone)."""
        return self.popular_ads.head(k)["ad_id"].tolist()

    def recommend_all(self, user_ids, k=10):
        """Return top-K recommendations for a list of users."""
        top_k = self.popular_ads.head(k)["ad_id"].tolist()
        return {uid: top_k for uid in user_ids}

    def get_scores(self):
        """Return the full popularity ranking."""
        return self.popular_ads.copy()
