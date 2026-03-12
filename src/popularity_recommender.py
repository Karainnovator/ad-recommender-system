"""
Popularity-Based Recommender System (Baseline).

Recommends the K advertisements with the highest historical engagement rate
to ALL users equally. This is a non-personalized baseline that reflects how
engagement-driven ranking concentrates exposure on already popular content.
"""
import pandas as pd


class PopularityRecommender:
    """
    Non-personalized recommender that ranks ads by global engagement rate.
    Every user receives the exact same recommendation list.
    """

    def __init__(self, min_impressions=5):
        """
        Args:
            min_impressions: minimum number of times an ad must have been shown
                             to be eligible for recommendation (avoids noise from
                             ads with very few views).
        """
        self.min_impressions = min_impressions
        self.popular_ads = None

    def fit(self, interactions: pd.DataFrame):
        """
        Compute engagement rate per ad from training data.

        Engagement rate = number of engagements / number of impressions.
        Ads with fewer than min_impressions are filtered out.
        """
        stats = interactions.groupby("ad_id").agg(
            impressions=("engaged", "count"),
            engagements=("engaged", "sum"),
        ).reset_index()

        stats = stats[stats["impressions"] >= self.min_impressions]
        stats["engagement_rate"] = stats["engagements"] / stats["impressions"]
        stats = stats.sort_values("engagement_rate", ascending=False)

        self.popular_ads = stats
        return self

    def recommend(self, user_id, k=10):
        """Return top-K ad_ids. Same list for every user."""
        return self.popular_ads.head(k)["ad_id"].tolist()

    def recommend_all(self, user_ids, k=10):
        """Return top-K recommendations for a list of users."""
        top_k = self.popular_ads.head(k)["ad_id"].tolist()
        return {uid: top_k for uid in user_ids}

    def get_scores(self):
        """Return the full popularity ranking as a DataFrame."""
        return self.popular_ads.copy()
