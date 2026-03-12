"""
Personalized Recommender System.

Uses a LightGBM gradient-boosted decision tree to predict the probability
that a specific user will engage with a specific advertisement, based on:
  - User features: age, gender, location, device_type
  - Ad features:   ad_type, ad_category, ad_duration

For each user, ads are ranked by predicted engagement probability and the
top-K are recommended.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class PersonalizedRecommender:
    """
    Content-based personalized recommender using LightGBM for
    click-through rate (CTR) prediction.
    """

    # Columns used as categorical features
    CATEGORICAL = ["gender", "location", "device_type", "ad_type", "ad_category"]
    # Columns used as numerical features
    NUMERICAL = ["age", "ad_duration"]

    def __init__(self):
        self.model = None
        self.encoders = {}  # LabelEncoder per categorical column
        self.feature_cols = None

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _encode(self, df: pd.DataFrame, fit=False) -> pd.DataFrame:
        """Label-encode categorical columns for LightGBM."""
        df = df.copy()
        for col in self.CATEGORICAL:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col + "_enc"] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
            else:
                le = self.encoders[col]
                mapping = {label: i for i, label in enumerate(le.classes_)}
                df[col + "_enc"] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
        return df

    def _feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select encoded categorical + numerical columns as feature matrix."""
        enc = [c + "_enc" for c in self.CATEGORICAL if c + "_enc" in df.columns]
        num = [c for c in self.NUMERICAL if c in df.columns]
        return df[enc + num]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, interactions: pd.DataFrame):
        """
        Train engagement prediction model on historical interactions.

        Args:
            interactions: DataFrame with user features, ad features, and
                          binary 'engaged' column.
        """
        import lightgbm as lgb

        df = self._encode(interactions, fit=True)
        X = self._feature_matrix(df)
        y = df["engaged"]
        self.feature_cols = X.columns.tolist()

        cat_features = [c + "_enc" for c in self.CATEGORICAL if c + "_enc" in X.columns]
        train_set = lgb.Dataset(X, label=y, categorical_feature=cat_features)

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        self.model = lgb.train(params, train_set, num_boost_round=200)
        return self

    # ------------------------------------------------------------------
    # Prediction & recommendation
    # ------------------------------------------------------------------

    def predict(self, interactions: pd.DataFrame) -> np.ndarray:
        """Predict P(engaged) for each user-ad pair."""
        df = self._encode(interactions, fit=False)
        X = self._feature_matrix(df)
        return self.model.predict(X)

    def recommend(self, user_row: dict, candidate_ads: pd.DataFrame, k=10):
        """Recommend top-K ads for a single user."""
        pairs = candidate_ads.copy()
        for col, val in user_row.items():
            pairs[col] = val

        scores = self.predict(pairs)
        pairs["score"] = scores
        top_k = pairs.nlargest(k, "score")
        return top_k["ad_id"].tolist(), top_k["score"].tolist()

    def recommend_all(self, users: pd.DataFrame, ads: pd.DataFrame, k=10):
        """
        Generate top-K recommendations for every user.

        Returns:
            recommendations: dict {user_id: [ad_id, ...]}
            scores:          dict {user_id: [score, ...]}
        """
        recommendations = {}
        scores_dict = {}

        for _, user in users.iterrows():
            pairs = ads.copy()
            for col in ["user_id", "age", "gender", "location", "device_type"]:
                if col in user.index:
                    pairs[col] = user[col]

            preds = self.predict(pairs)
            top_idx = np.argsort(preds)[::-1][:k]
            recommendations[user["user_id"]] = ads.iloc[top_idx]["ad_id"].tolist()
            scores_dict[user["user_id"]] = preds[top_idx].tolist()

        return recommendations, scores_dict
