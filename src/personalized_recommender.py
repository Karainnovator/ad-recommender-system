"""
Personalized Recommender System.
Uses LightGBM to predict P(engagement | user, ad) and recommends top-K per user.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class PersonalizedRecommender:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.categorical_cols = ["gender", "location", "device", "ad_type", "category"]
        self.feature_cols = None

    def _encode(self, df: pd.DataFrame, fit=False) -> pd.DataFrame:
        """Label-encode categorical columns."""
        df = df.copy()
        for col in self.categorical_cols:
            if col not in df.columns:
                continue
            if fit:
                le = LabelEncoder()
                df[col + "_enc"] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # Handle unseen labels
                df[col + "_enc"] = df[col].astype(str).map(
                    {label: idx for idx, label in enumerate(le.classes_)}
                ).fillna(-1).astype(int)
        return df

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract feature matrix."""
        enc_cols = [c + "_enc" for c in self.categorical_cols if c + "_enc" in df.columns]
        num_cols = ["age", "duration"]
        num_cols = [c for c in num_cols if c in df.columns]
        return df[enc_cols + num_cols]

    def fit(self, interactions: pd.DataFrame):
        """Train the engagement prediction model."""
        import lightgbm as lgb

        df = self._encode(interactions, fit=True)
        X = self._get_features(df)
        y = df["engaged"]

        self.feature_cols = X.columns.tolist()

        train_data = lgb.Dataset(X, label=y, categorical_feature=[
            c + "_enc" for c in self.categorical_cols if c + "_enc" in X.columns
        ])

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

        self.model = lgb.train(params, train_data, num_boost_round=200)
        return self

    def predict(self, interactions: pd.DataFrame) -> np.ndarray:
        """Predict engagement probability for user-ad pairs."""
        df = self._encode(interactions, fit=False)
        X = self._get_features(df)
        return self.model.predict(X)

    def recommend(self, user_row: dict, candidate_ads: pd.DataFrame, k=10):
        """Recommend top-K ads for a single user."""
        # Create user-ad pairs
        pairs = candidate_ads.copy()
        for col, val in user_row.items():
            pairs[col] = val

        scores = self.predict(pairs)
        pairs["score"] = scores
        top_k = pairs.nlargest(k, "score")
        return top_k["ad_id"].tolist(), top_k["score"].tolist()

    def recommend_all(self, users: pd.DataFrame, ads: pd.DataFrame, k=10):
        """Recommend top-K ads for all users. Returns dict {user_id: [ad_ids]}."""
        recommendations = {}
        scores_dict = {}

        for _, user in users.iterrows():
            pairs = ads.copy()
            for col in ["user_id", "gender", "age", "location", "device"]:
                pairs[col] = user[col]

            preds = self.predict(pairs)
            top_idx = np.argsort(preds)[::-1][:k]
            recommendations[user["user_id"]] = ads.iloc[top_idx]["ad_id"].tolist()
            scores_dict[user["user_id"]] = preds[top_idx].tolist()

        return recommendations, scores_dict
