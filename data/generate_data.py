"""
Generate a synthetic social media advertising engagement dataset.

Mirrors the schema of the Kaggle 'Social Media Ad Engagement Dataset'
(https://www.kaggle.com/datasets/ziya07/social-media-ad-engagement-dataset).

Only used when the real dataset is not available. The real CSV can be placed
at data/social_media_ad_engagement.csv and main.py will load it automatically.
"""
import numpy as np
import pandas as pd
from pathlib import Path


def generate_dataset(n_users=5000, n_ads=200, n_interactions=100000, seed=42):
    """
    Generate synthetic user-ad interactions with demographic-dependent
    engagement patterns to simulate real-world bias.

    Returns:
        users (DataFrame): user_id, age, gender, location, device_type
        ads (DataFrame):   ad_id, ad_type, ad_category, ad_duration
        interactions (DataFrame): merged user+ad features with 'engaged' target
    """
    rng = np.random.default_rng(seed)

    # --- Users ---
    genders = rng.choice(["Male", "Female", "Other"], n_users, p=[0.48, 0.48, 0.04])
    ages = rng.integers(18, 65, n_users)
    locations = rng.choice(
        ["US", "UK", "Germany", "Netherlands", "France", "Brazil", "India", "Japan"],
        n_users,
        p=[0.25, 0.12, 0.10, 0.08, 0.10, 0.12, 0.15, 0.08],
    )
    devices = rng.choice(["Mobile", "Desktop", "Tablet"], n_users, p=[0.60, 0.30, 0.10])

    users = pd.DataFrame({
        "user_id": range(n_users),
        "age": ages,
        "gender": genders,
        "location": locations,
        "device_type": devices,
    })

    # --- Advertisements ---
    ad_types = rng.choice(["Image", "Video", "Text", "Carousel"], n_ads, p=[0.30, 0.35, 0.15, 0.20])
    categories = rng.choice(
        ["Technology", "Fashion", "Automotive", "Finance", "Food", "Travel", "Gaming", "Health"],
        n_ads,
        p=[0.15, 0.15, 0.10, 0.12, 0.12, 0.10, 0.14, 0.12],
    )
    durations = rng.choice([15, 30, 45, 60], n_ads, p=[0.30, 0.35, 0.20, 0.15])

    # Inherent ad quality (some ads are just better made)
    ad_quality = rng.beta(2, 5, n_ads)

    ads = pd.DataFrame({
        "ad_id": range(n_ads),
        "ad_type": ad_types,
        "ad_category": categories,
        "ad_duration": durations,
        "_quality": ad_quality,  # internal, dropped before saving
    })

    # --- Interactions ---
    user_ids = rng.integers(0, n_users, n_interactions)
    ad_ids = rng.integers(0, n_ads, n_interactions)
    time_of_day = rng.choice(["Morning", "Afternoon", "Evening", "Night"],
                              n_interactions, p=[0.20, 0.30, 0.35, 0.15])

    interactions = pd.DataFrame({
        "user_id": user_ids,
        "ad_id": ad_ids,
        "time_of_day": time_of_day,
    })
    interactions = interactions.merge(users, on="user_id").merge(ads, on="ad_id")

    # --- Engagement model (logistic with demographic-dependent patterns) ---
    logit = np.full(n_interactions, -1.5)

    # Ad quality is the strongest predictor
    logit += interactions["_quality"].values * 3.0

    # Video and Carousel ads get higher engagement
    logit += (interactions["ad_type"] == "Video").astype(float).values * 0.4
    logit += (interactions["ad_type"] == "Carousel").astype(float).values * 0.3

    # Age effects: younger users engage more with Gaming/Tech
    young = (interactions["age"].values < 30).astype(float)
    logit += young * interactions["ad_category"].isin(["Gaming", "Technology"]).astype(float).values * 0.6

    # Older users engage more with Finance/Health
    older = (interactions["age"].values >= 45).astype(float)
    logit += older * interactions["ad_category"].isin(["Finance", "Health"]).astype(float).values * 0.5

    # Gender effects (subtle bias to simulate real-world patterns)
    is_female = (interactions["gender"] == "Female").astype(float).values
    is_male = (interactions["gender"] == "Male").astype(float).values
    logit += is_female * interactions["ad_category"].isin(["Fashion", "Health"]).astype(float).values * 0.3
    logit += is_male * interactions["ad_category"].isin(["Automotive", "Gaming"]).astype(float).values * 0.3

    # Device effects: mobile slightly lower engagement
    logit -= (interactions["device_type"] == "Mobile").astype(float).values * 0.2

    # Time of day: evening is peak engagement
    logit += (interactions["time_of_day"] == "Evening").astype(float).values * 0.2

    # Location effects
    logit += interactions["location"].isin(["US", "UK"]).astype(float).values * 0.15

    # Sample binary engagement
    prob = 1 / (1 + np.exp(-logit))
    interactions["engaged"] = rng.binomial(1, prob)

    # Drop internal quality column
    interactions = interactions.drop(columns=["_quality"])
    ads = ads.drop(columns=["_quality"])

    return users, ads, interactions


if __name__ == "__main__":
    print("Generating synthetic dataset (mirrors Kaggle schema)...")
    users, ads, interactions = generate_dataset()

    out_dir = Path(__file__).parent
    users.to_csv(out_dir / "users.csv", index=False)
    ads.to_csv(out_dir / "ads.csv", index=False)
    interactions.to_csv(out_dir / "interactions.csv", index=False)

    print(f"  Users:           {len(users):,}")
    print(f"  Ads:             {len(ads):,}")
    print(f"  Interactions:    {len(interactions):,}")
    print(f"  Engagement rate: {interactions['engaged'].mean():.1%}")
    print(f"  Saved to {out_dir}")
