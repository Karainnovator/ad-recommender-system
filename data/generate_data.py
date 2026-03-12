"""
Generate a realistic synthetic social media advertising engagement dataset.
Simulates user-ad interactions with demographic-dependent engagement patterns.
"""
import numpy as np
import pandas as pd
from pathlib import Path


def generate_dataset(n_users=5000, n_ads=200, n_interactions=100000, seed=42):
    rng = np.random.default_rng(seed)

    # --- Users ---
    genders = rng.choice(["Male", "Female", "Non-binary"], n_users, p=[0.48, 0.48, 0.04])
    ages = rng.integers(18, 65, n_users)
    locations = rng.choice(
        ["US", "UK", "DE", "NL", "FR", "BR", "IN", "JP"],
        n_users,
        p=[0.25, 0.12, 0.10, 0.08, 0.10, 0.12, 0.15, 0.08],
    )
    devices = rng.choice(["Mobile", "Desktop", "Tablet"], n_users, p=[0.60, 0.30, 0.10])

    users = pd.DataFrame({
        "user_id": range(n_users),
        "gender": genders,
        "age": ages,
        "location": locations,
        "device": devices,
    })

    # --- Advertisements ---
    ad_types = rng.choice(["Display", "Video", "Native"], n_ads, p=[0.40, 0.35, 0.25])
    categories = rng.choice(
        ["Technology", "Fashion", "Automotive", "Finance", "Food", "Travel", "Gaming", "Health"],
        n_ads,
        p=[0.15, 0.15, 0.10, 0.12, 0.12, 0.10, 0.14, 0.12],
    )
    durations = rng.choice([15, 30, 45, 60], n_ads, p=[0.30, 0.35, 0.20, 0.15])

    # Assign each ad an inherent popularity score (some ads are just better)
    ad_base_quality = rng.beta(2, 5, n_ads)  # skewed low → most ads have low base CTR

    ads = pd.DataFrame({
        "ad_id": range(n_ads),
        "ad_type": ad_types,
        "category": categories,
        "duration": durations,
        "base_quality": ad_base_quality,
    })

    # --- Interactions ---
    user_ids = rng.integers(0, n_users, n_interactions)
    ad_ids = rng.integers(0, n_ads, n_interactions)

    interactions = pd.DataFrame({"user_id": user_ids, "ad_id": ad_ids})
    interactions = interactions.merge(users, on="user_id").merge(ads, on="ad_id")

    # --- Engagement model (introduces realistic demographic-dependent patterns) ---
    logit = np.full(n_interactions, -1.5)  # base rate ~18%

    # Ad quality matters most
    logit += interactions["base_quality"].values * 3.0

    # Video ads get higher engagement
    logit += (interactions["ad_type"] == "Video").astype(float).values * 0.4

    # Age effects: younger users engage more with Gaming/Tech
    young = (interactions["age"].values < 30).astype(float)
    logit += young * (interactions["category"].isin(["Gaming", "Technology"])).astype(float).values * 0.6
    # Older users engage more with Finance/Health
    older = (interactions["age"].values >= 45).astype(float)
    logit += older * (interactions["category"].isin(["Finance", "Health"])).astype(float).values * 0.5

    # Gender effects (subtle, to simulate real-world bias patterns)
    is_female = (interactions["gender"] == "Female").astype(float).values
    is_male = (interactions["gender"] == "Male").astype(float).values
    logit += is_female * (interactions["category"].isin(["Fashion", "Health"])).astype(float).values * 0.3
    logit += is_male * (interactions["category"].isin(["Automotive", "Gaming"])).astype(float).values * 0.3

    # Mobile users have slightly lower engagement (smaller screen, distractions)
    logit -= (interactions["device"] == "Mobile").astype(float).values * 0.2

    # Location effects
    logit += (interactions["location"].isin(["US", "UK"])).astype(float).values * 0.15

    # Convert to probability and sample
    prob = 1 / (1 + np.exp(-logit))
    interactions["engaged"] = rng.binomial(1, prob)

    # Add timestamp for realism
    interactions["timestamp"] = pd.date_range("2024-01-01", periods=n_interactions, freq="30s")

    # Drop helper columns
    interactions = interactions.drop(columns=["base_quality"])

    return users, ads, interactions


if __name__ == "__main__":
    print("Generating synthetic social media advertising dataset...")
    users, ads, interactions = generate_dataset()

    out_dir = Path(__file__).parent
    users.to_csv(out_dir / "users.csv", index=False)
    ads.to_csv(out_dir / "ads.csv", index=False)
    interactions.to_csv(out_dir / "interactions.csv", index=False)

    print(f"Users:        {len(users):,}")
    print(f"Ads:          {len(ads):,}")
    print(f"Interactions: {len(interactions):,}")
    print(f"Engagement rate: {interactions['engaged'].mean():.1%}")
    print(f"Saved to: {out_dir}")
