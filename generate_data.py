"""
generate_data.py
Creates a synthetic dataset of anime/MMORPG reviews (MyAnimeList / Anilist style)
and saves it to data/anime_reviews.csv.

Run once to (re)generate the sample dataset:
    python generate_data.py
"""

import random
import pandas as pd
from datetime import date

random.seed(42)

TITLES = {
    "Shonen Action": ["Blade Requiem", "Ashen Fang", "Crimson Arc", "Void Hunter Z"],
    "Isekai": ["Reborn as a Vending Machine God", "Another World Chronicles", "Loop of Eternity"],
    "Slice of Life": ["Quiet Afternoons", "The Tea House Diaries", "Small Town Sketchbook"],
    "Romance": ["Cherry Blossom Letters", "Two Seats Apart", "Midnight Confession"],
    "Fantasy MMORPG": ["Aetherlands Online", "Shattered Realms", "Bastion Ascendant"],
    "Sci-Fi MMORPG": ["Nova Frontier", "Orbital Drift", "Starforge Online"],
    "Horror": ["Hollow Static", "The Withering House", "Red Corridor"],
    "Sports": ["Full Court Legends", "Sprint to Glory", "Iron Grip Wrestling"],
}

SEASONS = ["Winter", "Spring", "Summer", "Fall"]
YEARS = [2023, 2024, 2025]

POSITIVE_SNIPPETS = [
    "The pacing kept me hooked from episode one.",
    "Character growth felt earned and emotional.",
    "Combat animation and sound design are top tier.",
    "The world-building is some of the best I've seen this year.",
    "Genuinely made me tear up by the finale.",
    "The soundtrack elevates every major scene.",
    "Grinding actually feels rewarding instead of tedious.",
    "The community events tie into the story really well.",
]

NEGATIVE_SNIPPETS = [
    "Pacing dragged badly in the middle stretch.",
    "The main character's decisions made no sense.",
    "Pay-to-win mechanics ruin the late game.",
    "Animation quality noticeably dipped after episode 6.",
    "Plot armor got exhausting by the second arc.",
    "Server queues and lag made raids unplayable.",
    "Felt like a rehash of better shows in the genre.",
    "The ending wrapped up way too fast and left plot holes.",
]

NEUTRAL_SNIPPETS = [
    "It's fine, nothing special but not bad either.",
    "Solid if you already like the genre, skippable otherwise.",
    "Some episodes are great, others are filler.",
    "Decent grind loop but the story is forgettable.",
    "Visuals are nice but the writing is average.",
    "Worth a watch/play once, not sure about replay value.",
]

REVIEWERS = [
    "otaku_wanderer", "grindmaster99", "sakura_scribe", "loot_goblin",
    "seasonal_watcher", "questline_kid", "midnight_marathoner", "guildless_gary",
    "arcane_archivist", "casual_clearer",
]


def random_review_text(sentiment: str) -> str:
    if sentiment == "positive":
        pool = POSITIVE_SNIPPETS
    elif sentiment == "negative":
        pool = NEGATIVE_SNIPPETS
    else:
        pool = NEUTRAL_SNIPPETS
    return " ".join(random.sample(pool, k=random.randint(1, 2)))


def build_dataset(n_rows: int = 300) -> pd.DataFrame:
    rows = []
    for i in range(1, n_rows + 1):
        genre = random.choice(list(TITLES.keys()))
        title = random.choice(TITLES[genre])
        season = random.choice(SEASONS)
        year = random.choice(YEARS)

        # bias genres slightly so trends look realistic across seasons
        sentiment_weights = [0.55, 0.25, 0.20]
        if genre in ("Isekai", "Fantasy MMORPG") and season in ("Fall", "Winter"):
            sentiment_weights = [0.65, 0.20, 0.15]
        sentiment = random.choices(
            ["positive", "negative", "neutral"], weights=sentiment_weights
        )[0]

        rating = {
            "positive": random.randint(7, 10),
            "neutral": random.randint(5, 7),
            "negative": random.randint(1, 5),
        }[sentiment]

        rows.append(
            {
                "review_id": i,
                "title": title,
                "genre": genre,
                "season": season,
                "year": year,
                "reviewer": random.choice(REVIEWERS),
                "rating": rating,
                "review_text": random_review_text(sentiment),
                "review_date": date(year, {"Winter": 1, "Spring": 4, "Summer": 7, "Fall": 10}[season], random.randint(1, 28)).isoformat(),
            }
        )

    df = pd.DataFrame(rows)
    return df


if __name__ == "__main__":
    df = build_dataset(300)
    df.to_csv("data/anime_reviews.csv", index=False)
    print(f"Wrote {len(df)} rows to data/anime_reviews.csv")
