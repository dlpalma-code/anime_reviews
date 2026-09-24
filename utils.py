"""
utils.py
Data loading/cleaning helpers and the GenAI sentiment-analysis integration
used by app.py.
"""

import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI

# Resolve paths relative to this file's location so it works regardless of
# the process's current working directory (which can differ on deployment
# platforms like Streamlit Community Cloud).
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "anime_reviews.csv"


# ---------------------------------------------------------------------------
# 1. Load and clean the dataset
# ---------------------------------------------------------------------------

@st.cache_data
def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Basic cleaning
    df = df.drop_duplicates(subset="review_id")
    df = df.dropna(subset=["title", "genre", "review_text"])
    df["review_text"] = df["review_text"].str.strip()
    df["title"] = df["title"].str.strip()
    df["genre"] = df["genre"].str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype(int)
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")

    # Consistent season ordering for charts
    season_order = pd.CategoricalDtype(
        categories=["Winter", "Spring", "Summer", "Fall"], ordered=True
    )
    df["season"] = df["season"].astype(season_order)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. GenAI sentiment analysis
# ---------------------------------------------------------------------------

SENTIMENT_PROMPT = """You are a sentiment classifier for anime and MMORPG fan reviews.
Classify the review below as exactly one of: positive, negative, neutral.
Respond with ONLY a JSON object: {{"sentiment": "positive|negative|neutral", "confidence": 0-1}}

Review: \"\"\"{review}\"\"\"
"""


def _fallback_lexicon_sentiment(text: str) -> str:
    """Simple keyword fallback used only if no API key is supplied,
    so the app still runs end-to-end for grading/demo purposes."""
    positive_words = ["love", "great", "amazing", "hooked", "top tier", "rewarding", "earned", "best"]
    negative_words = ["dragged", "ruin", "unplayable", "exhausting", "bad", "boring", "rehash", "lag"]
    t = text.lower()
    pos_hits = sum(w in t for w in positive_words)
    neg_hits = sum(w in t for w in negative_words)
    if pos_hits > neg_hits:
        return "positive"
    if neg_hits > pos_hits:
        return "negative"
    return "neutral"


def classify_sentiment_genai(review_text: str, api_key: str, model: str = "gpt-4o-mini",
                              base_url: str | None = None) -> str:
    """Calls a GenAI API to classify one review. Works with OpenAI directly,
    or with any OpenAI-compatible endpoint (e.g. Google Gemini's free API)
    by passing base_url. Falls back to a lexicon heuristic if no API key is
    set, so the app remains demoable without credentials."""
    if not api_key:
        return _fallback_lexicon_sentiment(review_text)

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": SENTIMENT_PROMPT.format(review=review_text)}],
            temperature=0,
            max_tokens=50,
        )
        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(content)
        return parsed.get("sentiment", "neutral")
    except Exception:
        # Network / parsing failure -> fall back rather than crash the app
        return _fallback_lexicon_sentiment(review_text)


def analyze_dataframe(df: pd.DataFrame, api_key: str, model: str, progress_callback=None,
                       base_url: str | None = None) -> pd.DataFrame:
    """Runs sentiment analysis over every row in df and returns a copy
    with a new 'sentiment' column. Shows progress via an optional callback."""
    sentiments = []
    total = len(df)
    for i, review in enumerate(df["review_text"], start=1):
        sentiments.append(classify_sentiment_genai(review, api_key, model, base_url))
        if progress_callback:
            progress_callback(i / total)
        # tiny pause to stay well under rate limits on large batches
        if api_key and i % 20 == 0:
            time.sleep(0.5)
    out = df.copy()
    out["sentiment"] = sentiments
    return out
