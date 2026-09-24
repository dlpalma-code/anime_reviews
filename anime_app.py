"""
app.py
Anime/MMORPG Review Sentiment & Trend Explorer
A Streamlit app that loads fan review data, runs GenAI-powered sentiment
analysis, and visualizes seasonal genre/show popularity trends.
"""

import streamlit as st
import plotly.express as px

from utils import load_data, analyze_dataframe

st.set_page_config(
    page_title="Anime & MMORPG Review Explorer",
    page_icon="🎮",
    layout="wide",
)

st.title("🎮 Anime & MMORPG Review Explorer")
st.caption(
    "Sentiment analysis and seasonal trend visualization for fan reviews "
    "(sample dataset in the style of MyAnimeList / Anilist exports)."
)

# ---------------------------------------------------------------------------
# Sidebar: GenAI settings + filters
# ---------------------------------------------------------------------------

PROVIDERS = {
    "Google Gemini (free tier)": {
        "models": ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.5-pro"],
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "secret_key": "GEMINI_API_KEY",
        "key_label": "Gemini API key (optional)",
        "help": "Free, no credit card needed — get one at aistudio.google.com/apikey. "
                "Leave blank to use the built-in keyword fallback instead.",
    },
    "OpenAI (paid)": {
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        "base_url": None,
        "secret_key": "OPENAI_API_KEY",
        "key_label": "OpenAI API key (optional)",
        "help": "Leave blank to use the built-in keyword fallback instead.",
    },
}

with st.sidebar:
    st.header("⚙️ GenAI Settings")
    provider_name = st.selectbox("Provider", list(PROVIDERS.keys()), index=0)
    provider = PROVIDERS[provider_name]

    api_key = st.text_input(
        provider["key_label"],
        value=st.secrets.get(provider["secret_key"], ""),
        type="password",
        help=provider["help"],
    )
    model = st.selectbox("Model", provider["models"], index=0)
    base_url = provider["base_url"]

    st.divider()
    st.header("🔍 Filters")

df = load_data()

with st.sidebar:
    genres = st.multiselect("Genre", sorted(df["genre"].unique()), default=list(df["genre"].unique()))
    seasons = st.multiselect("Season", ["Winter", "Spring", "Summer", "Fall"],
                              default=["Winter", "Spring", "Summer", "Fall"])
    years = st.multiselect("Year", sorted(df["year"].unique()), default=list(df["year"].unique()))
    min_rating, max_rating = st.slider("Rating range", 1, 10, (1, 10))

filtered = df[
    df["genre"].isin(genres)
    & df["season"].isin(seasons)
    & df["year"].isin(years)
    & df["rating"].between(min_rating, max_rating)
]

st.markdown(f"**{len(filtered)}** reviews match your filters (of {len(df)} total).")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_data, tab_sentiment, tab_trends = st.tabs(
    ["📄 Data", "💬 Sentiment Analysis", "📈 Seasonal Trends"]
)

with tab_data:
    st.subheader("Filtered reviews")
    st.dataframe(
        filtered[["title", "genre", "season", "year", "reviewer", "rating", "review_text"]],
        use_container_width=True,
        height=400,
    )

with tab_sentiment:
    st.subheader("Run GenAI sentiment analysis")
    st.write(
        "Classifies each filtered review as positive, negative, or neutral. "
        "Without an API key, a lightweight keyword fallback is used so the "
        "demo still works."
    )

    if "analyzed" not in st.session_state:
        st.session_state.analyzed = None

    if st.button("Analyze sentiment", type="primary", disabled=filtered.empty):
        progress_bar = st.progress(0.0, text="Analyzing reviews...")

        def update(pct):
            progress_bar.progress(pct, text=f"Analyzing reviews... {int(pct * 100)}%")

        st.session_state.analyzed = analyze_dataframe(filtered, api_key, model, update, base_url)
        progress_bar.empty()

    result = st.session_state.analyzed
    if result is not None:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Sentiment breakdown**")
            counts = result["sentiment"].value_counts().reindex(
                ["positive", "neutral", "negative"]
            ).fillna(0)
            fig = px.pie(
                values=counts.values,
                names=counts.index,
                color=counts.index,
                color_discrete_map={"positive": "#4CAF50", "neutral": "#FFC107", "negative": "#F44336"},
                hole=0.4,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Average rating by sentiment**")
            avg_rating = result.groupby("sentiment", observed=True)["rating"].mean().reindex(
                ["positive", "neutral", "negative"]
            )
            fig2 = px.bar(
                x=avg_rating.index, y=avg_rating.values,
                labels={"x": "Sentiment", "y": "Avg. rating"},
                color=avg_rating.index,
                color_discrete_map={"positive": "#4CAF50", "neutral": "#FFC107", "negative": "#F44336"},
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Reviews with sentiment labels**")
        st.dataframe(
            result[["title", "genre", "rating", "sentiment", "review_text"]],
            use_container_width=True,
            height=350,
        )
    else:
        st.info("Click **Analyze sentiment** to classify the filtered reviews.")

with tab_trends:
    st.subheader("Which genres spike by season?")

    trend = (
        filtered.groupby(["season", "genre"], observed=True)
        .size()
        .reset_index(name="review_count")
    )
    fig3 = px.bar(
        trend, x="season", y="review_count", color="genre",
        barmode="group",
        category_orders={"season": ["Winter", "Spring", "Summer", "Fall"]},
        labels={"review_count": "Number of reviews", "season": "Season"},
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Average rating by genre across years")
    rating_trend = (
        filtered.groupby(["year", "genre"], observed=True)["rating"]
        .mean()
        .reset_index()
    )
    fig4 = px.line(
        rating_trend, x="year", y="rating", color="genre", markers=True,
        labels={"rating": "Avg. rating", "year": "Year"},
    )
    st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.caption(
    "Built for CS 315 – Application Development and Emerging Technologies, Activity 3. "
    "Dataset is synthetically generated for demonstration; swap in a real "
    "MyAnimeList/Anilist export by replacing data/anime_reviews.csv with the same columns."
)
