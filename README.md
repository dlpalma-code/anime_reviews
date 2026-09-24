# Anime & MMORPG Review Explorer

A Streamlit app for **CS 315 – Application Development and Emerging Technologies, Activity 3**.

It loads a dataset of anime/MMORPG fan reviews (MyAnimeList/Anilist-style),
runs GenAI-powered sentiment analysis on the reviews, and visualizes which
genres/shows spike in popularity by season.

## Project structure

```
anime_review_analyzer/
├── data/
│   └── anime_reviews.csv     # sample dataset (300 synthetic reviews)
├── app.py                    # Streamlit UI
├── utils.py                  # data loading/cleaning + GenAI sentiment logic
├── generate_data.py          # regenerates the sample dataset
├── requirements.txt
└── README.md
```

## 1–3. Dataset & cleaning

`data/anime_reviews.csv` has columns: `review_id, title, genre, season, year,
reviewer, rating, review_text, review_date`. `utils.load_data()` deduplicates,
drops missing values, coerces types, and orders seasons for charting.

To swap in a **real** MyAnimeList or Anilist export, replace
`data/anime_reviews.csv` with a CSV that has the same column names (or adjust
the column names in `utils.py`).

To regenerate the synthetic sample data:

```bash
python generate_data.py
```

## 4. GenAI integration

`utils.classify_sentiment_genai()` classifies each review as
positive/negative/neutral. The sidebar lets you pick a **provider**:

- **Google Gemini (free tier)** — recommended for this assignment. Get a
  free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
  (no credit card needed). It uses Gemini's OpenAI-compatible endpoint, so
  the app talks to it with the same `openai` Python library, just pointed at
  a different `base_url`.
- **OpenAI (paid)** — pay-as-you-go, billed per token.

If no API key is entered for either provider, the app automatically falls
back to a small keyword-based classifier so it still runs for demos/grading
without any credentials at all.

## 5–6. Streamlit interface & visualization

- Sidebar: API key input, model choice, and filters (genre, season, year, rating).
- **Data** tab: filtered review table.
- **Sentiment Analysis** tab: run classification, see a sentiment pie chart
  and average-rating-by-sentiment bar chart (Plotly).
- **Seasonal Trends** tab: genre popularity by season (grouped bar chart) and
  average rating by genre across years (line chart).

## 7. Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Test with different filter combinations and, if you have an OpenAI key, compare
the GenAI classifications against the fallback classifier's output.

## 8. Deploy to Streamlit Community Cloud

1. Push this folder to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in, and click
   **New app**.
3. Point it at your repo, branch, and `app.py`.
4. (Optional) Add your API key as a secret instead of typing it in the
   sidebar each time — in `Settings → Secrets`:
   ```toml
   GEMINI_API_KEY = "..."
   OPENAI_API_KEY = "sk-..."
   ```
   The sidebar already reads whichever key matches the selected provider via
   `st.secrets`, so it'll be pre-filled automatically once deployed.

## 9. Next goals (not yet implemented)

- Add dedicated filter widgets for individual titles, not just genre.
- Add a chatbot tab (GenAI-powered) that answers questions about the dataset,
  e.g. "Which genre had the most negative reviews in Winter 2024?"
