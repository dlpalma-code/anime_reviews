"""
generate_data.py
(Re)generates anime_reviews.csv using the same synthetic-dataset logic
utils.py uses as its in-app fallback.

Run:
    python generate_data.py
"""

from utils import generate_dataset, BASE_DIR

if __name__ == "__main__":
    df = generate_dataset()
    out_path = BASE_DIR / "anime_reviews.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
 
