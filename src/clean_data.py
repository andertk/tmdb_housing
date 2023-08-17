import pandas as pd


RAW_MOVIES_PATH = "data/raw/top_1000_popular_movies_tmdb.csv"


def clean_data():
    clean_data = (
        pd.read_csv(RAW_MOVIES_PATH, engine="python", index_col=0)
        .astype({"id": "object"})
        .assign(release_date=lambda x: pd.to_datetime(x["release_date"], errors="coerce"))
    )
    return clean_data


if __name__ == "__main__":
    clean_data = clean_data()
    