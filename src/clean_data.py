import pandas as pd


RAW_MOVIES_PATH = "data/raw/top_1000_popular_movies_tmdb.csv"


def clean_data():
    df = (
        pd.read_csv(RAW_MOVIES_PATH, engine="python", index_col=0)
        .astype({"id": "object"})
        .assign(
            release_date=lambda x: pd.to_datetime(x["release_date"], errors="coerce"),
            genres=lambda x: x.genres.apply(eval),
            production_companies=lambda x: x.production_companies.apply(eval)
        )
        .query("vote_count > 0")
        .query("budget > 0")
        .query("revenue > 0")
        .set_index("id")
    )

    return pd.concat(
        [df, pd.get_dummies(df.genres.explode()).groupby(level=0).sum()], 
        axis=1
    )


if __name__ == "__main__":
    clean_data = clean_data()
    