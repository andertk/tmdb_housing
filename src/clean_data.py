import pandas as pd


def clean_data(raw_path, clean_path):
    def get_dummies_from_lists(x):
        return (
            x
            .apply(eval)
            .explode()
            .pipe(pd.get_dummies)
            .groupby('id')
            .sum()
            .clip(upper=1)
        )
    df = (
        pd.read_csv(raw_path, engine='python', index_col=0)
        .astype({'id': 'object'})
        .assign(release_date=lambda x: pd.to_datetime(x['release_date'], errors='coerce'))
        .query('vote_count > 0')
        .query('budget > 0')
        .query('revenue > 0')
        .set_index('id')
    )

    genres = get_dummies_from_lists(df['genres'])
    production_companies = get_dummies_from_lists(df['production_companies'])

    df = pd.concat([df, genres, production_companies], axis=1)
    df.to_csv(clean_path)
