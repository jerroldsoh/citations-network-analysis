import pandas as pd
import numpy as np
from .BaseTransformer import BaseTransformer

class CiteTransformer(BaseTransformer):

    def __init__(self):
        self.required_keys = ('ls_cited_cases', 'judgment_word_count', 'decision_date')

    def get_ls_cited_ages_per_row(self, df_row, date_series):
        citing_date = pd.to_datetime(df_row['decision_date'])
        ls_cited_ages = []
        for cited in df_row['ls_cited_cases']:
            try:
                cited_date = pd.to_datetime(date_series[cited])
                delta_days = round((citing_date - cited_date).days, 4) # using days is most precise because of leap years etc
                ls_cited_ages.append(delta_days)
            except KeyError:
                ls_cited_ages.append(np.nan)
        return ls_cited_ages

    def transform(self, df):
        new_cols = pd.DataFrame()
        new_cols['num_cited_cases'] = df['ls_cited_cases'].apply(len)
        new_cols['cites_per_000_word'] = new_cols['num_cited_cases'] / (df['judgment_word_count'] / 1000)

        # get cite ages - 
        new_cols['ls_cited_ages'] = df.apply(
            self.get_ls_cited_ages_per_row, axis=1, args=(df['decision_date'],)
        )

        new_cols['mean_cited_ages'] = new_cols['ls_cited_ages'].apply(
            lambda x: round(np.nanmean(x), 4) if list(filter(pd.notnull, x)) else None
        )
        return new_cols
        