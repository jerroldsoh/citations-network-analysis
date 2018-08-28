from .BaseTransformer import BaseTransformer
import re
import string
import os
import pandas as pd
import ast

class MetaDataTransformer(BaseTransformer):

    def __init__(self):
        self.required_keys = ('decision_date', 'clean_catch_words')

    def get_list_of_first_items(self, df_cell):
        return list(set([x[0] for x in df_cell]))

    def get_topic_dummies(self, clean_catch_words_series):
        topics_appearing = clean_catch_words_series.apply(self.get_list_of_first_items)
        dummies = pd.get_dummies(topics_appearing.apply(pd.Series).stack(), prefix='has_topic').sum(level=0)
        return dummies

    def find_bench(self, date):
        yong_appt_date = pd.to_datetime('1990/09/28') # https://en.wikipedia.org/wiki/Yong_Pung_How
        chan_appt_date = pd.to_datetime('2006/04/11') # http://www.pmo.gov.sg/newsroom/re-appointment-chan-sek-keong-chief-justice
        menon_appt_date = pd.to_datetime('2012/11/6') # https://www.supremecourt.gov.sg/news/media-releases/retirement-of-chief-justice-chan-sek-keong-and-appointment-of-justice-sundaresh-menon-as-chief-justice
        
        if date < yong_appt_date:
            return 'wee'
        if date < chan_appt_date:
            return 'yong'
        if date < menon_appt_date:
            return 'chan'
        return 'menon'

    def transform(self, df):
        new_cols = pd.DataFrame()
        new_cols['decision_date'] = pd.to_datetime(df['decision_date'])
        new_cols['decision_year'] = new_cols['decision_date'].apply(lambda x: x.year)
        new_cols['decision_month'] = new_cols['decision_date'].apply(lambda x: x.month)
        new_cols['clean_catch_words'] = df['clean_catch_words'].str.lower().apply(ast.literal_eval)
        topic_dummies = self.get_topic_dummies(new_cols['clean_catch_words'])
        old_rows = new_cols.shape[0]
        new_cols = new_cols.join(topic_dummies.set_index(new_cols.index))
        assert old_rows == new_cols.shape[0]
        new_cols['num_topics'] = new_cols[[col for col in new_cols.columns if col.startswith('has_topic')]].sum(axis=1)

        new_cols['bench'] = new_cols['decision_date'].apply(self.find_bench)
        return new_cols

    def test(self):
        first_items = self.get_list_of_first_items([['abc','b'],['def','d']])
        assert 'abc' in first_items and 'def' in first_items

        test_df = pd.DataFrame()
        test_df['decision_date'] = ['20-11-2007','20-11-2011']
        test_df['clean_catch_words'] = [
            str([['tort','negligence'],['contract','breach']]),
            str([['tort','negligence'],['crime','breach']]),
        ]
        test_df['reporter_citation'] = ['[2007] 1 SLR 123', '[2011] 1 SLR 123']
        transformed_test_df = self.fit_transform(test_df)
        assert transformed_test_df.loc['[2007] 1 SLR 123', 'has_topic_contract']
        assert transformed_test_df.loc['[2011] 1 SLR 123', 'has_topic_crime']
        assert transformed_test_df.loc['[2011] 1 SLR 123', 'bench'] == 'chan'

        test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../Data/test_base_data.csv'))
        transformed_test_df = self.fit_transform(test_df)
        assert transformed_test_df.loc['[2003] 2 SLR(R) 33', 'has_topic_tort']
        assert transformed_test_df.loc['[2003] 2 SLR(R) 33', 'bench'] == 'yong'

if __name__ == '__main__':
    mdt = MetaDataTransformer()
    mdt.test()
