from Transformers.MetaDataTransformer import MetaDataTransformer
from Transformers.CiteExtractor import CiteExtractor
from .BaseTransformer import BaseTransformer
import re
import string
import os
import pandas as pd
import ast
import numpy as np

class EdgeDataTransformer(BaseTransformer):

    def __init__(self):
        self.required_keys = ('ls_cited_cases',)

    def pair_up(self, df_row):
        this_case = df_row.name
        cited_cases = df_row['ls_cited_cases']
        # must return in string otherwise pandas.apply tries to construct it into some data frame and raises ValueError
        return str([(this_case, cited_case) for cited_case in cited_cases])

    def make_edges(self, df):
        str_edges = df.apply(self.pair_up, axis=1)
        ls_edges = str_edges.apply(ast.literal_eval)
        flat_edges = [edge for edge_group in ls_edges.ravel() for edge in edge_group]
        edge_df = pd.DataFrame(flat_edges).rename({0: 'citing', 1: 'cited'}, axis=1)
        return edge_df

    def get_isolates(self, edge_df, all_nodes):
        non_isolates = set(edge_df['citing'].unique()) | set(edge_df['cited'].unique())
        isolates = list(filter(
            lambda x: x not in non_isolates, all_nodes
        ))

        # use a single col df so that we can easily append back to edge_df
        out = pd.DataFrame()
        out['citing'] = isolates
        return out

    def get_citation_info(self, citation, df, colname):
        if pd.isnull(citation):
            return np.nan
        return df.loc[df.index == citation, colname].iloc[0]

    def transform(self, df):
        edge_df = self.make_edges(df)
        orig_cases = df.index.unique() # expects reporter_citation to be the index
        
        # mask out cases cited but not in original df
        edge_df = edge_df[edge_df['cited'].apply(lambda x: x in orig_cases)]
        
        # append topics of cited case. must be done before isolates introduce NAs to cited
        old_rows = edge_df.shape[0]
        cited_prefix = 'cited_'
        edge_df = pd.merge(edge_df,
            df[[col for col in df.columns if col.startswith('has_topic')]],
            left_on = 'cited',
            right_index = True,
        )
        edge_df = edge_df.rename({col: cited_prefix+col
            for col in edge_df.columns if col.startswith('has_topic')}, axis=1)
        assert edge_df.shape[0] == old_rows
       
        # append isolates to the df as edges with NAs for cited
        old_rows = edge_df.shape[0]
        isolates = self.get_isolates(edge_df, orig_cases)
        edge_df = edge_df.append(isolates)
        assert edge_df.shape[0] == old_rows + isolates.shape[0]

        # append topics of citing case
        old_rows = edge_df.shape[0]
        edge_df = pd.merge(edge_df,
            df[[col for col in df.columns if col.startswith('has_topic')]],
            left_on = 'citing',
            right_index = True,
        )
        citing_prefix = 'citing_'
        edge_df = edge_df.rename({col: citing_prefix+col
            for col in edge_df.columns if col.startswith('has_topic')}, axis=1)
        assert edge_df.shape[0] == old_rows

        # find date corresponding to citation
        edge_df['citation_birth_date'] = pd.to_datetime(edge_df['citing'].apply(
            self.get_citation_info, args = (df, 'decision_date')))
        
        # cited_cases_date = pd.to_datetime(edge_df['cited'].apply(
        #     self.get_citation_info, args = (df, 'decision_date')))
        
        # edge_df['citation_age'] = ((edge_df['citation_birth_date'] 
        #     - cited_cases_date).apply(lambda x: x.days)
        #     #/ np.timedelta64(1, 'Y')
        #     ) # convert to decimal years

        id_dict = dict(zip(orig_cases, range(1, len(orig_cases)+1)))
        reverse_dict = dict(zip(range(1, len(orig_cases)+1), orig_cases))

        # Replace each case cite with that number
        edge_df[['citing_id', 'cited_id']] = edge_df[['citing', 'cited']].apply(lambda x: x.replace(id_dict), axis = 0)

        return edge_df, id_dict, reverse_dict

if __name__ == '__main__':
    test_df = pd.read_csv(os.path.join(os.path.dirname(__file__), '../Data/test_base_data.csv'))
    mdt = MetaDataTransformer.MetaDataTransformer()
    transformed_test_df = mdt.fit_transform(test_df)

    ce = CiteExtractor.CiteExtractor()
    transformed_test_df['ls_cited_cases'] = ce.fit_transform(transformed_test_df)

    edt = EdgeDataTransformer()
    edge_df = edt.fit_transform(transformed_test_df.reset_index())
