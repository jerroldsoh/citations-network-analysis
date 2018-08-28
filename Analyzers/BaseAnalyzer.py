import pandas as pd
import numpy as np
import os

PATH_TO_CITES_REF = os.path.join(os.path.dirname(__file__), '..\\Data\\citations_to_names.csv')
PATH_TO_IDS_REF = os.path.join(os.path.dirname(__file__), '..\\Data\\ids_to_citations.csv')

class BaseAnalyzer():
    def __init__(self):
        self.citations_to_names = pd.read_csv(PATH_TO_CITES_REF,
        index_col = 0, header=None).to_dict()[1] # maps citations to case names
        self.ids_to_citations = pd.read_csv(PATH_TO_IDS_REF,
        index_col = 0, header=None).to_dict()[1] # maps ids to case citations

    def get_case_name_from_id_num(self, num, with_cite=False):
        cite = self.ids_to_citations[num]
        try:
            name = self.citations_to_names[cite]
            if with_cite:
                name = name + ' ' + cite
            return name
        except KeyError:
            print(num, 'not found in dataset')
            return None

    def get_citation_from_id(self, case_id):
        if pd.isnull(case_id):
            return np.nan
        return self.ids_to_citations[case_id]

    def get_meta_data_by_id(self, case_id, meta_df, cols):
        cite = self.ids_to_citations[case_id]
        case_row = meta_df[meta_df['reporter_citation'] == cite].iloc[0]
        case_row = case_row[cols]
        return case_row

    def convert_id_df_to_case_names(self, df, with_cite=False):
        return df.apply(np.vectorize(self.get_case_name_from_id_num), args=(with_cite,))

    # custom transpose into publishing format
    def format_for_pub(self, stat_df):
        if 'std' in stat_df.index:
            stat_df.loc['std'] = stat_df.loc['std'].apply(lambda x: '({})'.format(round(x, 3)))

        formatted = pd.DataFrame()
        for col in stat_df.columns:
            for stat in stat_df.index:
                formatted.loc[col+'_'+stat, 'Measure'] = stat_df.loc[stat, col]

        return formatted

    def readify_varnames(self, name):
        if name.endswith('_std'):
            return ''
        if name.endswith('_mean'):
            name = name.replace('_mean', '')
        if name.endswith('_size'):
            return 'Observations In Sample'.title()
        
        mapping = {
            'judgment_word_count': 'Word Count',
            'num_cited_cases': 'Outward citations',
            'num_citing_cases': 'Inward citations',
            'cites_per_000_word': 'Outward citations per \'000 words',
            'has_topic_': '',
            'in_degree': 'inward citations',
            'out_degree': 'outward citations',

        }
        for old, new in mapping.items():
            name = name.replace(old, new)
        name = name.replace('_', ' ')
        return name.title()

    def format_groupby_for_pub(self, df):
        for x_i in df.index:
            if x_i[-1] == 'std':
                df.loc[x_i] = df.loc[x_i].apply(lambda x: '('+str(np.round(x,4))+')')
        df = df.reset_index()
        df.index = df['level_0'] + '_' +df['level_1']
        df.index = pd.Series(df.index).apply(self.readify_varnames)
        df = df.drop(['level_0', 'level_1'], axis=1)
        df.index.name = None
        return df

    def tabulate_global_stats(self, df, cols_to_tabulate, stats_to_tabulate=['mean', 'std']):
        table = df[cols_to_tabulate].describe().loc[stats_to_tabulate]      
        return self.format_for_pub(table)

    # conditional analysis by topic
    # groupby alone doesnt work because we read a matrix of indicators
    def tabulate_topical_stats(self, df, cols_to_tabulate, topics_to_tabulate=[], topic_prefix='has_topic_', stats_to_tabulate=['mean', 'std']):
        table = pd.DataFrame()



        for col in df.columns:
            if col.startswith(topic_prefix) and col.strip() in topics_to_tabulate:
                topic_name = col.replace(topic_prefix, '')
                subset = df[df[col]==1]
                print(topic_name, subset.shape)
                stats = subset[cols_to_tabulate].describe().loc[stats_to_tabulate]

                for tgt_col in cols_to_tabulate:
                    for stat in stats_to_tabulate:
                        table.loc[stat, topic_name+'_'+tgt_col] = stats.loc[stat, tgt_col]

        # reshape the table so that each topic is a column and each statistic a row with SD below
        return table

    def tabulate_time_series(self, df, cols_to_tabulate, time_colname):
        function_map = {
            col: [np.mean, np.std] for col in cols_to_tabulate[:-1]
        }
        # special treatment for last one as I want the bottom row to have n_obs
        function_map.update({
            cols_to_tabulate[-1]: [np.mean, np.std, np.size]
        })
        table = df.groupby(time_colname).agg(function_map).transpose()
        return table

