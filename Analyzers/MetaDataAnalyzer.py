import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
from .BaseAnalyzer import BaseAnalyzer

class MetaDataAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()

    ### Methods to analyse how the number of topics in a case affects covariates like word counts
    def get_perc_with_sole_topic(self, df, topic):
        sole_topic = df[(df['num_topics']==1) & (df[topic] ==1)].shape[0]
        all_with_topic = df[df[topic] ==1].shape[0]
        return sole_topic / all_with_topic

    def get_stats_by_topic_num(self, df, topic_name, topic_num, topic_prefix='has_topic_'):
        topic_name = topic_prefix+topic_name
        return df[(df[topic_name]==1) & (df['num_topics']==topic_num)][cols_to_tabulate].describe()

    def get_meta_data_by_id(case_id, df):
        cite = self.ids_to_citations[case_id]
        case_row = df[df['reporter_citation'] == cite].iloc[0]
        return case_row

    def plot_yearmonth_map(self, ax, df, col, aggfunc=np.mean, cmap = 'gist_heat_r', with_rowmeans = False, with_colmeans = False, **kwargs):
        ### Visualise aspects of the metadata. Can provide any one column and custom aggfuncs

        # prep 2d matrix that fits sns requirements
        mtx = pd.pivot_table(df, index=['decision_year', 'decision_month'], values=col, aggfunc=aggfunc).unstack()
        mtx.columns = [calendar.month_abbr[i] for i in range(1,13)]
        
        if with_colmeans:
            colmeans = mtx.mean(axis=0)
            colmeans.name = 'Month Avg'
            colmeans.index = mtx.columns
            mtx = mtx.append(colmeans)
            
        if with_rowmeans:
            rowmeans = mtx.mean(axis=1)
            rowmeans.name = 'Year\nAvg'
            rowmeans.index = mtx.index
            mtx['Year\nAvg'] = rowmeans
        
        # if we call both, the bottom right value becomes an avg of the month avgs 
        # which complicats the analysis, so remove
        if with_colmeans and with_rowmeans:
            mtx.iloc[-1,-1] = np.nan
        
        hmap = sns.heatmap(mtx, ax=ax, cmap=cmap, linewidths=1, linecolor='w', **kwargs)
        hmap.set_yticklabels(hmap.get_yticklabels(), rotation = 0)
        hmap.set_ylabel('Year Decided')
        hmap.set_xlabel('Month Decided')
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.tick_params(top=False, left=False)
        return hmap, ax
