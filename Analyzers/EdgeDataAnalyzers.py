'''
Takes an edge dataframe and runs all graph analysis
'''
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.ticker import MaxNLocator
import imageio
import math
from .BaseAnalyzer import BaseAnalyzer

class EdgeDataAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()

    def find_cases_citing(self, case_id, edge_df, case_title=True):
        subset = edge_df.loc[edge_df['cited_id']==case_id, 'citing_id']
        cases_citing = subset.values.tolist()
        if case_title:
            cases_citing = [self.get_case_name_from_id_num(citing, True) for citing in cases_citing]
        return cases_citing
    
class CentralityAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
        self.measures = [
            'in_degree_cent',
            'out_degree',
            'out_degree_cent',
            'pagerank',
            'hub_score',
            'authority_score',
    #         'eigen_cent',
    #         'betweenness_cent',
        ]

    def get_measures(self):
        return self.measures

    def safe_get_hits(self, dg):
        try:
            hits = nx.algorithms.link_analysis.hits(dg, max_iter=300)
        except nx.PowerIterationFailedConvergence:
            print('HITS failed to converge, trying with lower tolerance')
            hits = nx.algorithms.link_analysis.hits(dg, max_iter=1000, tol=1e-6)
        return hits

    def get_centrality(self, dg, measure):
        if measure == 'out_degree':
            return dict(dg.out_degree())
        if measure == 'out_degree_cent':
            return nx.algorithms.centrality.out_degree_centrality(dg)
        if measure == 'in_degree':
            return dict(dg.in_degree())
        if measure == 'in_degree_cent':
            return nx.algorithms.centrality.in_degree_centrality(dg)
        if measure == 'pagerank':
            return nx.algorithms.link_analysis.pagerank(dg)
        if measure == 'hub_score':
            hits = self.safe_get_hits(dg)
            return hits[0]
        if measure == 'authority_score':
            hits = self.safe_get_hits(dg)
            return hits[1]
        if measure == 'hits':
            hits = self.safe_get_hits(dg)
            return hits
        if measure == 'eigen_cent':
            return nx.algorithms.centrality.eigenvector_centrality(dg)
        if measure == 'betweenness_cent':
            return nx.algorithms.centrality.betweenness_centrality(dg)
        return None

    def get_centrality_df(self, dg, measures_to_include=[]):
        # seed one to prepare the index
        indeg = self.get_centrality(dg, 'in_degree')
        central_df = pd.DataFrame.from_dict(indeg, orient='index').rename({0:'in_degree'}, axis=1)  
        if not measures_to_include:
            measures_to_include = self.measures

        for measure in measures_to_include:
            central_df = central_df.join(pd.Series(self.get_centrality(dg, measure), name=measure))
        central_df.index = pd.Series(num for num in central_df.index.values)
        return central_df[['in_degree']+measures_to_include] # re-order to intuitive order

    def get_rank_scores(self, cent_df, meta_df, sort_by):
        cent_df['judgment_word_count'] = pd.Series(cent_df.index).apply(self.get_meta_data_by_id, args=(meta_df,'judgment_word_count')).values
        cent_df['judgment_word_count'] = cent_df['judgment_word_count'].astype(int)
        cent_df['reporter_citation'] = pd.Series(cent_df.index).replace(self.ids_to_citations)

        scores = cent_df.sort_values(sort_by, ascending=False).drop(['reporter_citation', 'in_degree_cent', 'out_degree_cent', 'case_title']
            +[col for col in cent_df.columns if col.startswith('has_topic')], axis=1, errors='ignore') # drop only if it exists
        rank = scores.rank(ascending=False, method='min')
        
        rsuffix='_Prank'
        rank_scores = scores.join(rank.astype(int), rsuffix=rsuffix)
        for col in rank_scores:
            if not col.endswith(rsuffix):
                rank_scores[col] = np.round(rank_scores[col], 4).astype(str) + ' (' + rank_scores[col+rsuffix].astype(str) + ')'
        rank_scores = rank_scores[[col for col in rank_scores if not col.endswith(rsuffix)]]
        rank_scores['case_title'] = pd.Series(rank_scores.index).apply(self.get_case_name_from_id_num, args=(True,)).tolist()
        return rank_scores

    # Methods for partitioning and calculating scores by period
    def get_relevant_subset(self, edge_df, topic_name):
        if not topic_name:
            return edge_df.copy()
        else:
            return edge_df[edge_df['citing_has_topic_'+topic_name]==1].copy()

    def get_key_cases_ts(self, edge_df, rank_scores, key_cases, topic_name=''):
        relevant_subset = get_relevant_subset(edge_df, topic_name)
        case_time_series = {}
        for case in key_cases:
            case_name = eda.get_case_name_from_id_num(case)
            print('Getting scores for case {}'.format(case_name))
            start, end = '20000101', '20171231'
            case_measures_over_time = ca.get_all_case_measures_over_time(
                relevant_subset, case, '20000101', '20171231', 
                measures_to_include = [
                    'in_degree_cent',
                    'hub_score',
                    'authority_score',
                ]
            )
            case_time_series[case] = case_measures_over_time
        return case_time_series
        
    def get_topic_rank_scores(edge_df, meta_df, sort_by, topic_name=''):
        relevant_subset = self.get_relevant_subset(edge_df, topic_name)
        gv = GraphVisualizer()
        topic_dg = gv.prepare_graph(relevant_subset, 'citing_id', 'cited_id', create_using=nx.DiGraph())
        topic_central_df = self.get_centrality_df(topic_dg)
        rank_scores = self.get_rank_scores(topic_central_df, meta_df, sort_by)
        return rank_scores

    def format_rank_scores(self, rank_scores):
        rank_scores = rank_scores.set_index('case_title')
        rank_scores.index.name = None
        rank_scores.columns = pd.Series(rank_scores.columns).apply(self.readify_varnames)
        return rank_scores

    def get_top_k_per_measure(self, centrality_df, k=10, cols_to_exclude=['in_degree', 'out_degree']):        
        central_cases = pd.DataFrame()
        measures = centrality_df.columns.difference(cols_to_exclude)
        for measure in measures:
            top_k = centrality_df.sort_values(measure, ascending=False).head(k)
            central_cases[measure] = top_k.index
        return central_cases

    # get most important cases by time period
    def get_top_k_over_time(self, edge_df, rank_by, start_date, end_date, interval='M', k=5):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        date_range = pd.date_range(start=start_date, end=end_date, freq=interval)
        all_years_df = pd.DataFrame()

        for period in date_range:
            subset = edge_df[pd.to_datetime(edge_df['citation_birth_date']) <= period]
            dg = nx.from_pandas_edgelist(subset, 'citing_id', 'cited_id', create_using=nx.DiGraph())
            centrality = self.get_centrality_df(dg, measures_to_include=[rank_by])
            top_k = self.get_top_k_per_measure(centrality, k)
            all_years_df[period] = top_k

        return all_years_df

    def plot_central_cases_hist(self, central_cases, **kwargs):
        ax = pd.Series(central_cases.stack().value_counts(ascending=True)).plot(kind='barh', **kwargs)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        return ax

    # focus on one case over time
    def get_case_scores_over_time(self, df,
        case_id, start_date, end_date, 
        interval='M', measure_type='in_degree', score_type = 'raw'):

        # robust to string inputs - will NOT throw if already datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        df.loc[:,'citation_birth_date'] = pd.to_datetime(df['citation_birth_date'])
        
        # naive algo - for each period, subset, make graph, and find score
        # if anyone can think of a faster way that'll be great
        date_range = pd.date_range(start=start_date, end=end_date, freq=interval)
        case_scores = pd.Series(index=date_range)
        for period in date_range:
            subset = df[df['citation_birth_date'] <= period]

            # if case hasn't been decided or citing in this period, no score
            if not (case_id in subset['citing_id'].unique()
                    or case_id in subset['cited_id'].unique()):
                score_this_period = np.nan
            else:
                dg = nx.from_pandas_edgelist(subset,
                    'citing_id', 'cited_id', create_using=nx.DiGraph())
                global_scores = pd.Series(self.get_centrality(dg, measure_type))
                # need to convert to percentile rank for everything except in and out degree
                # else not comparable across time as number of cases change
                if score_type == 'perc' and measure_type != 'in_degree':
                    global_scores = global_scores.rank(pct=True)

                score_this_period = global_scores.loc[case_id]
            
            case_scores.loc[period] = score_this_period

        return case_scores

    def get_scores_for_cases_citing_case(self, case_id, score, edge_df, cent_df):
        citing_case_indices = self.find_cases_citing(case_id, edge_df, False)
        return cent_df.iloc[citing_case_indices][score]

    def get_all_case_measures_over_time(self, df,
        case_id, start_date, end_date, 
        interval='M', measures_to_include=None):
        if not measures_to_include:
            measures_to_include = self.measures
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        all_measures_df = pd.DataFrame()

        for measure in measures_to_include:
            print('Getting {} over time...'.format(measure))
            case_scores = self.get_case_scores_over_time(df,
                case_id, start_date, end_date, 
                interval, measure)
            all_measures_df[measure] = case_scores
        return all_measures_df

    def get_case_time_series(self, cases_to_track):
        for case in cases_to_track:
        case_name = self.get_case_name_from_id_num(case)
        print('Getting scores for case {}'.format(case_name))
        start, end = '20000101', '20171231'
        case_measures_over_time = ca.get_all_case_measures_over_time(
            topic_edges, case, start, end, 
            measures_to_include = [
                'in_degree_cent',
                'hub_score',
                'authority_score',
            ]
        )
        case_time_series[case] = case_measures_over_time
        return case_time_series

    def plot_case_measures_over_time(self, all_case_measures,
        start_date='20000101', 
        end_date='20171231'):
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        linestyles = [':','-.','--', '-']
        fig, ax = plt.subplots()
        ax.set_xlim((start_date, end_date))

        # WILL THROW IF N_MEASURES != 4
        for measure, style in zip(all_case_measures, linestyles):
            ax.plot(all_case_measures[measure], label=measure, linestyle = style)

        plt.legend() 
        return fig, ax

class ClusterAnalyzer():
    pass

class GraphVisualizer(BaseAnalyzer):
    def __init__(self):
        super().__init__()
    
    def prepare_plot_name(self, node):
        return self.get_case_name_from_id_num(node)[0:12]#+'...'+ids_to_citations[node]

    def get_full_names(self, labels):
        full_names = {}
        for node, label in labels.items():
            if label:
                full_names[node] = self.get_case_name_from_id_num(node) + ' ' +self.ids_to_citations[node]
        return full_names

    def prep_specific_labels(self, nodes, centrality, centrality_thresh, ls_specific_nodes=[]):
        labels = {node: self.prepare_plot_name(node) if centrality[node] > centrality_thresh 
                  else '' for node in nodes.keys()}
        labels.update({
            node: self.prepare_plot_name(node) for node in ls_specific_nodes
        })
        return labels

    def prep_top_labels(self, nodes, centrality, thresh=None):
        if not thresh:
            thresh = np.max(list(centrality.values()))

        labels = {node: self.prepare_plot_name(node) if centrality[node] >= thresh
                  else '' for node in nodes.keys()}
        return labels

    def prepare_graph(self, df, from_colname, to_colname, create_using):
        # some nodes link nowhere as they are isolated
        # using sent_val as I cant figure out how to target 'nan' as a key
        graph_df = df.copy() # to avoid filling na on df slice
        sentinel_val = -999
        graph_df[to_colname].fillna(sentinel_val, inplace=True)
        g_to_return = nx.from_pandas_edgelist(graph_df, from_colname, to_colname, create_using=create_using)
        # nx will read 'nan' as nodes too. remove by detecting a sent_val 
        # to isolate the isolates!
        if sentinel_val in g_to_return.nodes():
            g_to_return.remove_node(sentinel_val)
       
        return g_to_return

    def subset_dict(self, the_dict, ls_keys):
        return {k: the_dict[k] for k in ls_keys}

    # choose only the most important nodes to plot to avoid hairballs
    # note this WILL create new isolates when a node has all its related nodes removed
    def subgraph_for_plotting(self, g, importance, importance_thresh):
        to_keep = [node for node in importance if importance[node] >= importance_thresh]
        subg = g.subgraph(to_keep)
        sub_importance = self.subset_dict(importance, to_keep)
        return subg, sub_importance

    def prepare_plot(self, g, importance_type = None,
        importance_thresh = None):

        if importance_type == 'pagerank':
            importance = nx.link_analysis.pagerank(g)
        elif importance_type == 'authority':
            hits = nx.algorithms.link_analysis.hits(g, max_iter=200)    
            importance = hits[1]
        elif importance_type == 'hub':
            hits = nx.algorithms.link_analysis.hits(g, max_iter=200)    
            importance = hits[0]
        else:
            if type(g) is nx.DiGraph or type(g) is nx.classes.graphviews.SubDiGraph:
                importance = dict(g.in_degree())
            else:
                importance = dict(g.degree())

        if importance_thresh:
            g, importance = self.subgraph_for_plotting(g, importance, importance_thresh)

        return g, importance

    def plot_graph(self, ax, g,
        importance,
        size_multiplier = 50,       
        with_labels=False, label_thresh=None,
        with_box=False,
        node_kwargs = {'alpha': 0.85},
        edge_kwargs = {'edge_color':'k', 'width':0.5, 'alpha': 0.6,}):

        pos = nx.drawing.layout.spring_layout(g, random_state=36)
        nodes = g.nodes()
        node_size = np.asarray([importance[n] for n in nodes]) * size_multiplier + 50
        cmap = cm.Blues(np.linspace(0,1,20))
        cmap = colors.ListedColormap(cmap[5:,:-1])
        nx.draw_networkx_nodes(g, node_size = node_size, pos=pos, ax=ax, node_color = node_size,
                               cmap=cmap, **node_kwargs)
        nx.draw_networkx_edges(g, pos=pos, ax=ax, **edge_kwargs)
        if with_labels:
            labels = self.prep_top_labels(nodes, importance, label_thresh)
            nx.draw_networkx_labels(g, pos=pos, ax=ax, labels=labels, font_size=11, font_color='orangered')
        if not with_box:
            plt.axis('off')
        else:
            plt.xticks([])
            plt.yticks([])
        return ax

    def plot_empty(self, ax):
        ax.text(0.33, 0.5, 'No cases this year')
        plt.xticks([])
        plt.yticks([])
        
    def plot_yearly_graph(self, df, start_year, end_year, label_min_year = None, cumulative=False, num_cols = 3, node_kwargs = {}, edge_kwargs = {}):
        df['citation_birth_date'] = pd.to_datetime(df['citation_birth_date'])

        num_years = end_year - start_year
        num_rows = math.ceil(num_years / num_cols)

        if not label_min_year:
            label_min_year = start_year
        width = 4
        fig = plt.figure(figsize=(width*num_cols, width*num_rows))
        for i, year in enumerate(range(start_year, end_year)):
            ax = plt.subplot(num_rows, num_cols, i+1)
            if cumulative:
                mask = df['citation_birth_date'].apply(lambda x:x.year) <= year
            else:
                mask = df['citation_birth_date'].apply(lambda x:x.year) == year
            
            subset = df[mask]
            if subset.shape[0]==0:
                self.plot_empty(ax)
            else:
                g, pos, nodes, impt = self.prepare_graph(subset, 'citing_id', 'cited_id')
                with_labels = True if year >= label_min_year else False
                self.plot_graph(ax, g, pos, nodes, impt, with_box=True, with_labels=with_labels, node_kwargs=node_kwargs, edge_kwargs=edge_kwargs)
            plt.title(year)
        return fig

    def save_yearly_graph(self, df, start_year, end_year, savepath, label_min_year = None, plot_width=3, cumulative=False, node_kwargs = {}, edge_kwargs = {}):
        df['citation_birth_date'] = pd.to_datetime(df['citation_birth_date'])
        num_years = end_year - start_year

        if not label_min_year:
            label_min_year = start_year

        for i, year in enumerate(range(start_year, end_year)):
            fig = plt.figure(figsize=(plot_width, plot_width))
            ax = plt.gca()
            if cumulative:
                mask = df['citation_birth_date'].apply(lambda x:x.year) <= year
            else:
                mask = df['citation_birth_date'].apply(lambda x:x.year) == year
            
            subset = df[mask]
            if subset.shape[0]==0:
                self.plot_empty(ax)
            else:
                g, pos, nodes, impt = self.prepare_graph(subset, 'citing_id', 'cited_id')
                with_labels = True if year >= label_min_year else False
                self.plot_graph(ax, g, pos, nodes, impt, with_box=True, with_labels=with_labels, node_kwargs=node_kwargs, edge_kwargs=edge_kwargs)
            plt.title(year)
            plt.savefig(savepath+'/{}.png'.format(year))
        return fig, ax

    def make_gif(self, png_dir, **kwargs):
        images = []
        for file_name in os.listdir(png_dir):
            if file_name.endswith('.png'):
                file_path = os.path.join(png_dir, file_name)
                images.append(imageio.imread(file_path))
        imageio.mimsave(png_dir+'/animated.gif', images, **kwargs)

    def test(self, edge_df):
        pass
