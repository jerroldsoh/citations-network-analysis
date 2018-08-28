import os
import pandas as pd
import numpy as np
import ast
import argparse
import networkx as nx # used to find incoming cites quickly
from Transformers.CiteExtractor import CiteExtractor
from Transformers.CiteTransformer import CiteTransformer
from Transformers.EdgeDataTransformer import EdgeDataTransformer 
from Transformers.MetaDataTransformer import MetaDataTransformer

def run(infile, meta_out, edge_out):
	data_from_html_parser = pd.read_csv(infile).set_index('reporter_citation')
	
	print('Now extracting citations...')
	ce = CiteExtractor()
	cites = ce.fit_transform(data_from_html_parser)
	data_from_html_parser['ls_cited_cases'] = cites
	
	print('Now enriching citations data...')
	ct = CiteTransformer()
	cite_meta = ct.fit_transform(data_from_html_parser)
	data_from_html_parser = data_from_html_parser.join(cite_meta)

	print('Now extracting metadata...')
	mdt = MetaDataTransformer()
	metadata = mdt.fit_transform(data_from_html_parser)
	extracted_metadata = metadata.combine_first(data_from_html_parser)

	print('Now reshaping to edge data...')
	edt = EdgeDataTransformer()
	edge_data, id_dict, reverse_dict = edt.fit_transform(extracted_metadata)

	# get number of cases citing each case 
	dg = nx.from_pandas_edgelist(edge_data, 'citing', 'cited', create_using=nx.DiGraph())
	in_deg = pd.Series(dict(dg.in_degree()), name='num_citing_cases')

	# both have index reporter citation
	extracted_metadata = extracted_metadata.join(in_deg)
	# fillna because isolates will not feature in in_deg, fill with 0
	extracted_metadata['num_citing_cases'] = extracted_metadata['num_citing_cases'].fillna(0)

	colname_prefixes_to_drop = ('counsel', 'firm_', 'filename', 'judgment_paras', 'all_citations', 'double_check',
		'is_reported', 'type', 'file_name',
		'raw_parties', 'dissenting', 'list_', 'lawnet_subject_matter', 'topleft_', 'suit_num',)
	extracted_metadata = extracted_metadata[[
		col for col in extracted_metadata.columns if not col.startswith(colname_prefixes_to_drop)
	]]

	print('Now saving metadata...')
	extracted_metadata.to_csv(os.path.join(os.path.dirname(__file__), '..\\Data\\{}.csv'.format(meta_out)))
	print('Now saving edge data...')
	edge_data.to_csv(os.path.join(os.path.dirname(__file__), '..\\Data\\{}.csv'.format(edge_out)), index=False)
	print('Now saving reference dicts...')
	pd.Series(reverse_dict).to_csv(os.path.join(os.path.dirname(__file__), '..\\Data\\ids_to_citations.csv'))
	extracted_metadata['case_title'].to_csv(
		os.path.join(os.path.dirname(__file__), '..\\Data\\citations_to_names.csv')
	)

	return extracted_metadata, edge_data
	
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('infile', help='relative path to lawnet_processor output')
	parser.add_argument('--meta_out', default='transformed_metadata', help='relative path to output meta data')
	parser.add_argument('--edge_out', default='reshaped_edge_data', help='relative path to output edge data')
	args = parser.parse_args()
	return run(args.infile, args.meta_out, args.edge_out)

if __name__ == '__main__':
	main()
