import os
import pandas as pd
import numpy as np
import ast
import argparse
import networkx as nx # used to find incoming cites quickly
from Transformers.CiteExtractor import CiteExtractor
from Transformers.CiteTransformer import CiteTransformer

def run(infile):
	data_from_html_parser = pd.read_csv(infile).set_index('reporter_citation')
	
	print('Now extracting citations...')
	ce = CiteExtractor()
	cites = ce.test_transform(data_from_html_parser)
	data_from_html_parser['ls_cited_cases'] = cites
	print(data_from_html_parser['ls_cited_cases'].apply(len).sum())
	data_from_html_parser.to_csv('test_sgca_regex.csv')

	return None
	
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('infile', help='relative path to lawnet_processor output')
	args = parser.parse_args()
	return run(args.infile)

if __name__ == '__main__':
	main()
