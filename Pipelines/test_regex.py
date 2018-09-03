import os
import pandas as pd
import argparse
from Transformers.CiteExtractor import CiteExtractor

# usage: python -m Pipelines.test_regex --infile path\to\testdata

def run(infile):
	data_from_html_parser = pd.read_csv(infile).set_index('reporter_citation')
	
	print('Now extracting citations...')
	ce = CiteExtractor()
	cites = ce.test_transform(data_from_html_parser)
	data_from_html_parser['ls_cited_cases'] = cites
	print('Printing 5 sample processed cases...')
	print(data_from_html_parser.sample(5))

	return None
	
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--infile', help='absolute path to lawnet_processor output',
		default=None)
	args = parser.parse_args()
	infile = args.infile if args.infile else os.path.join(os.path.dirname(__file__), '..\\Data\\test_data\\test_html_parser_output.csv')
	return run(infile)

if __name__ == '__main__':
	main()
