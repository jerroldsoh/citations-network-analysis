'''
Extracts case citations given regex patterns for citations and
a dataframe with column containing the raw text

TODO: Generalise to take from .config file
'''
import logging
from sklearn.base import BaseEstimator, TransformerMixin
import re
import string
import os
import pandas as pd

class CiteExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, output_named_groups = False, output_unique_only = True):
        self.reports_to_consider = ['SLR', 'SLR(R)'] # unused for now
        self.output_named_groups = output_named_groups
        self.output_unique_only = output_unique_only

    def preprocess_for_comparison(self, the_str):
        the_str = ''.join([c for c in the_str if not c in string.punctuation])
        if the_str != 'PP': # to avoid matching public prosecutor when looking for pages
            the_str = the_str.lower()
        return the_str

    def fit(self, df):
        return self

    def find_cites(self, case_text):
        if not case_text:
            print('No case text for', case_text)
            return []
        pat = re.compile(r'''
            \[(?P<year>\d{1,4})\]\s
            (?P<volume>\d{1,3}\s)?                  # optional in case of things like [YYYY] SLR 1
            (?P<report>
            S?SLR(?:\(R\))?)                        # SLR, SSLR, SLR(R)
            \s
            (?P<page>\d{1,4})
        ''', re.VERBOSE | re.I)
        matches = pat.finditer(case_text)
        if not matches:
            return []

        if self.output_named_groups:
            out = [match.groupdict() for match in matches] # so that it returns in a nicely named dict
        else:
            out = [match.group(0) for match in matches]

        if self.output_unique_only:
            out = list(set(out))

        return out

    def transform(self, df):
        cites = df['judgment_paras'].apply(self.find_cites)
        cites.name = 'ls_cited_cases'
        return cites


    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def test_one_regex(self, case_text, regex_to_test):
        if not case_text:
            print('No case text for', case_text)
            return []

        matches = regex_to_test.finditer(case_text)
        if not matches:
            return []

        if self.output_named_groups:
            out = [match.groupdict() for match in matches] # so that it returns in a nicely named dict
        else:
            out = [match.group(0) for match in matches]

        if self.output_unique_only:
            out = list(set(out))

        return out

    def test_transform(self, df, regex_to_test=''):
        if not regex_to_test:
            regex_to_test = re.compile(r'''
                \[(?P<year>\d{1,4})\]\s
                (?P<volume>\d{1,3}\s)?                  # optional in case of things like [YYYY] SLR 1
                (?P<report>
                SGCA)
                \s
                (?P<page>\d{1,4})
            ''', re.VERBOSE | re.I)
        cites = df['judgment_paras'].apply(self.test_one_regex, args=(regex_to_test,))
        cites.name = 'ls_cited_cases'
        return cites

    def get_feature_names(self):
        if not self.feature_names:
            print('Feature names not set. Transform must be called first')
        else:
            return self.feature_names

    def test(self):
        assert len(find_cites('sadadas [2014] 1 SLR 2014 131232145')) == 1
        assert len(find_cites('I rely upon [2014] SSLR 2014.')) == 1
        assert not find_cites('I wrongly cited [2014] 0 SLR(R)  ')
        assert len(find_cites('Article (2014) SacLJ 9. See also [2014] MLJ 4')) == 1
        assert len(find_cites('A malaysia case by judge 1231256 dsa [2014] MC 14')) == 1

if __name__ == '__main__':
    ce = CiteExtractor()
    ce.test()
