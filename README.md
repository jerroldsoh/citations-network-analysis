# citations-network-analysis
Functions for legal citations analysis. For enquires contact Jerrold Soh, Singapore Management University School of Law (jerroldsoh@smu.edu.sg)

## Overview

This is a WIP repo for a system that can analyze Singapore reported judgements. Most of the stuff is currently hard-coded to my dataset in the sense that the column header names are written in. System architecture is as follows:

1. Data flows in the following manner:
    1. INPUT - for now it expects a .csv 
    1. TRANSFORMERS - essentially functions that extract citations and enrich them with metadata. 
    1. ANALYZERS - classes with prebuilt functions that perform typical citations analysis tasks such as get_centrality_scores (including over time for each case). See Fowler and Jeon 2007 "The Authority of Supreme Court Precedent" for theory and examples.
    1. PIPELINES - scripts that instantiate and run classes implemented in TRANSFORMERS and/or ANALYZERS. Pipelines are meant for repetitive set tasks like creating data, testing, etc. Kinda like the poor man's Airflow / ETLs
    1. NOTEBOOKS - notebooks used for data analysis. Similar to pipelines except analysis is way more iterative.
    1. ARTEFACTS - Graphs and tables usually output by notebooks using functions like .save(), .to_latex(), etc for inclusion in the ultimate research paper.
1. INPUT: Assumes user has already downloaded and parsed the judgements into a .csv file which contains minimally
    1. Text of the judgement in its own column known as 'judgment_paras'
    1. Other meta-data like 'decision_date', 'judgment_word_count', etc
    1. Note that in real life I used a HTML parser which I can't put up here for reasons.
1. TRANSFORMERS: See create_datasets.py for the flow and how they are used. Transformers will output 2 csvs - one edgelist for the graph which each row is one citation and one metadata.csv that contains metadata on cases (each row is one case). Both are instrumental for the subsequent analysis.
1. ANALYZERS: Analysis is split into different types - functions for analysing graph data, case data, and within each there are functions for analysing by topic vs by time series. The class inheritance reflects this. The functions have some overlap and are not totally optimised. See notebooks for how they are used. Note that the functions are written for usage from Jupyter and may not be optimal to run from terminal.
1. PIPELINES: Meant for running in terminal. Need to supply input and outputh filepaths.
1. NOTEBOOKS: Usually one notebook for one paper, with code organised in the order that the tables/figures appear (but not always). Expect notebooks to be messy.
1. ARTEFACTS: Static files like .png and .tex files that should not be pushed to repo

## Repo contribution guide

By invite only for now. Contact jerroldsoh@smu.edu.sg for inquiries.
Otherwise we follow standard issue-pullrequent-review-merge workflow.

## Maintenance details:

* Unit testing not implemented
* Code does not validate input

## Licence

[GNU v3](https://www.gnu.org/licenses/gpl-3.0.en.html) but always subject to any terms imposed by Singapore Management University on my work, which shall always take precedence over the GNU terms.
