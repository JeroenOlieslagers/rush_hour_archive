from plotting import *

import numpy as np
import pandas as pd
from tqdm import tqdm

from collections import namedtuple
import re
import os
import json


RAW_FILE = 'data/raw/trialdata.csv'
PROBLEM_DIR = 'data/problems'


def read_raw_data(filename):
    """
    Reads data from all subjects into list of named tuples
    """
    PsiturkRec = namedtuple('PsiturkRec','worker assignment ord t event piece move_nu move instance')
    PsiturkRec.__new__.__defaults__ = (None,) * len(PsiturkRec._fields)

    recs=[]
    with open(filename,'r') as log:
        for l in log:
            if 'event' not in l:
                continue
            v = l.split(',')[0].split(':')
            v.append(l.split(',')[1])
            v += [s.replace('[','').replace(']','') for s in re.findall('\[.*?\]',l)]
            recs.append(PsiturkRec(*v))
    return recs


def get_problem_opt_length_df(problem_dir):
    """
    Reads optimal length of puzzles from json file name into dict
    """
    jsons = os.listdir(problem_dir)
    opt_len_dict = dict([(j.split('_')[0],int(j.split('_')[1].split('.')[0])) for j in jsons])
    return opt_len_dict


def filter_weird(recs):
    """
    Removes invalid subjects
    """
    with open('Weird_subjects_2', 'r') as f:
        bad_workers = [w.strip() for w in f.readlines()]
    return [r for r in recs if r.worker not in bad_workers]


if __name__ == "__main__":
    recs = filter_weird(read_raw_data(RAW_FILE))
    opt_len_dict = get_problem_opt_length_df(PROBLEM_DIR)

    dfs = pd.DataFrame(recs)
    #dfs['puzzle_number'] = dfs['instance'].apply(lambda x: int(x[3:]))
    dfs['opt_len'] = dfs['instance'].apply(lambda x: opt_len_dict[x])

    #puzzle_matshow(dfs, opt_len_dict)
    separate_block_puzzle_matshow(dfs, opt_len_dict)

