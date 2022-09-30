import functools as ft
from json.encoder import INFINITY
from xmlrpc.client import MAXINT
from scipy import stats
import random
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
import subprocess
import os
import sys
import argparse
from datetime import datetime
import tqdm
import pylab as plb
import misc
import multiprocessing as mp
import shutil

time_suffix = datetime.today().strftime("%d-%m-%Y_%H-%M-%S")
script_abspath, script_name_w_ext = misc.get_script_abspath_n_name(__file__)
script_name = script_name_w_ext.split('.')[0]


def find_all_paths(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]
    if not start in graph:
        return []
    paths = []
    for node in graph[start]:
        if node not in path:
            newpaths = find_all_paths(graph, node, end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths


def main():
    # Parse console arguments
    parser = argparse.ArgumentParser(description="Find all paths in the graph")
    parser.add_argument("-i", "--input", type=str,
                        default="./", help="Input dirpath")
    parser.add_argument("-o", "--output", type=str,
                        default="./", help="Output dirpath")
    parser.add_argument("-mxw", "--maxweight", type=int,
                        default=MAXINT, help="Path max weight")
    parser.add_argument("-mxe", "--maxedges", type=int,
                        default=MAXINT, help="Path max edges")
    args = parser.parse_args()
    args.input = args.input.rstrip('/')
    args.output = args.output.rstrip('/')
    if args.output == "./":
        args.output = args.input

    # Create output folder
    out_dirpath = args.output + f'/all-paths_{time_suffix}'
    if not os.path.isdir(out_dirpath):
        os.mkdir(out_dirpath)

    # Create logger
    logger = misc.get_logger(
        __name__, log_fpath=out_dirpath + f'/{script_name}.log')
    logger.info(f"Started {script_abspath=} with args: {args}")

    logger.info(f"Read {args.input}/links.csv")
    links_df = pd.read_csv(args.input + '/links.csv')

    logger.info(f"Create graph")
    G = nx.from_pandas_edgelist(
        links_df, source='src', target='trg', edge_attr='weight', create_using=nx.Graph)

    logger.info(f"Find all paths")
    all_paths = []
    for source in tqdm.tqdm(G.nodes):
        for target in G.nodes:
            if source == target:
                continue
            for path in nx.all_simple_paths(G, source=source, target=target):
                all_paths.append(path)

    logger.info(f'Sort paths')
    all_paths.sort(key=len)

    all_paths = [path for path in all_paths if len(path) <= args.maxedges]
    all_paths = [path for path in all_paths if sum(
        [G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1)]) <= args.maxweight]

    logger.info(f"Save all paths to {out_dirpath}/PathList.csv")
    header = ['PathID', 'SourceID', 'TargetID', 'LinkPath']
    with open(out_dirpath + '/PathList.csv', 'w') as f:
        f.write(','.join(header) + '\n')
        for path_id, path in enumerate(all_paths):
            f.write(f'{path_id},{path[0]},{path[-1]},{",".join(path)}' + '\n')


if __name__ == '__main__':
    main()
