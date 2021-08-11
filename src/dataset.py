import torch
import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import torch.utils.data
import itertools
from collections import Counter
from random import shuffle
import json
#
from networkx.readwrite import json_graph
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import pdb
import time
import random
import pickle
import os.path
import torch_geometric as tg
import torch_geometric.datasets
import time

from torch_geometric.data import Data, DataLoader

from utils import precompute_dist_data, get_link_mask, duplicate_edges, deduplicate_edges

###
#
#  Adapted from Jiyuan You https://github.com/JiaxuanYou/P-GNN
#
#
#
###
def get_tg_dataset(args, dataset_name, use_cache=True, remove_feature=False):
    print('access')
    try:
        data = load_tg_dataset(dataset_name, args.labelpath)#args.labelpath)
    except:
        raise NotImplementedError

    dataset_name = dataset_name.split('/')[-1]
    # precompute shortest path
    if not os.path.isdir('datasets'):
        os.mkdir('datasets')

    if not os.path.isdir('datasets/cache'):
        os.mkdir('datasets/cache')

    f1_name = 'datasets/cache/' + dataset_name + str(-1) + '_dists.dat'
    f2_name = 'datasets/cache/' + dataset_name + str(-1)+ '_dists_removed.dat'
    f3_name = 'datasets/cache/' + dataset_name + str(-1)+ '_links_train.dat'
    f4_name = 'datasets/cache/' + dataset_name + str(-1)+ '_links_val.dat'
    f5_name = 'datasets/cache/' + dataset_name + str(-1)+ '_links_test.dat'

    if use_cache and ((os.path.isfile(f2_name) and args.task=='link') or (os.path.isfile(f1_name) and args.task!='link')):
        with open(f3_name, 'rb') as f3, \
            open(f4_name, 'rb') as f4, \
            open(f5_name, 'rb') as f5:
            links_train_list = pickle.load(f3)
            links_val_list = pickle.load(f4)
            links_test_list = pickle.load(f5)

        if args.task=='link':
            with open(f2_name, 'rb') as f2:
                dists_removed_list = pickle.load(f2)
        else:
                dists_list = pickle.load(f1)

        print('Cache loaded!')
        data_list = []

        if args.task == 'link':
            data.mask_link_positive = deduplicate_edges(data.edge_index.numpy())
        data.mask_link_positive_train = links_train_list
        data.mask_link_positive_val = links_val_list
        data.mask_link_positive_test = links_test_list
        get_link_mask(data, resplit=False)

        if args.task=='link':
            data.dists = torch.from_numpy(dists_removed_list).float()
            data.edge_index = torch.from_numpy(duplicate_edges(data.mask_link_positive_train)).long()
        else:
            data.dists = torch.from_numpy(dists_list).float()
        if remove_feature:
            data.x = torch.ones((data.x.shape[0],1))
    else:
        # data_list = []
        dists_list = []
        dists_removed_list = []
        links_train_list = []
        links_val_list = []
        links_test_list = []

        if 'link' in args.task:
            get_link_mask(data, args.remove_link_ratio, resplit=True,
                          infer_link_positive=True if args.task == 'link' else False)


        links_train_list=data.mask_link_positive_train
        links_val_list=data.mask_link_positive_val
        links_test_list= data.mask_link_positive_test


        if args.task=='link':
            dists_removed = precompute_dist_data(data.mask_link_positive_train, data.num_nodes,
                                                 approximate=args.approximate)
            dists_removed_list = dists_removed
            data.dists = torch.from_numpy(dists_removed).float()
            data.edge_index = torch.from_numpy(duplicate_edges(data.mask_link_positive_train)).long()

        else:
            dists = precompute_dist_data(data.edge_index.numpy(), data.num_nodes, approximate=args.approximate)
            dists_list.append(dists)
            data.dists = torch.from_numpy(dists).float()
        if remove_feature:
            data.x = torch.ones((data.x.shape[0],1))
        # data_list.append(data)

        with open(f1_name, 'wb') as f1, \
            open(f2_name, 'wb') as f2, \
            open(f3_name, 'wb') as f3, \
            open(f4_name, 'wb') as f4, \
            open(f5_name, 'wb') as f5:
            if args.task=='link':
                pickle.dump(dists_removed_list, f2)
            else:
                pickle.dump(dists_list, f1)
            pickle.dump(links_train_list, f3)
            pickle.dump(links_val_list, f4)
            pickle.dump(links_test_list, f5)

    return data


def nx_to_tg_data(graph_o, feature, edge_label=None):
#
    feature = feature
    graph = graph_o.copy()
    graph.remove_edges_from(nx.selfloop_edges(graph))
#
    # relabel graphs
    keys = list(graph.nodes)
    vals = range(graph.number_of_nodes())
    mapping = dict(zip(keys, vals))
    nx.relabel_nodes(graph, mapping, copy=False)
#
    x = np.zeros(feature.shape)
    graph_nodes = list(graph.nodes)
    for m in range(feature.shape[0]):
        x[graph_nodes[m]] = feature[m]
#
    x = torch.from_numpy(x).float()
#
    # get edges
    edge_index = np.array(list(graph.edges))
    edge_index = np.concatenate((edge_index, edge_index[:,::-1]), axis=0)
    edge_index = torch.from_numpy(edge_index).long().permute(1,0)
#
    #print(f"graphs: {graph}\n \"x\": {x}\nedge_index: {edge_index}")
#
    data = Data(x=x, edge_index=edge_index)
    # get edge_labels
    if edge_label[0] is not None:
        mask_link_positive = np.stack(np.nonzero(edge_label))
        data.mask_link_positive = mask_link_positive
#
    return data



def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


# main data load function
def load_graphs(dataset_str, labelpath):
    node_labels = [None]
    edge_labels = [None]
    idx_train = [None]
    idx_val = [None]
    idx_test = [None]
#
    # elif dataset_str == 'email':
    dataset_str='../staging/edge_gt20_C8_microC_1Mto1100K.txt'
    with open(dataset_str, 'rb') as f:
        hh = nx.read_edgelist(f)
#
    # with open('/home/spmoran/temp_spmoran/torch2/test_chr8/edgeonlyCHR8_1MBto2MB.txt', 'rb') as f:
    #     hh = nx.read_edgelist(f)
    # with open('/home/spmoran/temp_spmoran/torch2/old.P-GNN/data/email.txt', 'rb') as f:
    #     hh = nx.read_edgelist(f)
#
    # label_all = np.loadtxt(labelpath)
    label_all = np.loadtxt('../thresholding/gt20_labels.txt')
    #label_all = np.loadtxt('/home/spmoran/temp_spmoran/torch2/chr8v1/staging/graph_labels.txt')
#
    graph_label_all = label_all.copy()
    graph_label_all[:,1] = graph_label_all[:,1]//6
#
    label_dict = dict()
    for u,v in graph_label_all:
        key = str(u).split(".")[0]
        label_dict[key] = v
#
    for u in hh.nodes():
        if u not in label_dict:
            hh.remove_node(u)
            print(f"removed node: {u}")
            continue
#
    for edge in list(hh.edges()):
        try:
            assert label_dict[edge[0]]
        except KeyError:
            hh.remove_node(edge[0])
#
    n = hh.number_of_nodes()
    #former features
    feature = np.ones((n, 1))
    #formed edge labels
    edge_label = np.zeros((n, n),dtype=int)
    for i, u in enumerate(hh.nodes()):
        for j, v in enumerate(hh.nodes()):
            if label_dict[str(u)] == label_dict[str(v)] and i>j:
                edge_label[i, j] = 1
#
    return hh, feature, edge_label, node_labels, idx_train, idx_val, idx_test


def load_tg_dataset(name, labelpath):
    graphs, features, edge_labels,_,_,_,_ = load_graphs(name, labelpath)
    return nx_to_tg_data(graphs, features, edge_labels)

