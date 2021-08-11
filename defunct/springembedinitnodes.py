#!/usr/bin/env python3

import _pickle as cPickle
import matplotlib
import matplotlib.pyplot as plt
import torch_geometric as tg
import networkx as nx
import sys

def spring_loaded_embedding(G2, name="spring", node_color = "#1f78b4", seed=173):
    #edges = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))
    pos = nx.spring_layout(G2, k=5, iterations=100, seed=seed)
    alphas = [0.7 for z in G2.nodes()]
    # for i, arc in enumerate(edges):
    #     arc.set_alpha(alphas[i])
    #nx.draw_networkx_nodes(g, pos, node_size=nodesizes,linewidth=edgewidths,node_color=edgecolors)
    nx.draw(G2, pos, node_size=25, node_color=node_color, with_labels=False, alpha=0.4)
    plt.savefig("{}_graph.pdf".format(name))
    plt.clf()
    return None
    
fname =  str(sys.argv[1])
ffragname = fname.split('/')[-1]
with open(fname, 'rb') as f:
    data_obj = cPickle.load(f)

fullG, community_dict = data_obj

colorbank = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
#make bank large enough
#identify communities by "boundary edge"

color_map = list()
group_set = list(set(community_dict.values()))

for node in fullG:
    for i, group in enumerate(group_set):
        if community_dict[node] == group:
            color_map.append(colorbank[i])
            continue

spring_loaded_embedding(fullG, f"fullspring_{ffragname}", color_map)
