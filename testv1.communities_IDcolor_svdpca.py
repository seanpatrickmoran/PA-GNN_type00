import _pickle as cPickle
import networkx as nx
from sys import argv
#from sys import argv

import torch
import numpy as np
import torch.utils.data
#
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

import time
import random


#fname = 'edge_gt20_C8_microC_1Mto1100K_outANDdata.pkl' 
fname = argv[1]
with open(fname, 'rb') as f:
    data_obj = cPickle.load(f)

A,B = data_obj

#with open('/home/spmoran/temp_spmoran/torch2/microC_pilot_redux/single_redux/staging/edge_gt20_C8_microC_1Mto1100K.txt', 'rb') as f:
#    hh = nx.read_edgelist(f)
#
#labelpath = '/home/spmoran/temp_spmoran/torch2/microC_pilot_redux/single_redux/thresholding/gt20_labels.txt'

with open(argv[2], 'rb') as f:
    hh = nx.read_edgelist(f)

labelpath = argv[3]
label_all = np.loadtxt(labelpath)

tg_dict = {}
for i,n in enumerate(hh.nodes):
    tg_dict[i+1] = n

graph_label_all = label_all.copy()
label_dict = dict()
for u,v in graph_label_all:
    key = str(u).split(".")[0]
    label_dict[key] = v


for u in hh.nodes():
    if u not in label_dict:
        hh.remove_node(u)
        continue

for edge in list(hh.edges()):
    cantrip = False
    try:
        assert label_dict[edge[0]]
    except KeyError:
        hh.remove_node(edge[0])
        cantrip = True
    if cantrip:
        continue
    if label_dict[edge[0]] != label_dict[edge[1]]:
        hh.remove_edge(edge[0], edge[1])


###Community Detection
comps = [comp for comp in nx.connected_components(hh) if len(comp)>10] ##fix this
graphs = [hh.subgraph(comp) for comp in comps]

###Color mapping
group_dict = dict()
for n in range(1,hh.number_of_nodes()+1):
    local = tg_dict[n]
    flag = False
    for i,z in enumerate(comps,1):
        if local in z:
            group_dict[n]=i
            flag = True
    if not flag:
        group_dict[n]=0

groups = list(group_dict.values())

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
for k in ['w','white','wheat','silver','whitesmoke', 'aliceblue','snow','ivory','ghostwhite','seashell','floralwhite','mintcream']:
    colors.pop(k,None)

colormapping = np.random.choice(list(colors.keys()), len(set(groups)), replace=False)

cdict = dict()
for i,c in enumerate(colormapping):
    cdict[i]=c


###Embedding to lower dimension
U,S,V = torch.linalg.svd(A, full_matrices=False)
k = 2
eb_u = U[:,:k]*S[:k]
embedding = eb_u.cpu().detach().numpy()

fig, ax = plt.subplots()
for g in np.unique(groups):
    ix = np.where(groups == g)
    ax.scatter(embedding[ix,0], embedding[ix,1], c = cdict[g], label = g, s=1)


ax.legend()
plt.show()
plt.savefig(f'{argv[4]}_fullcolors_embedding.png')

