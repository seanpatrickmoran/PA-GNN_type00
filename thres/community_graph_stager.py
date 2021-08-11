#! /usr/bin/env python3

import networkx as nx
import numpy as np
import sys

"""
./community_graph_stager.py chromosome_HiC_contactfreq_edgelist.txt tadcaller_bounds_coordinates.txt > graphNN_labels.txt

"""

def between(item, low, high):
    return (low <= item <= high)

TAD_BANK = list()
with open(sys.argv[2], "r") as target:
    for line in target:
        _, hi  = line.rstrip("\n").split("\t")
        TAD_BANK.append(hi)
# with open("/home/spmoran/1to2MB_weak/weak8_1to2MB_formfeed.txt", "r") as target:
#     for line in target:
#         _, hi  = line.rstrip("\n").split("\t")
#         TAD_BANK.append(hi)

with open(sys.argv[1], "r") as f:
    graph = nx.read_edgelist(f)

TAD_BANK.sort()

community_ids = np.zeros(len(graph.nodes))
community_ids[:] = np.NaN
ptr1 = 0
ptr2 = 0
for j in range(0, len(TAD_BANK)):
    ptr1 = ptr2
    ptr2 = int(TAD_BANK[j])
    for i,n in enumerate(graph.nodes):
        if between(int(n), ptr1, ptr2):
            community_ids[i] = j
            continue
        elif int(n) >= int(TAD_BANK[-1]):
            community_ids[i] = len(TAD_BANK)
            continue
        elif int(n) <= int(TAD_BANK[0]):
            community_ids[i] = 0
            continue

            nodelist = list(graph.nodes())

TAD_dict = dict()
for i in range(len(nodelist)):
    TAD_dict[nodelist[i]] = int(community_ids[i])

TAD_CENTERS = list()
for community in range(0, len(TAD_BANK)+1):
    temp_list = list()
    for key, value in TAD_dict.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if value == community:
            temp_list.append(key)
    temp_list.sort()
    TAD_CENTERS.append(temp_list[len(temp_list)//2])

write_to_file = f""
for i in range(0, len(nodelist)):
    centered_community = TAD_CENTERS[TAD_dict[nodelist[i]]]
    write_to_file += f"{nodelist[i]}\t{int(centered_community)}\n" ###, coalesce IDs into a node!!!

print(write_to_file)
