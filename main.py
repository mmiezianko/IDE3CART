import numpy as np
import pandas as pd
from decision_tree.grow_decision_tree import grow_tree

import csv
import numpy as np
import pandas as pd
from decision_tree.grow_decision_tree import grow_tree, plot, prune, predict
from decision_tree.open_file import loadCSV
import matplotlib.pyplot as plt
import networkx as nx



def get_hash_dict(d_tree, hash_dict):
    if d_tree.value is not None:
        hash_dict[d_tree.__hash__()] = [d_tree.branch_with_value.__hash__(), d_tree.branch_with_others.__hash__()]
        get_hash_dict(d_tree.branch_with_value,hash_dict)
        get_hash_dict(d_tree.branch_with_others,hash_dict)
def get_neighborhood_list(d_tree):
    dict = {}
    get_hash_dict(d_tree, dict)
    print(dict)
    values = [k for k in dict.keys()]
    for v in dict.values():
        values.extend(v)
    hashes = set(values)
    hash_to_int = {k:i for i,k in enumerate(hashes)}
    print(hash_to_int)
    neighborhood_list = {}
    for k,v in dict.items():
        neighborhood_list[hash_to_int[k]]=[hash_to_int[v[0]], hash_to_int[v[1]]]
    print(neighborhood_list)
    return neighborhood_list

def draw_graph(nb_list):
    G = nx.Graph()
    G.add_nodes_from(nb_list.keys())
    for k, v in nb_list.items():
        G.add_edge(k,v[0])
        G.add_edge(k,v[1])
    plt.subplot(1)
    print("piniting...")
    nx.draw_shell(G, with_labels=True)

if __name__ == "__main__":
    trainingData3 = loadCSV('data/iris.csv')  # demo data from matlab

    decisionTree3 = grow_tree(trainingData3)
    plot(decisionTree3)

    print(get_neighborhood_list(decisionTree3))
