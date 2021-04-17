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




def get_neighborhood_list(d_tree):
    dict = {}
    get_hash_dict(d_tree, dict)
    print(dict)
    values = [k for k in dict.keys()]
    for v in dict.values():
        values.extend(v['children'])
    hashes = set(values)
    hash_to_int = {k:i for i,k in enumerate(hashes)}
    print(hash_to_int)
    neighborhood_list = {}
    for k,v in dict.items():
        neighborhood_list[hash_to_int[k]]={'value': v['value'], 'children':[hash_to_int[v['children'][0]], hash_to_int[v['children'][1]]]}
    print(neighborhood_list)
    return neighborhood_list

def draw_graph(nb_list):
    G = nx.Graph()
    vertices = list(set([k for k in nb_list.keys()]))
    for v in nb_list.values():
        vertices.extend(v['children'])
    G.add_nodes_from(set(vertices))
    for k, v in nb_list.items():
        G.add_edge(k,v['children'][0])
        G.add_edge(k,v['children'][1])

    return G
def get_hash_dict(d_tree, hash_dict):
    if d_tree.branch_with_value is not None:
        hash_dict[d_tree.__hash__()] = {'value':d_tree.value, 'children': [d_tree.branch_with_value.__hash__(), d_tree.branch_with_others.__hash__()]}
        get_hash_dict(d_tree.branch_with_value, hash_dict)
        get_hash_dict(d_tree.branch_with_others, hash_dict)


if __name__ == "__main__":
    trainingData3 = loadCSV('data/iris.csv')  # demo data from matlab

    decisionTree3 = grow_tree(trainingData3)
    plot(decisionTree3)
    prune(decisionTree3, 0.5)
    plot(decisionTree3)
    G = draw_graph(get_neighborhood_list(decisionTree3))
    plt.subplot(111)
    nx.draw_spring(G)
    plt.show()