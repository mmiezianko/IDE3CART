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
        neighborhood_list[hash_to_int[k]]={'value': v['value'],
                                           'col':v['col'],
                                           'children':[hash_to_int[v['children'][0]], hash_to_int[v['children'][1]]],

                                           }
    print(neighborhood_list)
    return neighborhood_list

def draw_graph(nb_list):
    G = nx.DiGraph()
    vertices = list(set([k for k in nb_list.keys()]))
    for v in nb_list.values():
        vertices.extend(v['children'])
    G.add_nodes_from(set(vertices))
    edge_dict={}
    for k, v in nb_list.items():
        G.add_edge(k,v['children'][0], minlen=10)
        edge_dict[(k,v['children'][0])] = {'value':v['value'], 'col':v['col']}
        G.add_edge(k,v['children'][1], minlen=10)
        edge_dict[(k, v['children'][1])] = {'value': 'others', 'col': v['col']}

    for k,v in nb_list.items():
        G.nodes[k]['value'] = v['value']

    return G, edge_dict
def get_hash_dict(d_tree, hash_dict):
    if d_tree.branch_with_value is not None:
        hash_dict[d_tree.__hash__()] = {'value':d_tree.value,
                                        'col':d_tree.col,
                                        'children': [d_tree.branch_with_value.__hash__(), d_tree.branch_with_others.__hash__()]
                                        }
        get_hash_dict(d_tree.branch_with_value, hash_dict)
        get_hash_dict(d_tree.branch_with_others, hash_dict)





if __name__ == "__main__":
    trainingData3 = loadCSV('data/iris.csv')  # demo data from matlab

    decisionTree3 = grow_tree(trainingData3)
    plot(decisionTree3)
    prune(decisionTree3, 0.5)
    plot(decisionTree3)
    G, edge_dict = draw_graph(get_neighborhood_list(decisionTree3))
    plt.figure(figsize=(15,15))
    plt.subplot(111)

    pos = nx.kamada_kawai_layout(G,scale= 5)
    nx.draw_networkx_edge_labels(G,pos,edge_dict,verticalalignment='center')
    nx.draw(G, pos)
    plt.show()