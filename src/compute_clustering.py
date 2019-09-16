"""
compute_clustering.py
---------------------

Compute the clustering coefficients of several hypergraphs.

"""

import numpy as np
import networkx as nx
from hyper import read_hypergraph, read_graph
import matplotlib.pyplot as plt


def main():
    """Load data and compute."""
    hyper = read_hypergraph('data/email-Enron/')
    graph = read_graph('data/email-Enron/')
    # hyper = read_hypergraph('data/human-genome/')
    # graph = read_graph('data/human-genome/')
    edge_fill = hyper.fill_coefficient()
    node_clus = hyper.clustering()
    # edges = [e for e in hyper.edges if 5 <= len(e) <= 8]
    # edges = [e for e in hyper.edges if 2 <= len(e) <= 4]
    edges = [e for e in hyper.edges if 2 <= len(e) <= 8]

    hyper_clus = {e: np.mean([node_clus[n] for n in e]) for e in edges}
    graph_clus = nx.clustering(graph)
    graph_clus = {e: np.mean([graph_clus[n] for n in e]) for e in edges}

    xx = [graph_clus[e] for e in edges]
    yy = [hyper_clus[e] for e in edges]
    color = [np.log(edge_fill[e]) for e in edges]
    size = [len(e)**(2.5) for e in edges]
    plt.figure(figsize=(16, 9))
    plt.scatter(xx, yy, s=size, cmap='viridis', c=color, alpha=1)
    plt.colorbar(label=r'$\log$ fill coefficient')
    plt.title('2-to-8-hyperedges')
    plt.xlabel('mean node clustering (in the graph)')
    plt.ylabel('mean node clustering (in the hypergraph)')
    # plt.savefig('pics/email-enron-28-medium.pdf', dpi=600)
    plt.show()



if __name__ == '__main__':
    main()
