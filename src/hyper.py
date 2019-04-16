"""
hyper.py
--------

Utilities for handling hypergraphs, simplicial complexes, HONS, and graphs.

"""

import numpy as np


def to_set(row):
    """Convert an indicator vector to a set of nodes.

    An indivator vector is a row of an incidence matrix. For example, the
    indicator vector (0, 1, 0, 0, 1) corresponds to the set of nodes {1, 4}.

    """
    indices = np.array(range(row.shape[0]))
    return frozenset(indices[row > 0])


def to_simp(hyper):
    """Convert a hypergraph to a simplicial complex.

    Params
    ------

    hyper (np.array): A hypergraph given as incidence matrix.

    """
    simplices = set()
    for row in hyper:
        nodes = to_set(row)
        for simp in simplices:
            if simp.issubset(nodes):
                simplices.remove(simp)
                simplices.add(nodes)
                break
        else:
            simplices.add(nodes)

    return simplices
