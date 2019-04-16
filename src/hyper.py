"""
hyper.py
--------

Utilities for handling hypergraphs, simplicial complexes, HONS, and graphs.

"""

import numpy as np
from os import path
from itertools import combinations
from collections import defaultdict
from scipy.special import binom


def read_hypergraph(dirpath):
    """Read a set hypergraph from disk.

    Data sets are assumed to be in the format given by [1]. Each data set
    is in a separate directory, which contains three files: a file with
    node labels, a file with the number of vertices in each hyperedge, and
    a file with the vertices in each hyperedge.

    Params
    ------

    dirpath (path): the path to the directory containing the three files.

    References
    ----------

    http://www.cs.cornell.edu/~arb/data/index.html. Accessed April 2019.

    """
    name = path.split(dirpath.rstrip('/'))[-1]
    nodes_fn = path.join(dirpath, '{}-nverts.txt'.format(name))
    edges_fn = path.join(dirpath, '{}-simplices.txt'.format(name))

    edges = set()
    with open(nodes_fn) as nodes_file, open(edges_fn) as edges_file:
        nodes_in_edges = iter(edges_file)
        for numnodes in nodes_file:
            edge = set()
            for _ in range(int(numnodes)):
                edge.add(int(next(nodes_in_edges)))
            edges.add(frozenset(edge))
    return Hypergraph(edges)


class Hypergraph:
    """A hypergraph is a tuple (V, E) where the elements of V are called nodes
    or vertices and each element of E, called an edge or hyperedge, is a
    subset of V.

    """

    def __init__(self, edges=None):
        self.edges = set()
        self.nodes = set()
        self.degree = defaultdict(int)
        self._neighbors = None
        if edges is not None:
            for edge in edges:
                self.add_edge(edge)

    def add_edge(self, edge):
        """Add a hyperedge.

        Params
        ------

        edge (set): a set of nodes.

        """
        self.edges.add(frozenset(edge))
        for node in edge:
            self.degree[node] += 1
            self.nodes.add(node)

    def hyperedges(self, node=None):
        """The edges containing the given node.

        Params
        ------

        node (int): if None(default), return all hyperedges. Else, return
        the hyperedges containing the given node.

        """
        if node is None:
            return self.edges
        return [e for e in self.edges if node in e]

    def clustering(self, node):
        """Compute the local clustering coefficient of a node.

        See equation (4) in [1].

        References
        ----------

        [1] Zhou, Wanding, and Luay Nakhleh. "Properties of metabolic graphs:
        biological organization or representation artifacts?." BMC
        bioinformatics 12.1 (2011): 132.

        """
        deg = self.degree[node]
        if deg < 2:
            return 0

        edges = self.hyperedges(node)
        value = sum(self.extra_overlap(edge1, edge2)
                    for edge1, edge2 in combinations(edges, 2))

        return value / binom(deg, 2)

    def extra_overlap(self, edge1, edge2):
        """Compute the extra overlap between two hyperedges.

        See equation (6) in [1].

        Params
        ------

        edge1, edge2 (set): two hyperedges belonging to this hypergraph.

        References
        ----------

        [1] Zhou, Wanding, and Luay Nakhleh. "Properties of metabolic graphs:
        biological organization or representation artifacts?." BMC
        bioinformatics 12.1 (2011): 132.

        """
        if edge1 == edge2:
            return 0

        diff_1 = edge1 - edge2
        diff_2 = edge2 - edge1
        numerator = len(self.neighborhood(diff_1) & diff_2) + \
                    len(self.neighborhood(diff_2) & diff_1)
        denominator = len(diff_1) + len(diff_2)
        return numerator / denominator

    def neighborhood(self, nodes):
        """The neighborhood of a set of nodes.

        The neighborhood of a node is the set of nodes that share at least
        one edge with it. The neighborhood of a set of nodes is the union
        of the neighborhoods of its nodes.

        See equation (1) in [1]. Note that there is a typo in the
        mathematical typeset of in the paragraph immediately following
        equation (1) - it should be a \cup not a \cap.

        Params
        ------

        nodes (int or set).

        See also
        --------

        Hypergraph.neighbors

        """
        if not isinstance(nodes, set) and not isinstance(nodes, frozenset):
            return self.neighbors(nodes)
        if not nodes:
            return set()
        return set().union(*(self.neighbors(n) for n in nodes))

    def neighbors(self, node):
        """The set of neighbors of a node.

        See also
        --------

        Hypergraph.neighborhood

        """
        if self._neighbors is None:
            neighs = {n: set() for n in self.nodes}
            for edge in self.edges:
                for _node in edge:
                    neighs[_node] |= edge
            for _node in self.nodes:
                neighs[_node] -= {_node}
            self._neighbors = neighs

        return self._neighbors[node]
