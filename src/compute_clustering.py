"""
compute_clustering.py
---------------------

Compute the clustering coefficients of several hypergraphs.

"""

from hyper import read_hypergraph


def main():
    """Load data and compute."""
    hyper = read_hypergraph('data/email-Enron/')
    print(hyper.clustering(1))


if __name__ == '__main__':
    main()
