
import sys
import numpy as np
from sklearn.cluster import KMeans


def random_uniform(n, nsubset):
    return np.random.choice(range(n), nsubset, replace=False)

def kmeans(positions, n, nsubset):
    """ Perform k-means clustering on source positions.
    """
    # perform kmeans clustering
    km = KMeans(n_clusters=nsubset, random_state=0).fit(positions)

    # randomly sample from clusters.
    subset = []
    for i in xrange(nsubset):
        cluster = [s for s, l in zip(range(n), km.labels_) if l == i]
        subset += [np.random.choice(cluster, 1)[0]]
        print cluster

    return subset