from typing import Any, List

import numpy
from numpy.random import *
from pyflann import *

from interface import NNSWrapper
from utility.metrics import Metrics


class FlannAnnoyWrapper(NNSWrapper):
    """
    Bruteforce realisation.
    """
    index = None
    is_built = False
    max_item_num = 0

    def __init__(self, dimensions: int, distance: str):
        self.index = FLANN()
        self.initial_index_vectors = []
        self.pool = []
        self.dimensions = dimensions
        self.distance_name = distance
        self.distance = Metrics.get_metric(self.distance_name)
        self.deleted_ids = []

    def add_vector_to_index(self, vector: numpy.ndarray):
        self.initial_index_vectors.append(vector)

    def add_vector_to_pool(self, vector: numpy.ndarray):
        self.pool.append(vector)

    def add_vector(self, vector: numpy.ndarray):
        if self.is_built:
            self.add_vector_to_pool(vector)
        else:
            self.add_vector_to_index(vector)

    def delete_vector(self, d_vector: numpy.ndarray):
        def flagEqual(a: Any, b: Any, flag: List[bool]):
            is_equal = a == b
            flag[0] = flag[0] or is_equal
            return is_equal

        flag = [False]
        self.pool = [vector for vector in self.pool if not flagEqual(vector, d_vector, flag)]

        if not flag[0]:
            self.deleted_ids.append(d_vector)

    def build_index(self, target_precision=0.9, log_level="info"):
        self.index.build_index(self.initial_index_vectors, target_precision=target_precision, log_level=log_level)
        self.is_built = True

    def rebuild_index(self, trees: int = 10):
        index = FLANN()

        skipped = 0
        for i in range(self.index.get_n_items()):
            if i not in self.deleted_ids:
                index.add_item(i - skipped, self.index.get_item_vector(i))
            else:
                skipped += 1
        for i, (id, vector) in enumerate(self.pool):
            index.add_item(self.index.get_n_items() - skipped + i, vector)

        index.build(trees)

        self.index = index
        self.pool = []
        self.deleted_ids = []

        self.is_built = True

    def search(self, vector: numpy.ndarray, neighbors: int, include_distances: bool = False):
        pool_distances = [(x[0], self.distance(vector, x[1])) for x in self.pool]
        index_distances = list(zip(*self.index.get_nns_by_vector(vector, neighbors * 2, include_distances=True)))
        distances = [x for x in pool_distances + index_distances if x[0] not in self.deleted_ids]
        res = sorted(distances, key=lambda x: x[1])[:neighbors]
        if include_distances:
            return res
        else:
            return [x[0] for x in res]


if __name__ == '__main__':
    dataset = rand(10000, 128)
    testset = rand(1000, 128)
    flann = FLANN()
    params = flann.build_index(dataset, target_precision=0.9, log_level="info");
    result, dists = flann.nn_index(testset, 5, checks=params["checks"]);
    print(result)
