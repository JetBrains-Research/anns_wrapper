from typing import Any, List

import numpy
import pyflann
from numpy.random import *
from pyflann import *

from interface import NNSWrapper
from utility.metrics import Metrics


class FlannWrapper(NNSWrapper):
    """
    Bruteforce realisation.
    """
    index = None
    is_built = False
    max_item_num = 0

    def __init__(self, dimensions: int, distance: str):
        pyflann.index.set_distance_type(distance)
        self.index = FLANN()
        self.initial_index_vectors = []
        self.pool = []
        self.dimensions = dimensions
        self.distance_name = distance
        self.distance = Metrics.get_metric(self.distance_name)
        self.deleted_ids = []
        self.params = None

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
        self.params = self.index.build_index(numpy.array(self.initial_index_vectors), target_precision=target_precision,
                                             log_level=log_level)
        self.initial_index_vectors = []
        self.is_built = True

    def rebuild_index(self, trees: int = 10):
        self.initial_index_vectors = [x for x in self.index.__dict__['_FLANN__curindex_data'] if
                                      x not in self.deleted_ids] + self.pool
        self.build_index()
        self.pool = []
        self.deleted_ids = []

    def search(self, vector: numpy.ndarray, neighbors: int, include_distances: bool = False):
        query = self.index.nn_index(numpy.array([vector]), neighbors * 2, checks=self.params["checks"])
        print(query)
        pool_distances = [(x, self.distance(vector, x)) for x in self.pool]
        index_distances = list(zip(*query))
        distances = [x for x in pool_distances + index_distances if x not in self.deleted_ids]
        res = sorted(distances, key=lambda x: x[1])[:neighbors]
        if include_distances:
            return res
        else:
            return [x[0] for x in res]


if __name__ == '__main__':
    dataset = rand(100, 128)
    testset = rand(1000, 128)
    flann = FLANN()
    # params = flann.build_index(dataset, target_precision=0.9, log_level="info")
    w = FlannWrapper(128, "euclidean")
    for i in dataset:
        w.add_vector(i)
    w.build_index()

    w.rebuild_index()
    print(w.search(dataset[0], 3))

    # print(flann.__dict__['_FLANN__curindex_data'])
    # result, dists = flann.nn_index(testset, 5, checks=params["checks"]);
    # print(result)
