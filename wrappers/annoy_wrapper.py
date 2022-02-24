from typing import Any, List

import numpy
from annoy import AnnoyIndex

from utility.metrics import Metrics
from wrappers.interface import NNSWrapper


class MVPAnnoyWrapper(NNSWrapper):
    """
    Bruteforce realisation.
    """
    index = None
    is_built = False
    max_item_num = 0

    def __init__(self, dimensions: int, distance: str):
        self.index = AnnoyIndex(dimensions, distance)
        self.pool = []
        self.dimensions = dimensions
        self.distance_name = distance
        self.distance = Metrics.get_metric(self.distance_name)
        self.deleted_ids = []

    def add_vector_to_index(self, vector: numpy.ndarray):
        self.index.add_item(self.max_item_num, vector)
        self.max_item_num += 1

    def add_vector_to_pool(self, vector: numpy.ndarray):
        self.pool.append([self.max_item_num, vector])
        self.max_item_num += 1

    def add_vector(self, vector: numpy.ndarray):
        if self.is_built:
            self.add_vector_to_pool(vector)
        else:
            self.add_vector_to_index(vector)

    def delete_vector(self, vector_id: int):
        def flagEqual(a: Any, b: Any, flag: List[bool]):
            is_equal = a == b
            flag[0] = flag[0] or is_equal
            return is_equal

        flag = [False]
        self.pool = [[id, vector] for id, vector in self.pool if not flagEqual(id, vector_id, flag)]

        if not flag[0]:
            self.deleted_ids.append(vector_id)

    def build_index(self, trees: int = 10):
        self.index.build(trees)
        self.is_built = True

    def rebuild_index(self, trees: int = 10):
        index = AnnoyIndex(self.dimensions, self.distance_name)

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
