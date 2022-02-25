import time
from typing import Any, List

import faiss  # make faiss available
import numpy

from utility.helpers import get_random_vectors
from utility.metrics import Metrics
from wrappers.interface import NNSWrapper


class FaissWrapper(NNSWrapper):
    """
    Bruteforce realisation.
    """
    index = None
    is_built = False
    max_item_num = 0

    def __init__(self, dimensions: int, distance: str):
        self.index = faiss.IndexFlatL2(dimensions)
        self.initial_index_vectors = []
        self.pool = []
        self.dimensions = dimensions
        self.distance_name = distance
        self.distance = Metrics.get_metric(self.distance_name)
        self.deleted_ids = []
        self.params = None

    def add_vector_to_index(self, vector: numpy.ndarray):
        self.initial_index_vectors.append(vector)
        self.max_item_num += 1

    def add_vector_to_pool(self, vector: numpy.ndarray):
        self.pool.append([self.max_item_num, vector])
        self.max_item_num += 1

    def add_vector(self, vector: numpy.ndarray):
        if self.is_built:
            self.add_vector_to_pool(vector)
        else:
            self.add_vector_to_index(vector)

    def delete_vector(self, d_vector_id: int):
        def flagEqual(a: Any, b: Any, flag: List[bool]):
            is_equal = a == b
            flag[0] = flag[0] or is_equal
            return is_equal

        flag = [False]
        self.pool = [[vector_id, vector] for vector_id, vector in self.pool if
                     not flagEqual(vector_id, d_vector_id, flag)]

        if not flag[0]:
            self.deleted_ids.append(d_vector_id)

    def build_index(self, target_precision=0.9, log_level="info"):
        self.params = self.index.build_index(numpy.array(self.initial_index_vectors), target_precision=target_precision,
                                             log_level=log_level)
        self.max_item_num = len(self.initial_index_vectors) + 1
        self.initial_index_vectors = []
        self.is_built = True

    def rebuild_index(self, trees: int = 10):
        old_vectors = self.index.__dict__['_FLANN__curindex_data']
        self.initial_index_vectors = [old_vectors[x] for x in range(len(old_vectors)) if
                                      x not in self.deleted_ids] + list(map(lambda x: x[1], self.pool))
        self.build_index()
        self.pool = []
        self.deleted_ids = []

    def search(self, vector: numpy.ndarray, neighbors: int, include_distances: bool = False):
        query = self.index.nn_index(numpy.array([vector]), neighbors * 2, checks=self.params["checks"])
        index_distances = list(zip(query[0][0], query[1][0]))
        pool_distances = [(x_id, self.distance(vector, x)) for x_id, x in self.pool]
        distances = [x for x in [*pool_distances, *index_distances] if x not in self.deleted_ids]
        res = sorted(distances, key=lambda x: x[1])[:neighbors]
        # print(res)
        if include_distances:
            return res
        else:
            return [x[0] for x in res]


if __name__ == '__main__':
    dataset = get_random_vectors(100000, 256, rank=10000, save=True).astype('float32')
    # d = 100  # dimension
    # nb = 1000000  # database size
    # nq = 10000  # nb of queries
    # np.random.seed(1234)  # make reproducible
    # xb = np.random.random((nb, d)).astype('float32')
    # xb[:, 0] += np.arange(nb) / 1000.
    # xq = np.random.random((nq, d)).astype('float32')
    # xq[:, 0] += np.arange(nq) / 1000.
    #
    index = faiss.IndexFlatL2(256)  # build the index
    # print(index.is_trained)
    t = time.time()
    index.add(dataset)  # add vectors to the index
    print(f"{(time.time() - t):.5f}")

    D, I = index.search(numpy.asarray([dataset[0]]), 10)
    print(D)

    t = time.time()
    index.add(dataset)  # add vectors to the index
    print(f"{(time.time() - t):.5f}")

    t = time.time()
    index.add(dataset)  # add vectors to the index
    print(f"{(time.time() - t):.5f}")
    print(index.is_trained)

    print(index.ntotal)
