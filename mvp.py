import datetime
from abc import ABC, abstractmethod

import nmslib
import numpy
from annoy import AnnoyIndex

from helpers import get_random_vectors
from metrics import Metrics


class NNSWrapper(ABC):
    index = None
    distance = None
    is_build = False
    pool = []
    deleted_ids = []
    max_item_num = 0

    @abstractmethod
    def __init__(self, dimensions: int, distance: str):
        pass

    @abstractmethod
    def add_vector(self, vector: numpy.ndarray):
        pass

    @abstractmethod
    def delete_vector(self, vector_id: int):
        pass

    @abstractmethod
    def build_index(self, *args, **kwargs):
        pass

    @abstractmethod
    def rebuild_index(self, *args, **kwargs):
        pass

    @abstractmethod
    def search(self, vector: numpy.ndarray, neighbors: int, include_distances: bool = False):
        pass

    @abstractmethod
    def search_multiple(self, vectors: numpy.ndarray, neighbors: int, include_distances: bool = False):
        pass


class MVPNMSLIBWrapper(NNSWrapper):
    distances = {"cosinesimil": "cosine"}
    index = None
    is_built = False
    max_item_num = 0

    def __init__(self, dimensions: None, distance: str):
        self.index = nmslib.init(method='hnsw', space=distance)
        self.pool = []
        self.distance = Metrics.get_metric(self.distances[distance])
        self.deleted_ids = []

    def add_vector(self, vector: numpy.ndarray):
        self.max_item_num += 1

        if self.is_built:
            self.pool.append([self.max_item_num, vector])
        else:
            self.index.addDataPoint(self.max_item_num, vector)

    def delete_vector(self, vector_id: int):
        self.deleted_ids.append(vector_id)

    def build_index(self, *args, **kwargs):
        self.index.createIndex({'post': 2})

    def rebuild_index(self, *args, **kwargs):
        pass

    def search(self, vector: numpy.ndarray, neighbors: int, include_distances: bool = False):
        pool_distances = [(x[0], self.distance(vector, x[1])) for x in self.pool]
        index_distances = list(zip(*self.index.knnQuery(vector, k=neighbors*2)))
        distances = [x for x in pool_distances + index_distances if x[0] not in self.deleted_ids]
        res = sorted(distances, key=lambda x: x[1])[:neighbors]
        if include_distances:
            return res
        else:
            return [x[0] for x in res]

    def search_multiple(self, vectors: numpy.ndarray, neighbors: int, include_distances: bool = False):
        pass


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
        self.distance = Metrics.get_metric(distance)
        self.deleted_ids = []

    def add_vector_to_index(self, vector: numpy.ndarray):
        self.index.add_item(self.max_item_num, vector)
        self.max_item_num += 1

    def add_vector_to_pool(self, vector: numpy.ndarray):
        self.max_item_num += 1
        self.pool.append([self.max_item_num, vector])

    def add_vector(self, vector: numpy.ndarray):
        if self.is_built:
            self.add_vector_to_pool(vector)
        else:
            self.add_vector_to_index(vector)

    def delete_vector(self, vector_id: int):
        self.deleted_ids.append(vector_id)

    def build_index(self, trees: int = 10):
        self.index.build(trees)
        self.is_built = True

    def rebuild_index(self, trees: int = 10):
        self.index.build(trees)
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

    def search_multiple(self, vectors: numpy.ndarray, neighbors: int, include_distances: bool = False):
        pass


if __name__ == '__main__':
    """
    Testing MVP implementation.
    """
    vectors = get_random_vectors(10000, 30, rank=10000, save=True)
    #
    # w = MVPAnnoyWrapper(30, 'hamming')
    # for v in vectors:
    #     w.add_vector_to_index(v)
    #
    # w.build_index()
    #
    # for i, v in enumerate(vectors):
    #     w.add_vector_to_pool(v)
    #     if i % 10 == 0:
    #         w.delete_vector(i)
    #
    # for i in range(10):
    #     t = datetime.datetime.now()
    #     print(w.search(vectors[i], 100))
    #     print(datetime.datetime.now() - t)
    w = MVPNMSLIBWrapper(None, 'cosinesimil')
    for v in vectors:
        w.add_vector(v)
    w.build_index()
    t = datetime.datetime.now()
    print(w.search(vectors[0], 10))
    print(datetime.datetime.now() - t)

    for i, v in enumerate(vectors):
        if i % 10 == 0:
            w.add_vector(v)
        else:
            w.delete_vector(i)

    t = datetime.datetime.now()
    print(w.search(vectors[0], 10))
    print(datetime.datetime.now() - t)