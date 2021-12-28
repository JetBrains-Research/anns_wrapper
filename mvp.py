import datetime

import numpy
from annoy import AnnoyIndex

from helpers import get_random_vectors


def euclidean(v1: numpy.ndarray, v2: numpy.ndarray):
    return numpy.linalg.norm(v1 - v2)


def cosine(v1: numpy.ndarray, v2: numpy.ndarray):
    return numpy.dot(v1, v2) / (numpy.linalg.norm(v1) * numpy.linalg.norm(v2))


def hamming(v1: numpy.ndarray, v2: numpy.ndarray):
    return numpy.count_nonzero(v1 != v2)


# Global dictionary containing correlation between metric space and it's distance function.
distances = {
    "euclidean": euclidean,
    "angular": cosine,
    "hamming": hamming,
}


class MVP_annoy_Wrapper:
    """
    Bruteforce realisation.
    """
    index = None
    is_built = False
    max_item_num = 0

    def __init__(self, dimensions: int, distance: str):
        self.index = AnnoyIndex(dimensions, distance)
        self.pool = []
        self.distance = distances[distance]
        self.deleted_ids = []

    def add_vectors_to_index(self, vectors: numpy.ndarray):
        for i, v in enumerate(vectors):
            self.index.add_item(i, v)
        self.max_item_num = self.index.get_n_items()

    def add_vector_to_pool(self, vector: numpy.ndarray):
        self.max_item_num += 1
        self.pool.append([self.max_item_num, vector])

    def delete_vector(self, vector_id: int):
        self.max_item_num += 1
        self.deleted_ids.append(vector_id)

    def build_index(self, trees: int = 10):
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


if __name__ == '__main__':
    """
    Testing MVP implementation.
    """
    vectors = get_random_vectors(10000, 30, rank=10000, save=True)

    w = MVP_annoy_Wrapper(30, 'hamming')
    w.add_vectors_to_index(vectors)

    w.build_index()

    for i, v in enumerate(vectors):
        w.add_vector_to_pool(v)
        if i % 10 == 0:
            w.delete_vector(i)

    for i in range(10):
        t = datetime.datetime.now()
        print(w.search(vectors[i], 100))
        print(datetime.datetime.now() - t)
