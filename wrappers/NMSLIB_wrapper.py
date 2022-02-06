import nmslib
import numpy

from interface import NNSWrapper
from utility.metrics import Metrics


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
        index_distances = list(zip(*self.index.knnQuery(vector, k=neighbors * 2)))
        distances = [x for x in pool_distances + index_distances if x[0] not in self.deleted_ids]
        res = sorted(distances, key=lambda x: x[1])[:neighbors]
        if include_distances:
            return res
        else:
            return [x[0] for x in res]
