from abc import ABC, abstractmethod

import numpy


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
