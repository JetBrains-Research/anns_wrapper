from typing import Callable

import numpy


class Metrics:
    @staticmethod
    def euclidean(v1: numpy.ndarray, v2: numpy.ndarray):
        return numpy.linalg.norm(v1 - v2)

    @staticmethod
    def cosine(v1: numpy.ndarray, v2: numpy.ndarray):
        return numpy.dot(v1, v2) / (numpy.linalg.norm(v1) * numpy.linalg.norm(v2))

    @staticmethod
    def hamming(v1: numpy.ndarray, v2: numpy.ndarray):
        return numpy.count_nonzero(v1 != v2)

    @classmethod
    def get_metric(cls, metric_name: str) -> Callable:
        return getattr(cls, metric_name)


