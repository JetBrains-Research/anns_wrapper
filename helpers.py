import os.path

import numpy as np


def generate_random_vectors(num: int, dimensions: int, rank: int = 1000):
    return np.random.random([num, dimensions]) * rank


def get_random_vectors(num: int, dimensions: int, rank: int = 1000, save: bool = True):
    """
    load or generate random vectors. Generated vectors can be saved if wanted.
    """
    filename = f"./datasets/{num}_{dimensions}_{rank}_vectors.csv"

    if os.path.isfile(filename):
        vectors = np.loadtxt(filename)
    else:
        vectors = generate_random_vectors(num, dimensions, rank)
        if save:
            np.savetxt(filename, vectors)
    return vectors
