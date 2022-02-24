import unittest

import numpy
import pyflann
from pyflann import FLANN

from utility.helpers import get_random_vectors
from wrappers.flann import FlannWrapper


class FlannMVPTestMethods(unittest.TestCase):

    def add_vectors(self):
        for v in self.dataset:
            self.wrapper.add_vector(v)

    def setUp(self) -> None:
        self.dimensions = 3
        self.vectors_num = 50
        self.trees = 10
        self.distance = 'euclidean'
        self.dataset = get_random_vectors(self.vectors_num, self.dimensions, rank=10000, save=True)
        self.wrapper = FlannWrapper(self.dimensions, self.distance)

    def test_add_vectors_to_index(self):
        """
        Checks if vectors are being added to index if it is not built.
        """
        self.test_build_index()

        self.assertEqual(len(self.wrapper.index.__dict__['_FLANN__curindex_data']), len(self.dataset))

    def test_add_vectors_to_pool(self):
        """
        Checks if vectors are being added to pool if index is built.
        """
        self.test_build_index()
        self.add_vectors()
        self.assertTrue(len(self.wrapper.index.__dict__['_FLANN__curindex_data']) == len(self.dataset) and len(
            self.wrapper.pool) == len(self.dataset))

    def test_delete_vectors_from_index(self):
        """
        Checks if vectors are being added to index if it is not built.
        """
        self.add_vectors()
        self.test_build_index()

        self.wrapper.delete_vector(0)

        self.assertTrue(0 in self.wrapper.deleted_ids)

    def test_delete_vectors_from_pool(self):
        """
        Checks if vectors are being added to pool if index is built.
        """
        self.test_build_index()
        self.add_vectors()
        deleted_vector_id = self.wrapper.pool[0][0]
        print(self.wrapper.pool[0])
        self.wrapper.delete_vector(deleted_vector_id)
        self.assertEqual(len(self.wrapper.deleted_ids), 0)
        self.assertTrue(not any([id == deleted_vector_id for id, _ in self.wrapper.pool]))

    def test_build_index(self):
        """
        Checks index building.
        """

        self.add_vectors()
        self.wrapper.build_index(self.trees)

    def test_rebuild_index(self):
        """
        Checks index rebuilding.
        """
        self.test_build_index()
        self.add_vectors()

        for i in range(len(self.dataset) // 2):
            self.wrapper.delete_vector(i)

        self.wrapper.rebuild_index(self.trees)

        self.assertEqual(len(self.wrapper.pool), 0)
        self.assertEqual(len(self.wrapper.deleted_ids), 0)
        self.assertEqual(self.wrapper.max_item_num - 1, len(self.dataset) * 2 - len(self.dataset) // 2)

    def test_search_index(self):
        self.test_add_vectors_to_index()

        pyflann.index.set_distance_type(self.distance)
        index = FLANN()
        params = index.build_index(numpy.array(self.dataset))

        confirmed_neighbours, distances = index.nn_index(numpy.array(self.dataset[0]), 10, checks=params["checks"])
        print(confirmed_neighbours)
        mvp_neighbours = self.wrapper.search(self.dataset[0], 10, include_distances=False)
        print(mvp_neighbours)

        precision = sum(
            [any([int(mvp_neighbour == neighbour) for neighbour in confirmed_neighbours[0]]) for mvp_neighbour in
             mvp_neighbours]) / len(mvp_neighbours)
        self.assertEqual(len(mvp_neighbours), 10)
        self.assertGreaterEqual(precision, 0.95)
