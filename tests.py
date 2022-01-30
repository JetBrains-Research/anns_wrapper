import unittest

from helpers import get_random_vectors
from mvp import MVPAnnoyWrapper


class AnnoyMVPTestMethods(unittest.TestCase):

    def add_vectors(self):
        for v in self.dataset:
            self.wrapper.add_vector(v)

    def setUp(self) -> None:
        self.dimensions = 3
        self.vectors_num = 10
        self.trees = 10
        self.distance = 'euclidean'
        self.dataset = get_random_vectors(self.vectors_num, self.dimensions, rank=10000, save=True)
        self.wrapper = MVPAnnoyWrapper(self.dimensions, self.distance)

    def test_add_vectors_to_index(self):
        """
        Checks if vectors are being added to index if it is not built.
        """
        self.add_vectors()
        self.test_build_index()

        self.assertEqual(self.wrapper.index.get_n_items(), len(self.dataset))

    def test_add_vectors_to_pool(self):
        """
        Checks if vectors are being added to pool if index is built.
        """
        self.test_build_index()
        self.add_vectors()
        self.assertTrue(self.wrapper.index.get_n_items() == 0 and len(self.wrapper.pool) == len(self.dataset))

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
        self.wrapper.delete_vector(deleted_vector_id)
        self.assertEqual(len(self.wrapper.deleted_ids), 0)
        self.assertTrue(not any([id == deleted_vector_id for id, _ in self.wrapper.pool]))

    def test_build_index(self):
        """
        Checks index building.
        """
        self.wrapper.build_index(self.trees)

    def test_rebuild_index(self):
        """
        Checks index rebuilding.
        """
        self.add_vectors()
        self.test_build_index()
        self.add_vectors()

        for i in range(len(self.dataset) // 2):
            self.wrapper.delete_vector(i)

        self.wrapper.rebuild_index(self.trees)

        self.assertEqual(len(self.wrapper.pool), 0)
        self.assertEqual(len(self.wrapper.deleted_ids), 0)
        self.assertEqual(self.wrapper.index.get_n_items(), len(self.dataset) * 2 - len(self.dataset) // 2)

    def test_search_index(self):
        from annoy import AnnoyIndex
        self.test_add_vectors_to_index()
        self.add_vectors()

        index = AnnoyIndex(self.dimensions, self.wrapper.distance_name)
        for i, v in enumerate(self.dataset):
            index.add_item(i, v)
        for i, v in enumerate(self.dataset):
            index.add_item(len(self.dataset) + i, v)
        index.build(self.trees)

        confirmed_neighbours = index.get_nns_by_vector(self.wrapper.pool[0][1], len(self.wrapper.pool) // 2)
        mvp_neighbours = self.wrapper.search(self.wrapper.pool[0][1], len(self.wrapper.pool) // 2,
                                             include_distances=False)

        precision = sum(
            [any([int(mvp_neighbour == neighbour) for neighbour in confirmed_neighbours]) for mvp_neighbour in
             mvp_neighbours]) / len(mvp_neighbours)
        self.assertEqual(len(mvp_neighbours), len(self.wrapper.pool) // 2)
        self.assertTrue(precision >= 0.95)