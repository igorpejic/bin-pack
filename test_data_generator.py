import unittest
import numpy as np
from data_generator import DataGenerator


class TestDataGenerator(unittest.TestCase):

    def test_gen_instance_visual(self):

        n = 20
        w = 40
        h = 40
        dg = DataGenerator()
        some_instance = dg.gen_instance_visual(n, w, h)
        self.assertEqual(len(some_instance), n)
        for _bin in some_instance:
            self.assertEqual(len(_bin), 3)
            self.assertTrue(_bin[0] <= w)
            self.assertTrue(_bin[1] <= h)

        # dg.visualize_2D(some_instance, w, h)

    def test_gen_instance(self):

        n = 20
        w = 40
        h = 40
        dg = DataGenerator()
        some_instance = dg.gen_instance(n, w, h)
        self.assertEqual(len(some_instance), n)
        for _bin in some_instance:
            self.assertEqual(len(_bin), 2)
            self.assertTrue(_bin[0] <= w)
            self.assertTrue(_bin[1] <= h)
            self.assertEqual(type(some_instance), np.ndarray)

    def test_transform_instance_to_matrix(self):

        w = 40
        h = 30
        dg = DataGenerator(w, h)
        tiles = np.array([[2, 3], [4, 5]])
        matrix = dg._transform_instance_to_matrix(tiles)

        self.assertEqual(matrix[0].shape , (h, w))
        ORIENTATIONS = 2
        self.assertEqual(matrix.shape , (len(tiles) * ORIENTATIONS, h, w))
        print(matrix)

        # first tile different orientation
        self.assertEqual(matrix[0][0][0] , 1)
        self.assertEqual(matrix[0][0][1] , 1)
        self.assertEqual(matrix[0][0][2] , 1)
        self.assertEqual(matrix[0][1][0] , 1)
        self.assertEqual(matrix[0][1][1] , 1)
        self.assertEqual(matrix[0][1][2] , 1)

        self.assertEqual(matrix[0][2][0] , 0)
        self.assertEqual(matrix[0][2][1] , 0)
        self.assertEqual(matrix[0][2][2] , 0)
        self.assertEqual(matrix[0][1][3] , 0)

        # first tile first orientation
        self.assertEqual(matrix[1][0][0] , 1)
        self.assertEqual(matrix[1][0][1] , 1)
        self.assertEqual(matrix[1][1][0] , 1)
        self.assertEqual(matrix[1][1][1] , 1)
        self.assertEqual(matrix[1][2][0] , 1)
        self.assertEqual(matrix[1][2][1] , 1)

        self.assertEqual(matrix[1][0][2] , 0)
        self.assertEqual(matrix[1][1][2] , 0)
        self.assertEqual(matrix[1][0][3] , 0)




    def test_split_bin(self):
        dg = DataGenerator()
        _bin = [21, 41, (0, 0)]
        self.assertEqual(
            dg._split_bin(_bin, 0, 14),
            [[14, 41, (0, 0)], [21 - 14, 41, (14, 0)]]
        )
        self.assertEqual(
            dg._split_bin(_bin, 1, 14),
            [[21, 14, (0, 0)], [21, 41 - 14, (0, 14)]]
        )

    def test_train_batch(self):
        dg = DataGenerator()
        batch_size = 10
        n = 20
        batch = dg.train_batch(10, n, 40, 40)
        self.assertEqual(len(batch), 10)
        self.assertEqual(len(batch[0]), n)

    def test_test_batch(self):
        dg = DataGenerator()
        batch_size = 10
        n = 20
        batch = dg.test_batch(10, n, 40, 40)
        self.assertEqual(len(batch), 10)
        self.assertEqual(len(batch[0]), n)
