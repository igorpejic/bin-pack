import unittest
from data_generator import DataGenerator


class TestDataGenerator(unittest.TestCase):

    def test_gen_instance(self):

        n = 20
        w = 40
        h = 40
        dg = DataGenerator()
        some_instance = dg.gen_instance(n, w, h)
        self.assertEqual(len(some_instance), n)
        for _bin in some_instance:
            self.assertEqual(len(_bin), 3)
            self.assertTrue(_bin[0] <= w)
            self.assertTrue(_bin[1] <= h)

        # dg.visualize_2D(some_instance, w, h)


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
