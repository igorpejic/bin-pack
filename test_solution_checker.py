import unittest
import numpy as np
from data_generator import DataGenerator
from solution_checker import SolutionChecker

from sortedcontainers import SortedKeyList

class TestDataGenerator(unittest.TestCase):

    # def test_gen_instance_visual(self):

    #     n = 20
    #     w = 40
    #     h = 40
    #     dg = DataGenerator()
    #     some_instance = dg.gen_instance_visual(n, w, h)
    #     self.assertEqual(len(some_instance), n)
    #     for _bin in some_instance:
    #         self.assertEqual(len(_bin), 3)
    #         self.assertTrue(_bin[0] <= w)
    #         self.assertTrue(_bin[1] <= h)

    #     # dg.visualize_2D(some_instance, w, h)


    def test_check_perfect_solution(self):
        n = 20
        w = 40
        h = 40
        dg = DataGenerator()
        some_instance_visual = dg.gen_instance_visual(n, w, h)
        perfect_bin_configuration = sorted(some_instance_visual, key=lambda x: (x[2][1], x[2][0]))
        some_instance_np_array = dg._transform_instance_visual_to_np_array(some_instance_visual)

        solution_checker = SolutionChecker(n, h, w)
        self.assertEqual(
            solution_checker.get_reward(perfect_bin_configuration),
            w * h
        )

    def test_bin_outside_border(self):

        n = 20
        h = 50
        w = 50

        solution_checker = SolutionChecker(n, h, w)
        #
        # 11  -------------
        #     |           |
        #     |           |          |
        #     |           |          |
        #     -----------------------|
        #                40

        solution_checker.LFBs = SortedKeyList([], key=lambda x: (x[1], x[0]))
        solution_checker.LFBs.add((40, 0))
        solution_checker.LFBs.add((0, 11))
        solution_checker.LFBs.add((40, 11))

        _bin = (10, 10)
        self.assertFalse(solution_checker.is_bin_outside_borders(_bin))

        _bin = (12, 10)
        self.assertTrue(solution_checker.is_bin_outside_borders(_bin))
