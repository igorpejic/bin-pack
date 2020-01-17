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
            0
        )

    def test_check_imperfect_solution(self):
        n = 4
        cols = 10
        rows = 5
        dg = DataGenerator()
        some_instance_np_array = np.array(
            [[10, 1, 1], [10, 2, 2], [10, 1, 3], [10, 5, 4], [10, 1, 1]])

        solution_checker = SolutionChecker(n, cols, rows)
        self.assertEqual(
            solution_checker.get_reward(some_instance_np_array),
            (10 * 5) / (cols * rows)
        )

    def test_check_imperfect_solution_2(self):
        n = 4
        cols = 10
        rows = 5
        dg = DataGenerator()
        some_instance_np_array = np.array(
            [[10, 1, 1], [10, 6, 2], [10, 2, 3], [10, 1, 4], [10, 1, 1]])

        solution_checker = SolutionChecker(n, cols, rows)
        self.assertEqual(
            solution_checker.get_reward(some_instance_np_array),
            (10 * 6) / (cols * rows)
        )
        
    def test_check_imperfect_solution_count_files_2(self):
        n = 4
        cols = 10
        rows = 5
        dg = DataGenerator()
        some_instance_np_array = np.array(
            [[10, 1, 1], [10, 6, 2], [10, 2, 3], [10, 1, 4], [10, 1, 1]])

        solution_checker = SolutionChecker(n, cols, rows)
        self.assertEqual(
            solution_checker.get_reward(some_instance_np_array, count_tiles=True),
            1 / n
        )

    def test_check_imperfect_solution_count_tiles(self):
        n = 4
        cols = 10
        rows = 5
        dg = DataGenerator()
        some_instance_visual = dg.gen_instance_visual(n, cols, rows)
        # NOTE: first bin always repeated
        some_instance_np_array = np.array(
            [[10, 1, 1], [10, 2, 2], [10, 1, 3], [10, 5, 4], [10, 1, 1]])

        solution_checker = SolutionChecker(n, cols, rows)
        self.assertEqual(
            solution_checker.get_reward(some_instance_np_array, count_tiles=True),
            1 / n
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
        solution_checker.LFBs.add((40, 11))

        _bin = (10, 10)
        self.assertFalse(solution_checker.is_bin_outside_borders(_bin))

        _bin = (12, 10)
        self.assertTrue(solution_checker.is_bin_outside_borders(_bin))

    def test_get_next_lfb_on_grid(self):
        state = np.array([[1, 1], [0, 0]])
        res = SolutionChecker.get_next_lfb_on_grid(state)
        self.assertEqual(res, (0, 1)) 

    def test_get_next_lfb_on_grid_2(self):
        state = np.array([[1, 1, 0], [0, 0, 0]])
        res = SolutionChecker.get_next_lfb_on_grid(state)
        self.assertEqual(res, (2, 0)) 

    def test_get_next_lfb_on_grid_2(self):
        state = np.array([[1, 1, 1], [1, 1, 1]])
        res = SolutionChecker.get_next_lfb_on_grid(state)
        self.assertIsNone(res) 



if __name__=='__main__':
    unittest.main()
