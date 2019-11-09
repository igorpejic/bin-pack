import numpy as np
import matplotlib.pyplot as plt
import math
import bisect
from sklearn.decomposition import PCA
from data_generator import DataGenerator

from sortedcontainers import SortedKeyList


class SolutionChecker(object):


    def __init__(self, n, cols, rows):
        self.n = n
        self.cols = cols
        self.rows = rows
        self.LFBs = SortedKeyList([], key=lambda x: (x[1], x[0]))
        self.LFBs.add((0, 0))
        self.grid = self.initialize_grid()


    def initialize_grid(self):
        return [[0 for x in range(self.cols)] for y in range(self.rows)]


    def get_rewards(self, batch_bins, count_tiles=False, combinatorial_reward=False):
        batch_rewards = []
        # print(batch_bins)
        for  _bin in batch_bins:
            self.grid = self.initialize_grid()
            batch_rewards.append(self.get_reward(_bin, count_tiles=count_tiles, combinatorial_reward=combinatorial_reward))
        # return np.mean(batch_rewards).astype(np.float32)
        return np.array(batch_rewards).astype(np.float32)

    def get_reward(self, bins, count_tiles=False, combinatorial_reward=False):
        '''
        perfect reward is 0 area wasted
        as_tiles_non_placed - reward is given by number of tiles non_placed from the total number of tiles
        combinatorial_reward - if True it will stop after first bin is not placed
        '''

        reward = 0
        bins = bins[:-1]
        for i, _bin in enumerate(bins):
            next_lfb = self.get_next_lfb()
            if not next_lfb:
                break

            placed = self.place_element_on_grid(_bin, next_lfb, i + 1)
            if not placed:
                if combinatorial_reward:
                    return 1
                if count_tiles:
                    reward += 1
                else:
                    reward += (_bin[0] * _bin[1])

        # scale from 0 to 1
        if reward == 0:
            return 0
        else:
            if count_tiles:
                return reward / self.n
            else:
                return reward / (self.cols * self.rows)

    def get_next_lfb(self):
        lfb = None
        for i, _ in enumerate(self.grid):
            for j, _val in enumerate(self.grid[i]):
                if self.grid[i][j] == 0:
                    return (j, i)
        return lfb

    def place_element_on_grid(self, _bin, position, val):
        if position[0] + _bin[0] > self.cols:
            # print(f'{position[0] + _bin[0]} bigger than width')
            return False
        if position[1] + _bin[1] > self.rows:
            # print(f'{position[1] + _bin[1]} bigger than height')
            return False

        for i in range(int(_bin[1])):
            for j in range(int(_bin[0])):
                row = self.grid[position[1] + i]
                if row[position[0] + j] != 0:
                    # print(f'position ({position[1] + i} {position[0] + j}) already taken')
                    return False
                row[position[0] + j] = val

        return True

    # def get_reward(self, bins):
    #     '''
    #     perfect reward is w * h - 0
    #     '''
    #     reward = self.w * self.rows

    #     bins_processed = []
    #     for _bin in bins:
    #         lfbs_to_add = []
    #         if self.is_bin_outside_borders(_bin):
    #             # TODO
    #             reward -= 1
    #             print(_bin, self.LFBs, 'could not fill')
    #         else:
    #             old_lfb = self.LFBs[0]

    #             left_point = old_lfb[0], old_lfb[1] + _bin[1]
    #             low_right_point = old_lfb[0] + _bin[0], old_lfb[1]
    #             high_right_point = old_lfb[0] + _bin[0], old_lfb[1] + _bin[1]

    #             if left_point[1] == self.rows:  # reached the ceiling
    #                 print('reached the ceiling')
    #                 if high_right_point[0] != self.w:
    #                     lfbs_to_add.extend([low_right_point])
    #             elif high_right_point[0] == self.w:
    #                 lfbs_to_add.extend([left_point])

    #             else:
    #                 lfbs_to_add.extend([left_point, low_right_point, high_right_point])
    #                 #self.LFBs.add(left_point)

    #                 #self.LFBs.add(low_right_point)
    #                 #self.LFBs.add(high_right_point)

    #         # TODO: now check which lfbs points are covered by the new edge
    #         elements_to_remove = []
    #         for _lfb in self.LFBs:

    #             overlaps = left_point[0] < _lfb[0] and high_right_point[0] > _lfb[0]
    #             left_edge_equal = left_point[0] == _lfb[0] and high_right_point[0] >= _lfb[0]
    #             right_edge_equal = (left_point[0] < _lfb[0] and high_right_point[0] >= _lfb[0])
    #             if (  # covering condition
    #                     overlaps or left_edge_equal or right_edge_equal
    #             ):
    #                 elements_to_remove.append(_lfb)
    #             print(overlaps, left_edge_equal, right_edge_equal, _lfb)

    #             # the following removes parts on flat
    #             # lfb is neighbor on right; we need to remove high_right_point
    #             if high_right_point[1] == _lfb[1] and high_right_point[0] == _lfb[0]:
    #                 if high_right_point in lfbs_to_add:
    #                     lfbs_to_add.remove(high_right_point)
    #                     elements_to_remove(high_right_point)

    #             # lfb is neighbor on left
    #             if high_right_point[1] == _lfb[1] and left_point[0] == _lfb[0]:
    #                 if left_point in lfbs_to_add:
    #                     lfbs_to_add.remove(left_point)
    #                     elements_to_remove(left_point)
    #             
    #         for element in elements_to_remove:
    #             if element in self.LFBs:
    #                 self.LFBs.remove(element)

    #         for element in lfbs_to_add:
    #             self.LFBs.add(element)

    #         bins_processed.append(_bin)
    #         DataGenerator().visualize_2D(bins_processed, self.w, self.rows, extreme_points=self.LFBs)

    #     return reward

    def is_bin_outside_borders(self, _bin):
        position_to_put_bin_into = self.LFBs[0]
        left_border = self.LFBs[0][0]

        closest_right = self._get_closest_right()

        ret = False
        if left_border +  _bin[0] > closest_right:  # clashes with box on right or total box border
            ret = True

        if self.LFBs[0][1] + _bin[1] > self.rows:  # taller than total box
            ret = True

        if ret:
            print(f'bin {_bin} could not fit into {self.LFBs} (closest_right: {closest_right}')

        return ret


    def _get_closest_right_point(self):
        left_border = self.LFBs[0][0]
        right_border = None

        closest_right = sorted((x for x in self.LFBs if x[0] > left_border), key=lambda x: x[1])
        if not closest_right:
            closest_right = (self.cols, 0)
        else:
            closest_right = closest_right[0]
        return closest_right

    def _get_closest_right(self):
        return self._get_closest_right_point()[0]


    def visualize_grid(self):

        import matplotlib
        matplotlib.use('GTK')
        np.random.seed(4)

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, aspect='equal')
        ax1.set_xticks(list(range(self.cols)))
        ax1.set_yticks(list(range(self.rows)))
        ax1.imshow(self.grid)
        plt.xlim(0, self.cols)
        plt.ylim(0, self.rows)
        plt.figure(1)
        plt.show()
        #plt.pause(0.2)
        plt.close()
