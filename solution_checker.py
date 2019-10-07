import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import math
import bisect
from sklearn.decomposition import PCA
from data_generator import DataGenerator

from sortedcontainers import SortedKeyList


class SolutionChecker(object):

    def __init__(self, n, w, h):
        self.n = n
        self.w = w
        self.h = h
        self.LFBs = SortedKeyList([], key=lambda x: (x[1], x[0]))
        self.LFBs.add((0, 0))

    def get_reward(self, bins):
        '''
        perfect reward is w * h - 0
        '''
        reward = self.w * self.h

        bins_processed = []
        for _bin in bins:
            lfbs_to_add = []
            if self.is_bin_outside_borders(_bin):
                # TODO
                reward -= 1
                print(_bin, self.LFBs, 'could not fill')
            else:
                old_lfb = self.LFBs[0]

                left_point = old_lfb[0], old_lfb[1] + _bin[1]
                low_right_point = old_lfb[0] + _bin[0], old_lfb[1]
                high_right_point = old_lfb[0] + _bin[0], old_lfb[1] + _bin[1]

                if left_point[1] == self.h:  # reached the ceiling
                    print('reached the ceiling')
                    # self.LFBs.add(low_right_point)
                    # self.LFBs.add(high_right_point)
                    lfbs_to_add.extend([low_right_point])
                else:
                    lfbs_to_add.extend([left_point, low_right_point, high_right_point])
                    #self.LFBs.add(left_point)

                    #self.LFBs.add(low_right_point)
                    #self.LFBs.add(high_right_point)

            # TODO: now check which lfbs points are covered by the new edge
            elements_to_remove = []
            for _lfb in self.LFBs:
                if left_point[1] < _lfb[1]:  # if the new edge is not higher it does not cover it for sure
                    continue

                overlaps = left_point[0] < _lfb[0] and high_right_point[0] > _lfb[0]
                left_edge_equal = left_point[0] == _lfb[0] and high_right_point[0] >= _lfb[0]
                right_edge_equal = (left_point[0] < _lfb[0] and high_right_point[0] >= _lfb[0])
                if (  # covering condition
                        overlaps or left_edge_equal or right_edge_equal
                ):
                    elements_to_remove.append(_lfb)
                
            print(elements_to_remove)
            for element in elements_to_remove:
                self.LFBs.remove(element)

            for element in lfbs_to_add:
                self.LFBs.add(element)

            bins_processed.append(_bin)
            DataGenerator().visualize_2D(bins_processed, self.w, self.h, extreme_points=self.LFBs)

        return reward

    def is_bin_outside_borders(self, _bin):
        position_to_put_bin_into = self.LFBs[0]
        left_border = self.LFBs[0][0]

        closest_right = self._get_closest_right()

        if left_border +  _bin[0] > closest_right:  # clashes with box on right or total box border
            return True

        if self.LFBs[0][1] + _bin[1] > self.h:  # taller than total box
            return True

        return False


    def _get_closest_right_point(self):
        left_border = self.LFBs[0][0]
        right_border = None

        closest_right = sorted((x for x in self.LFBs if x[0] > left_border), key=lambda x: x[1])
        if not closest_right:
            closest_right = (self.w, 0)
        else:
            closest_right = closest_right[0]
        return closest_right

    def _get_closest_right(self):
        return self._get_closest_right_point()[0]


    def visualize_extreme_points(self):

        import matplotlib
        matplotlib.use('GTK')
        np.random.seed(4)

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, aspect='equal')
        ax1.set_xticks(list(range(self.w)))
        ax1.set_yticks(list(range(self.h)))
        ax1.grid(which='both')
        x = [x[0] for x in self.LFBs]
        y = [x[1] for x in self.LFBs]
        plt.scatter(x, y)

        plt.xlim(0, self.w)
        plt.ylim(0, self.h)
        plt.figure(1)
        plt.show()
        #plt.pause(0.2)
        plt.close()
