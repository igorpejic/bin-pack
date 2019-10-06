import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import math
import bisect
from sklearn.decomposition import PCA

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
        reward = 0

        for _bin in bins:
            if not self.is_bin_outside_borders(_bin):
                old_lfb = self.LFBs[0]

                right_point = self._get_closest_right_point()

                self.LFBs.add((old_lfb[0], old_lfb[1] + _bin[1]))

                print(_bin, right_point, self.LFBs)
                self.LFBs.add((right_point[0], right_point[1] + _bin[1]))
                self.LFBs.pop(0)

            else:
                # remove LFB as the bin could not fill it
                print(_bin, self.LFBs, 'could not fill')
                pass

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
