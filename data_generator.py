import matplotlib
from itertools import cycle
import random
import numpy as np
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA

ORIENTATIONS = 2
class DataGenerator(object):

    def __init__(self, w=None, h=None):
        self.w = w
        self.h = h
        self.frozen_first_batch = None

    def gen_instance_visual(self, n, w, h, dimensions=2, seed=0): # Generate random bin-packing instance
        self.w = w
        self.h = h
        if seed!=0:
            np.random.seed(seed)
        bins = [[w, h, (0, 0)]]
        while len(bins) <  n:
            random_bin_index = np.random.randint(0, len(bins), size=1)[0]
            bin_to_split = bins[random_bin_index]

            axis_to_split = np.random.randint(0, 2, size=1)[0]
            
            if bin_to_split[axis_to_split] <= 1:
                # cant split anymore; this is minimum size
                continue

            bins.pop(random_bin_index)

            split_val = int(np.random.randint(1, bin_to_split[axis_to_split], size=1)[0])
            new_bins = self._split_bin(bin_to_split, axis_to_split, split_val)

            bins.insert(random_bin_index, new_bins[0])
            bins.insert(random_bin_index, new_bins[1])
        return bins

    def _transform_instance_visual_to_np_array(self, bins, dimensions=2):
        return np.array([x[:dimensions] for x in bins])

    def gen_instance(self, n, w, h, dimensions=2, seed=0): # Generate random bin-packing instance
        bins = self._transform_instance_visual_to_np_array(self.gen_instance_visual(n, w, h, seed=seed), dimensions=dimensions)
        return np.array(bins)

    def gen_matrix_instance(self, n, w, h, dimensions=2, seed=0):
        return self._transform_instance_to_matrix(self.gen_instance(
            n, w, h, dimensions=dimensions, seed=seed))

    @staticmethod
    def tile_to_matrix(tile, w, h):
        _slice = np.zeros([h, w])
        for i in range(tile[0]):
            for j in range(tile[1]):
                _slice[i][j] = 1
        return _slice



    def _transform_instance_to_matrix(self, tiles):
        """
        transforms list of bins:
        [[2, 3], [4, 5]]
        to stacks of bins with 2 orientations in left bottom corner
        of size (w, h):
        0000000000000
        0000000000000
        1100000000000
        1100000000000
        1100000000000
        """
        h = self.h
        w = self.w

        all_slices = None
        for tile in tiles:
            for orientation in range(ORIENTATIONS):
                if orientation == 0:
                    _slice = self.tile_to_matrix(tile, w, h)
                else:
                    _slice = self.tile_to_matrix((tile[1], tile[0]), w, h)
                if all_slices is not None:
                    _slice = np.reshape(_slice, (1, _slice.shape[0], _slice.shape[1]))
                    all_slices = np.concatenate((all_slices, _slice), axis=0)
                else:
                    all_slices = _slice
                    all_slices = np.reshape(all_slices, (1, _slice.shape[0], _slice.shape[1]))

        return all_slices


    @staticmethod
    def get_matrix_tile_dims(tile):
        matrix_w, matrix_h = tile.shape
        height = 0
        width = 0
        i = 0
        while tile[0][i] == 1: 
            height += 1
            i += 1
            if i >= matrix_h:
                break

        i = 0
        while tile[i][0] == 1: 
            width += 1
            i += 1
            if i >= matrix_w:
                break

        return (width, height)

    @staticmethod
    def get_valid_moves_mask(state, tiles):
        """
        state - 2d matrix representing current tile state
        tiles - list of 2 matrices with same size as state each presenting one orientation
        """
        height = state.shape[0]
        width = state.shape[1]

        for i, tile in enumerate(tiles):
            mask = state == 0
            for row in range(height):
                for col in range(width):
                    # no need to check this one as position is already taken
                    if mask[row][col] == False: 
                        continue

                    # checks if it clashes with already existing tiles
                    try:
                        DataGenerator.add_tile_to_state(
                            state, tile, (row, col))
                    except ValueError:
                        mask[row][col] = False

            if i == 0:
                first_mask = np.copy(mask)
            else:
                second_mask = np.copy(mask)

        final_mask = np.concatenate((first_mask, second_mask), axis=0)
        return final_mask

    @staticmethod
    def position_index_to_row_col(position, width, height):
        return (position // width, position % width)

    @staticmethod
    def add_tile_to_state(state, tile, position):
        new_state = np.copy(state)
        tile_w, tile_h = DataGenerator.get_matrix_tile_dims(tile)
        for col in range(tile_w):
            for row in range(tile_h):
                if position[0] + row >= state.shape[0]:
                    raise ValueError(
                        f'tile goes out of bin height {position}')
                if position[1] + col >= state.shape[1]:
                    raise ValueError(
                        f'tile goes out of bin width {position}')

                if new_state[position[0] + row ][position[1] + col] == 1:
                    raise ValueError('locus already taken')
                else:
                    new_state[position[0] + row ][position[1] + col] = 1

        return new_state

    def _split_bin(self, _bin, axis, value):
        assert len(_bin) == 3
        assert type(_bin[0]) == int, type(_bin[0])
        assert type(_bin[1]) == int, type(_bin[1])
        assert type(_bin[2]) == tuple, type(_bin[2])

        if axis == 0:
            ret = [ [value, _bin[1], _bin[2]], [_bin[0] - value, _bin[1], (_bin[2][0] + value, _bin[2][1])] ]
        elif axis == 1:
            ret = [ [_bin[0], value, _bin[2]], [_bin[0], _bin[1] - value, (_bin[2][0], _bin[2][1] + value)] ] 
        return ret

    def train_batch(self, batch_size, n, w, h, dimensions=2, seed=0, freeze_first_batch=False):
        input_batch = []
        if freeze_first_batch and self.frozen_first_batch:
            return self.frozen_first_batch
        for _ in range(batch_size):
            input_ = self.gen_instance(n, w, h, dimensions=dimensions, seed=seed)
            input_batch.append(input_)

        if freeze_first_batch:
            if not self.frozen_first_batch:
                self.frozen_first_batch = input_batch
                print('Using frozen batch', self.frozen_first_batch)
        return input_batch


    def test_batch(self, batch_size, n, w, h, dimensions=2, seed=0, shuffle=False): # Generate random batch for testing procedure
        input_batch = []
        input_ = self.gen_instance(n, w, h, dimensions=dimensions, seed=seed)
        for _ in range(batch_size): 
            sequence = np.copy(input_)
            if shuffle==True: 
                np.random.shuffle(sequence) # Shuffle sequence
            input_batch.append(sequence) # Store batch
        return input_batch

    def get_cmap(self, n, name='Accent'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name)

    def visualize_2D(self, bins, w, h, extreme_points=None): # Plot tour

        matplotlib.use('GTK')
        np.random.seed(4)
        cycol = cycle('bgrcmk')

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, aspect='equal')
        patterns = ('-', '+', 'x', '\\', '*', 'o', 'O', '.')
        cmap = self.get_cmap(len(bins))
        for i, _bin in enumerate(bins):
            color = np.random.rand(3,)
            ax1.add_patch(
                patches.Rectangle(
                    (_bin[2][0], _bin[2][1]), _bin[0], _bin[1],
                    # color=cmap(i),
                    color=color,
                    edgecolor=color,
                    hatch=patterns[random.randint(0, len(patterns) - 1)])
            )
            ax1.text(_bin[2][0] + _bin[0] / 2 - 2 , _bin[2][1] + _bin[1] / 2, str(_bin))

        if extreme_points:
            x = [x[0] for x in extreme_points]
            y = [x[1] for x in extreme_points]
            plt.scatter(x, y, s=500)

        ax1.set_xticks(list(range(w)))
        ax1.set_yticks(list(range(h)))
        ax1.grid(which='both')

        plt.xlim(0, w)
        plt.ylim(0, h)
        plt.figure(1)
        plt.show()
        #plt.pause(0.2)
        #plt.close()
