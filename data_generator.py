import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import math

class DataGenerator(object):

    def __init__(self):
        pass

    def gen_instance(self, n, w, h, seed=0): # Generate random bin-packing instance
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

    def train_batch(self, batch_size, n, w, h, seed=0):
        input_batch = []
        for _ in range(batch_size):
            input_ = self.gen_instance(n, w, h, seed=seed)
            input_batch.append(input_)
        return input_batch


    def test_batch(self, batch_size, n, w, h, seed=0, shuffle=False): # Generate random batch for testing procedure
        input_batch = []
        input_ = self.gen_instance(n, w, h, seed=seed)
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

    def visualize_2D(self, bins, w, h): # Plot tour

        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib
        matplotlib.use('GTK')
        import random
        np.random.seed(4)
        from itertools import cycle
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

        ax1.set_xticks(list(range(w)))
        ax1.set_yticks(list(range(h)))
        ax1.grid(which='both')

        plt.xlim(0, w)
        plt.ylim(0, h)
        plt.figure(1)
        plt.show()
