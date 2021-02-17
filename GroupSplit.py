"""
This module contains classes and functions
to split data to groups
"""

# Author: Dimitrios Karatasios <dimkarata@gmail.com>
#
# License: BSD 3-Clause License

# Module Imports
import numpy as np
import math


class GroupSplit:
    """
    Splits data to training/test set following the provided split method
    """
    def __init__(self, df, shuffle=True, random_state=None):
        self.size = df.shape[0]
        self.index = list(df.index)
        self.shuffle = shuffle
        self.random_state = random_state

        # shuffle the data if shuffle is True
        if self.shuffle:
            random_state = np.random.RandomState(self.random_state)
            random_state.shuffle(self.index)

    def split(self, test_frac=0.2, test_size=None):
        """
        Splits data into train/test set based on the inputs

        :param test_frac: float [0, 1], fraction of the total data to be used as test set
        :param test_size: integer, if specified test_frac is ignored and the provided value for test size is used

        :return: tuple of lists, containing the indices of train and test sets
        """

        if test_size is not None:
            if test_size >= self.size:
                raise ValueError(f"The test size must be smaller than the size of the data\n"
                                 f"Currently: test_size:{test_size} and data_size:{self.size}")

        return self._train_test_split(self.index, test_frac, test_size)

    @staticmethod
    def _train_test_split(a, test_frac, test_size=None):
        """
        splits an array to train and test size depending on the parameters specified
        :param a: array type containing indices
        :param test_frac: float [0, 1], fraction of the total data to be used as test set
        :param test_size: integer, if specified test_frac is ignored and the provided value for test size is used
        :return:
        """
        if test_size is None:
            test_size = math.ceil(len(a) * test_frac)
        train_size = len(a) - test_size

        train_idx = [y for x, y in np.ndenumerate(a) if x[0] < train_size]
        test_idx = [y for x, y in np.ndenumerate(a[train_size::])]

        return train_idx, test_idx

    def group_split(self, number_of_groups, overlap=None, test_set=True, test_frac=0.2):
        """
        Generates subsets of the data depending on the parameters

        :param number_of_groups: integer, number of subsets to create, must be at least 2
        :param overlap: float [0, 1], the percent each group overlaps the previous one
        :param test_set: bool, if False no test set in each subset is generated
        :param test_frac: float [0, 1], fraction of initial data values to be included in the test set

        :return: generator, yields train, test set groups if test_set is True. Else only one set.
        """

        if number_of_groups < 2:
            raise ValueError("Number of groups to split the data must be at least 2")
        elif number_of_groups > self.size:
            raise ValueError(f"Number of groups too large: {number_of_groups}\n"
                             f"It should be less than the size of the data: {self.size} ")
        group_size = self.size // number_of_groups
        remainder = self.size % number_of_groups
        groups = {group: group_size + 1 if group < remainder else group_size for group in range(number_of_groups)}

        if overlap is not None:
            if overlap > 1:
                raise ValueError('Overlap cannot be larger than 1 --> Permitted values: None or [0, 1]')

        indices = self._rolling_indices(groups, overlap)
        for start, end in indices:
            if not test_set:
                yield self.index[start:end]
            else:
                yield self._train_test_split(self.index[start:end], test_frac)

    @staticmethod
    def _rolling_indices(groups, overlap=None):
        start = 0
        end = -1
        if overlap is None:
            overlap = 0
        for group, group_size in groups.items():
            end += group_size
            yield int(start), int(end)
            start = end - overlap * end
