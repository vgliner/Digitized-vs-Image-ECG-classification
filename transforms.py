import numpy as np
import torch
from torch.utils.data import Dataset


class ECG_tuple_transform(object):
    """
    A transform that returns a new view of a tensor.
    """

    def __init__(self, *view_dims):
        self.view_dims = view_dims

    def __call__(self, ECG_tuple):
        if self.view_dims[0] == -1:
            np.shape(ECG_tuple[0][0])
            the_object = ECG_tuple[0][0]
            the_object = np.ndarray.flatten(the_object)
            a = np.squeeze(ECG_tuple[0][1])
            the_object = np.concatenate((the_object, a), axis=0)
            # the_object=np.expand_dims(the_object,axis=0)
            # the_object=np.expand_dims(the_object,axis=0)
            reshaped_vector = (the_object, ECG_tuple[1])

        else:
            reshaped_vector = ECG_tuple
        return reshaped_vector


class ECG_rendering_transform(object):
    """
    A transform that returns a new view of a tensor.
    """

    def __init__(self, *view_dims):
        self.view_dims = view_dims

    def __call__(self, ECG_tuple):
        if self.view_dims[0] == -1:
            the_object = ECG_tuple[0]
            the_object = np.ndarray.flatten(the_object)
            reshaped_vector = np.squeeze(the_object)
            reshaped_vector = (reshaped_vector, ECG_tuple[1])
        else:
            reshaped_vector = ECG_tuple
        return reshaped_vector


class SubsetDataset(Dataset):
    """
    A dataset that wraps another dataset, returning a subset from it.
    """

    def __init__(self, source_dataset: Dataset, subset_len, offset=0):
        """
        Create a SubsetDataset from another dataset.
        :param source_dataset: The dataset to take samples from.
        :param subset_len: The total number of sample in the subset.
        :param offset: The offset index to start taking samples from.
        """
        if offset + subset_len > len(source_dataset):
            raise ValueError("Not enough samples in source dataset")

        self.source_dataset = source_dataset
        self.subset_len = subset_len
        self.offset = offset

    def __getitem__(self, index):
        # Return the item at index + offset from the source dataset.
        # Make sure to raise an IndexError if index is out of bounds.

        # Check that the index is not out of bounds
        if index >= self.subset_len:
            raise IndexError("Index is out of bound, can't access element {}".format(index) +
                             ' in a sequence of length {}.'.format(self.subset_len))

        # Load item in case index is valid
        return self.source_dataset[self.offset + index]

    def __len__(self):
        return self.subset_len
