import os
import platform
import cv2
import numpy as np

import torch
from scipy.misc import imread
from multiprocessing import Process
from torch.utils.data import Dataset
from six.moves import cPickle as pickle

from utils import (
    showImage,
    rgb_to_bgr,
    bgr_to_rgb,
    bgr_to_ycrcb,
    multi_process_wrapper,
    rgb_to_ycrcb_channel_first,
)

global datasets


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def normalizes(data, subtract_mean=True, channel_first=True):
    X_train, X_test, X_val = data['X_train'], data['X_test'], data['X_val']
    # Normalize the data: subtract the mean image
    if subtract_mean:

        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

        # performan standard deviation and normalization
        # std_image = np.mean(X_train, axis=0)
        # X_train /= std_image
        # X_val /= std_image
        # X_test /= std_image

    # Transpose so that channels come first
    if channel_first:
        X_train = X_train.transpose(0, 3, 1, 2).copy()
        X_val = X_val.transpose(0, 3, 1, 2).copy()
        X_test = X_test.transpose(0, 3, 1, 2).copy()
    else:
        X_train = X_train.transpose(0, 1, 2, 3).copy()
        X_val = X_val.transpose(0, 1, 2, 3).copy()
        X_test = X_test.transpose(0, 1, 2, 3).copy()


def get_CIFAR10_data(
        num_training=49000,
        num_validation=1000,
        num_test=1000,
        subtract_mean=True,
        channel_first=True,
        visualizes=False):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = './datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]

    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    all_datas = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }

    if not visualizes:
        normalizes(all_datas, subtract_mean, channel_first)

    # Package data into a dictionary
    return all_datas


datasets = get_CIFAR10_data(visualizes=True)


class DataSets(Dataset):
    def __init__(self, dense=False, dataset='train'):
        super().__init__()
        variable_space = ['train', 'test', 'val']
        assert dataset in variable_space,\
            "Invalid Flag for %s the variable space is %s" % (
                dataset, variable_space)
        normalizes(datasets, dense, dense)
        self.dense = dense
        self.data = datasets['X_' + dataset].copy()
        self.label = datasets['X_' + dataset].copy() if not dense \
            else datasets['y_' + dataset].copy()
        if not dense:
            self.classes = datasets['y_' + dataset].copy()

    def __getitem__(self, index):
        if self.dense:
            # label = np.array([0 for i in range(10)])
            # label[self.label[index]] = 1
            return torch.from_numpy(
                self.data[index].transpose((2, 0, 1)).copy()
            ).float(), self.label[index]
        y, cr, cb = rgb_to_ycrcb_channel_first(
            self.data[index, :, :, :],
            upscale=1)
        y_label, _, _ = rgb_to_ycrcb_channel_first(
            self.label[index, :, :, :])
        return torch.from_numpy(y).float(), cr, cb,\
            self.classes[index],\
            torch.from_numpy(y_label).float()

    def __len__(self):
        return self.data.shape[0]


# Testing
if __name__ == "__main__":
    # testing for obtain densenet dataset
    dataset = DataSets(dense=True)
    x, y = dataset.__getitem__(10)
    print(x.shape, y.shape, len(dataset))

    # testing for obtain msrn dataset
    msrn_dataset = DataSets(dense=False)
    x, cr, cb, y = msrn_dataset.__getitem__(10)
    print(x.shape, y.shape, len(msrn_dataset))

    msrn_dataset = DataSets(dense=False, dataset="test")
    x, cr, cb, y = msrn_dataset.__getitem__(10)
    print(x.shape, y.shape, len(msrn_dataset))

    msrn_dataset = DataSets(dense=False, dataset="val")
    x, cr, cb, y = msrn_dataset.__getitem__(10)
    print(x.shape, y.shape, len(msrn_dataset))
    # testing for obtain validation dataset
    del datasets
