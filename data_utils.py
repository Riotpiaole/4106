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
        channel_first=True):
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

    normalizes(all_datas, subtract_mean, channel_first)

    # Package data into a dictionary
    return all_datas


def load_tiny_imagenet(path, dtype=np.float32, subtract_mean=True):
    """
    Load TinyImageNet. Each of TinyImageNet-100-A, TinyImageNet-100-B, and
    TinyImageNet-200 have the same directory structure, so this can be used
    to load any of them.

    Inputs:
    - path: String giving path to the directory to load.
    - dtype: numpy datatype used to load the data.
    - subtract_mean: Whether to subtract the mean training image.

    Returns: A dictionary with the following entries:
    - class_names: A list where class_names[i] is a list of strings giving the
      WordNet names for class i in the loaded dataset.
    - X_train: (N_tr, 3, 64, 64) array of training images
    - y_train: (N_tr,) array of training labels
    - X_val: (N_val, 3, 64, 64) array of validation images
    - y_val: (N_val,) array of validation labels
    - X_test: (N_test, 3, 64, 64) array of testing images.
    - y_test: (N_test,) array of test labels; if test labels are not available
      (such as in student code) then y_test will be None.
    - mean_image: (3, 64, 64) array giving mean training image
    """
    # First load wnids
    with open(os.path.join(path, 'wnids.txt'), 'r') as f:
        wnids = [x.strip() for x in f]

    # Map wnids to integer labels
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # Use words.txt to get names for each class
    with open(os.path.join(path, 'words.txt'), 'r') as f:
        wnid_to_words = dict(line.split('\t') for line in f)
        for wnid, words in wnid_to_words.iteritems():
            wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
    class_names = [wnid_to_words[wnid] for wnid in wnids]

    # Next load training data.
    X_train = []
    y_train = []
    for i, wnid in enumerate(wnids):
        if (i + 1) % 20 == 0:
            print(
                'loading training data for synset %d / %d' %
                (i + 1, len(wnids)))
        # To figure out the filenames we need to open the boxes file
        boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
        with open(boxes_file, 'r') as f:
            filenames = [x.split('\t')[0] for x in f]
        num_images = len(filenames)

        X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
        y_train_block = wnid_to_label[wnid] * \
            np.ones(num_images, dtype=np.int64)
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, 'train', wnid, 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                # grayscale file
                img.shape = (64, 64, 1)
            X_train_block[j] = img.transpose(2, 0, 1)
        X_train.append(X_train_block)
        y_train.append(y_train_block)

    # We need to concatenate all training data
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # Next load validation data
    with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
        img_files = []
        val_wnids = []
        for line in f:
            img_file, wnid = line.split('\t')[:2]
            img_files.append(img_file)
            val_wnids.append(wnid)
        num_val = len(img_files)
        y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
        X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, 'val', 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)
            X_val[i] = img.transpose(2, 0, 1)

    # Next load test images
    # Students won't have test labels, so we need to iterate over files in the
    # images directory.
    img_files = os.listdir(os.path.join(path, 'test', 'images'))
    X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
    for i, img_file in enumerate(img_files):
        img_file = os.path.join(path, 'test', 'images', img_file)
        img = imread(img_file)
        if img.ndim == 2:
            img.shape = (64, 64, 1)
        X_test[i] = img.transpose(2, 0, 1)

    y_test = None
    y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
    if os.path.isfile(y_test_file):
        with open(y_test_file, 'r') as f:
            img_file_to_wnid = {}
            for line in f:
                line = line.split('\t')
                img_file_to_wnid[line[0]] = line[1]
        y_test = [wnid_to_label[img_file_to_wnid[img_file]]
                  for img_file in img_files]
        y_test = np.array(y_test)

    mean_image = X_train.mean(axis=0)
    if subtract_mean:
        X_train -= mean_image[None]
        X_val -= mean_image[None]
        X_test -= mean_image[None]

    return {
        'class_names': class_names,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'class_names': class_names,
        'mean_image': mean_image,
    }


datasets = get_CIFAR10_data()


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

    def __getitem__(self, index):
        if self.dense:
            return torch.from_numpy(
                self.data[index]).float(), torch.from_numpy(
                np.array([self.label[index]]))

        return torch.from_numpy(
            rgb_to_ycrcb_channel_first(
                self.data[index, :, :, :],
                upscale=1)).float(), \
            torch.from_numpy(
            rgb_to_ycrcb_channel_first(
                self.label[index, :, :, :])).float()

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
    x, y = msrn_dataset.__getitem__(10)
    print(x.shape, y.shape, len(msrn_dataset))

    msrn_dataset = DataSets(dense=False, dataset="test")
    x, y = msrn_dataset.__getitem__(10)
    print(x.shape, y.shape, len(msrn_dataset))

    msrn_dataset = DataSets(dense=False, dataset="val")
    x, y = msrn_dataset.__getitem__(10)
    print(x.shape, y.shape, len(msrn_dataset))
    # testing for obtain validation dataset
    del datasets
