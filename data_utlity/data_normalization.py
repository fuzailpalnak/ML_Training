import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler

import utility.getter_setter as set_get_mean_std


def mean_std_over_dataset(img_array):
    mean = set_get_mean_std.get_mean_over_dataset()
    std_dev = set_get_mean_std.get_std_dev_over_dataset()

    height, width, bands = img_array.shape

    if mean is None or std_dev is None:
        print("Could not perform mean_std_over_dataset normalization")
        print("Mean={} and std_dev={}".format(mean, std_dev))
        sys.tracebacklimit = None
        raise SystemExit
    else:

        img_array = img_array.astype('float32')
        img_array = (img_array.reshape(height * width, bands) - mean) / std_dev
        img_array = img_array.reshape(height, width, bands)
    return img_array


def only_mean_over_dataset(img_array):
    mean = set_get_mean_std.get_mean_over_dataset()
    height, width, bands = img_array.shape

    if mean is None:
        print("Could not perform only_mean_over_dataset normalization")
        print("Mean={}".format(mean))
        sys.tracebacklimit = None
        raise SystemExit
    else:

        img_array = img_array.astype('float32')
        img_array = (img_array.reshape(height * width, bands) - mean)
        img_array = img_array.reshape(height, width, bands)
    return img_array


def std_dev_normalization(img_array):
    img_array = img_array.astype('float32')
    img_array = (img_array - np.mean(img_array)) / np.std(img_array)
    return img_array


def min_max(img_array):
    height, width, bands = img_array.shape

    img_array = img_array.reshape(height * width, bands)
    scaler = MinMaxScaler()
    img_array = scaler.fit_transform(img_array)
    img_array = img_array.reshape(height, width, bands)
    return img_array
