import numpy as np
import cv2
from skimage import exposure, color

import sys
import utility.getter_setter as set_get_mean_std
from utility.utilities import create_one_hot_code_for_each_image


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


def perform_normalization(img_array, normalization):
    """

    :param img_array: input image
    :param normalization: normalization to perform
    :return:
    """
    try:
        img_array = eval(normalization)(img_array)

        return img_array
    except Exception as ex:
        print(str(ex))
        return None


def convert_to_greyscale(img):
    try:
        img_grey = color.rgb2gray(img) * 255
        return img_grey
    except Exception as ex:
        print(str(ex))
        return None


def perform_clahe(img_array):
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = convert_to_greyscale(img_array).astype(np.uint8)
    cl1 = clahe.apply(gray)

    return cl1


def perform_binary_label_normalization(image, height, width, file_name):
    try:

        label_image = np.array(image).reshape((height, width, 1))
        label_image = label_image[:, :] / 255
    except Exception as ex:
        print(ex)
        print("{}-infile".format(file_name))

        print("Exception occured in perform_preprocessing_label")
        print("please check if given dir path is for labeled images")
        sys.tracebacklimit = None
        raise SystemExit
    return label_image


def perform_normalization_multi_label(image, height, width, file_name, classes, training_classes):
    try:
        try:
            label_image = create_one_hot_code_for_each_image(image, height, width, file_name, classes, training_classes)
        except Exception as ex:
            print(ex)
            print("{}-infile".format(file_name))
            print("Exception occured in perform_preprocessing_label when trying to perform one hot code")
            print("please check given color code combination are satisfied in labelled images")

            sys.tracebacklimit = None
            raise SystemExit
        label_image = np.array(label_image).reshape(
            (height, width, classes))
    except Exception as ex:
        print(ex)
        print("{}-infile".format(file_name))
        print("Exception occured in perform_preprocessing_label")
        print("please check if given dir path is for labeled images")
        sys.tracebacklimit = None
        raise SystemExit
    return label_image


def perform_preprocessing_label(label_image, file_name=None, Multi_label=False, image_dimension=None,
                                num_of_multi_label_classes=2, training_classes=None):
    """

    :param label_image: label image as numpy array
    :return: preprocessed image
    NOTE-if image is binary it will divide the image with 255 to bring the range between [0,1]
        -if image is multi labeled it create one hot code for the following label
    """

    try:

        if Multi_label:
            height, width, _ = image_dimension
            label_image = perform_normalization_multi_label(label_image, height, width, file_name,
                                                            num_of_multi_label_classes, training_classes)

        else:
            height, width = image_dimension
            label_image = perform_binary_label_normalization(label_image, height, width, file_name)

        return label_image

    except Exception as ex:
        print(str(ex))
        return None



