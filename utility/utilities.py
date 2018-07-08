import tifffile as tiff
import sys
import numpy as np


def random_crop(img, label, random_crop_size):
    # Note: image_data_format is 'channel_last'
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)

    if label is None:
        return img[y:(y + dy), x:(x + dx), :], None

    else:
        return img[y:(y + dy), x:(x + dx), :], label[y:(y+dy), x:(x+dx)]


def read_image(img_path):
    img = tiff.imread(img_path)
    return img


def read_label(label_path):
    if label_path is None:
        return None
    else:
        label = tiff.imread(label_path)
        return label


def get_image_label(img_path, label_path, model_input_dimension, input_image_dim, mean_cal_over_function=False):

    try:
        img = read_image(img_path)
        label = read_label(label_path)

        if mean_cal_over_function:
            return img, None

        if model_input_dimension != input_image_dim:
            img, label = random_crop(img, label, model_input_dimension)
        return img, label

    except Exception as ex:
        print(ex)
        print("Exception occured in get_image_label")
        print("{}-path to image".format(img_path))
        print("{}-path to label".format(label_path))
        sys.tracebacklimit = None
        raise SystemExit