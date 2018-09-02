import tifffile as tiff
import numpy as np
from keras.utils import to_categorical
import sys
from os.path import exists
from os import remove
from utility.print_handler import colored_dual_string_print


def random_crop(img, label, random_crop_size):
    # Note: image_data_format is 'channel_last'
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size[0], random_crop_size[1]
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
        colored_dual_string_print("Exception", ex, "red", "yellow", attrs=['blink'])
        colored_dual_string_print("Image Path", "{}".format(img_path), "red", "yellow", attrs=['blink'])
        colored_dual_string_print("Label Path", "{}".format(label_path), "red", "yellow", attrs=['blink'])
        sys.exit()


def create_feature_dict(features, color_code_for_features, num_of_multi_label_classes):
    """

    ex-
        input=
        features = ["EveryThingElse", "Paved", "Unpaved"]
        color_code_for_features = [(255,255,255), (255,0,0),(0,255,0)]

        output={0:['EveryThingElse',(255,255,255),0],1:['Paved',(255,0,0),1],2:['Unpaved',(0,255,0),2]}

    """
    class_for_training = {}
    if len(features) != num_of_multi_label_classes:
        colored_dual_string_print("Mismatch", "Mismatch in features and num_of_multi_label_classes",
                                  "red", "yellow", attrs=['blink'])

        return False

    if len(features) != len(color_code_for_features):
        colored_dual_string_print("Mismatch", "Mismatch in features and color_for_features",
                                  "red", "yellow", attrs=['blink'])
        return False

    for features_count, features in enumerate(features):
        class_for_training.setdefault(features_count, [])
        class_for_training[features_count].append(features)
        class_for_training[features_count].append(color_code_for_features[features_count])
        class_for_training[features_count].append(features_count)
    return class_for_training


def create_one_hot_code_for_each_image(img, height, width, file_name, num_of_classes, class_for_training):

    """
    https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
    https://en.wikipedia.org/wiki/One-hot


    ex-
        [house, car, tooth, car] becomes
        [[1,0,0,0],
        [0,1,0,1],
        [0,0,1,0]]
    """
    if len(img.shape) == 2:
        colored_dual_string_print("File Name", "{}".format(file_name), "red", "yellow", attrs=['blink'])
        colored_dual_string_print("Error", "Found Binary image ,Image Bands should be greater than 2",
                                  "red", "yellow", attrs=['blink'])
        colored_dual_string_print("Check", "Please check input , or MultiLabel Variable might be true in config file",
                                  "red", "yellow", attrs=['blink'])
        sys.exit()

    for feature_count in range(0, num_of_classes):
        img[np.where((img == [class_for_training[feature_count][1]]).all(axis=2))] = [class_for_training[feature_count][2]]

    if np.amax([img > num_of_classes]):
        if exists("inconsistent_pixel_values.txt"):
            remove("inconsistent_pixel_values.txt")
        file = open("inconsistent_pixel_values.txt", 'w')
        file.write("filename - {}".format(str(file_name))+"\n")

        for x in range(0, height):
            for y in range(0, width):
                for count in range(0, num_of_classes):
                    if tuple(img[x][y]) == (class_for_training[count][2], class_for_training[count][2],
                                            class_for_training[count][2]):
                        break
                    else:
                        file.write("pixel location - x= {} y={} pixel values-{}".format(str(x), str(y), str(img[x][y]))+"\n")
        file.close()
        colored_dual_string_print("File Name", "{}".format(file_name), "red", "yellow", attrs=['blink'])
        colored_dual_string_print("Error", "Inconsistent Pixel found please check the provided input label",
                                  "red", "yellow", attrs=['blink'])
        colored_dual_string_print("Check", "Check inconsistent_pixel_values.txt ",
                                  "red", "yellow", attrs=['blink'])
        sys.exit()

    img = img[:, :, :1]
    img_one_hot_coded = to_categorical(img, num_of_classes)
    img_one_hot_coded = np.array(img_one_hot_coded).reshape(height, width,
                                                            num_of_classes)
    return img_one_hot_coded
