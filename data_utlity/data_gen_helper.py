import os
import numpy as np

import data_utlity.data_preprocessing as preprocessing
import utility.getter_setter as set_get
import utility.utilities as utils


def get_mean_over_dataset(data_gen_object):
    data_gen_object.mean = 0
    print("Calculating mean over dataset")
    for i in range(0, data_gen_object.total_samples):
        file_name = data_gen_object.all_images_list[i]
        image, _ = utils.get_image_label(os.path.join(data_gen_object.config.train_images_dir, file_name),
                                         label_path=None,
                                         model_input_dimension=data_gen_object.config.model_input_dimension,
                                         input_image_dim=data_gen_object.config.image_dimension,
                                         mean_cal_over_function=True)

        data_gen_object.mean = data_gen_object.mean + np.mean(image.reshape(data_gen_object.height *
                                                                            data_gen_object.width,
                                                                            data_gen_object.bands), axis=0)
    data_gen_object.mean = data_gen_object.mean / data_gen_object.total_samples

    set_get.set_mean_over_dataset(data_gen_object.mean)

    print("Calculated mean {}".format(data_gen_object.mean))

    set_get.set_calculate_over_dataset(False)


def get_mean_std_over_dataset(data_gen_object):
    data_gen_object.mean = 0
    data_gen_object.std_dev = 0
    print("Calculating mean over dataset")
    for i in range(0, data_gen_object.total_samples):

        file_name = data_gen_object.all_images_list[i]
        image, _ = utils.get_image_label(os.path.join(data_gen_object.config.train_images_dir, file_name),
                                         label_path=None,
                                         model_input_dimension=data_gen_object.config.model_input_dimension,
                                         input_image_dim=data_gen_object.config.image_dimension,
                                         mean_cal_over_function=True)

        data_gen_object.mean = data_gen_object.mean + np.mean(image.reshape(data_gen_object.height *
                                                                            data_gen_object.width,
                                                                            data_gen_object.bands), axis=0)
        data_gen_object.std_dev = data_gen_object.std_dev + np.std(image.reshape(data_gen_object.height *
                                                                                 data_gen_object.width,
                                                                                 data_gen_object.bands), axis=0)
    data_gen_object.mean = data_gen_object.mean / data_gen_object.total_samples
    data_gen_object.std_dev = data_gen_object.std_dev / data_gen_object.total_samples

    set_get.set_mean_over_dataset(data_gen_object.mean)
    set_get.set_std_dev_over_dataset(data_gen_object.std_dev)

    print("Calculated mean {}".format(data_gen_object.mean))
    print("Calculated std_dev {}".format(data_gen_object.std_dev))
    set_get.set_calculate_over_dataset(False)


def generate_test_set(data_gen_object, i, test_images_list, test_images_file_name_list):
    test_file_name = data_gen_object.test_images_file_names[i]
    if i <= data_gen_object.test_loader_size - 1:
        test_images_file_name_list.append(test_file_name)
        test_image, _ = utils.get_image_label(os.path.join(data_gen_object.test_images_path, test_file_name),
                                              label_path=None,
                                              model_input_dimension=data_gen_object.config.model_input_dimension,
                                              input_image_dim=data_gen_object.config.image_dimension)

        test_image = preprocessing.perform_normalization(test_image,
                                                         data_gen_object.config.normalization)
        if i <= data_gen_object.test_loader_size - 1 and test_image is not None:
            test_images_list.append(test_image)

        if i == data_gen_object.test_loader_size - 1:
            set_get.set_test_images(test_images_list)
            set_get.set_test_images_file_name(test_images_file_name_list)
