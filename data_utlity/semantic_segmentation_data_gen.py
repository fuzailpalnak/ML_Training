import os
import threading
import copy
import random
import numpy as np
import sys

import data_utlity.data_augmentation as data_aug
import data_utlity.data_preprocessing as preprocessing

from data_utlity.logger import create_a_logger_file, get_logger_object
import utility.getter_setter as set_get_mean_std
import utility.utilities as utils
from data_utlity.data_gen_helper import get_mean_std_over_dataset, get_mean_over_dataset, generate_test_set


class ThreadsafeIter(object):

    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return ThreadsafeIter(f(*a, **kw))

    return g


class DataGenerator(object):
    # Generates data for Keras
    def __init__(self, config, images_path, labels_path, shuffle=False, using_val_generator=True):

        # Initialization

        self.config = config
        self.shuffle = shuffle
        self.images_check = []
        self.images_path = images_path
        self.labels_path = labels_path
        self.total_samples = 0
        self.initial_index = 0
        self.all_images_list = []

        self.mean = None
        self.std_dev = None
        self.using_val_generator = using_val_generator

        self.dir_structure = [main_dir for in_dir in os.walk(self.config.train_images_dir) if len(in_dir[1]) != 0
                              for main_dir in in_dir[1]]

        self.dir_structure_dict = {}
        self.dir_structure_dict_copy = {}
        self.read_me = {}
        self.height, self.width, self.bands = config.model_input_dimension
        if config.multi_label:
            self.class_for_training = utils.create_feature_dict(config.features, config.color_code,
                                                                config.num_of_multi_label_classes)
        else:
            self.class_for_training = {}

        if using_val_generator:
            self._populate_val_set()

        if self.using_val_generator is False:
            self.test_images_path = self.config.test_images_dir
            self.test_total_samples = len([file for file in os.listdir(self.test_images_path)])
            self.test_images_file_names = [file for file in os.listdir(self.test_images_path)]
            self.test_loader_size = self.config.test_loader_batch_size

            self._populate_train_set()

            if len(self.test_images_file_names) == 0 or self.test_total_samples == 0:

                print("Test Images can't be generated as test images folder is empty. "
                      "If not empty kindly check generate test data")
                print("Exception occured in datagenerator while trying to return")

                sys.tracebacklimit = None
                raise SystemExit

        if self.config.num_of_multi_label_classes is None:

            print("Number of multilabel classes found none while intitializing  DataGenerator")
            print("check you DataGenerator intilializer in train.py")

            sys.tracebacklimit = None
            raise SystemExit

        if shuffle:
            random.shuffle(self.all_images_list)

        if set_get_mean_std.get_calculate_over_dataset() is True:
            # http://cs231n.github.io/neural-networks-2/#datapre
            if self.config.normalization == "mean_std_over_dataset":
                get_mean_std_over_dataset(self)

            elif self.config.normalization == "only_mean_over_dataset":
                self.mean = 0
                get_mean_over_dataset(self)

    def _populate_train_set(self):
        if len(self.dir_structure) != 0:
            for folder_name in self.dir_structure:
                self.dir_structure_dict.setdefault(folder_name, [])
                self.dir_structure_dict[folder_name].append([file for file in
                                                             os.listdir(
                                                                 os.path.join(self.images_path, folder_name))])
                [self.all_images_list.append(os.path.join(folder_name, file)) for file in
                 os.listdir(os.path.join(self.images_path, folder_name))]
                self.total_samples = len(self.all_images_list)
                self.dir_structure_dict_copy = copy.deepcopy(self.dir_structure_dict)
        else:
            self.dir_structure_dict.setdefault("train", [])
            self.dir_structure_dict["train"].append([file for file in
                                                     os.listdir(self.images_path)])
            [self.all_images_list.append(file) for file in
             os.listdir(self.images_path)]
            self.total_samples = len(self.all_images_list)
            self.dir_structure.append("train")
            self.dir_structure_dict_copy = copy.deepcopy(self.dir_structure_dict)

    def _populate_val_set(self):
        if len(self.dir_structure) != 0:
            for folder_name in self.dir_structure:
                self.dir_structure_dict.setdefault(folder_name, [])
                self.dir_structure_dict[folder_name].append([file for file in
                                                             os.listdir(os.path.join(self.images_path,
                                                                                     folder_name))])
                [self.all_images_list.append(os.path.join(folder_name, file)) for file in
                 os.listdir(os.path.join(self.images_path, folder_name))]
                self.total_samples = len(self.all_images_list)
                self.dir_structure_dict_copy = copy.deepcopy(self.dir_structure_dict)
        else:
            self.dir_structure_dict.setdefault("val", [])

            self.dir_structure_dict["val"].append([file for file in
                                                   os.listdir(self.images_path)])
            [self.all_images_list.append(file) for file in
             os.listdir(self.images_path)]
            self.total_samples = len(self.all_images_list)
            self.dir_structure.append("val")
            self.dir_structure_dict_copy = copy.deepcopy(self.dir_structure_dict)

    def get_steps_per_epoch(self):
        """

        :return:
        """
        return self.total_samples / self.config.batch_size

    @threadsafe_generator
    def generate(self):
        # Generates batches of samples
        # Infinite loop

        while 1:
            images = []
            labels = []
            file_name_list = []

            test_images_list = []
            test_images_file_name_list = []

            if self.using_val_generator is False:
                random.shuffle(self.test_images_file_names)

            for i in range(0, self.config.batch_size):

                folder_name = self.dir_structure[self.initial_index]
                if len(self.dir_structure_dict[folder_name][0]) < self.config.batch_size:
                    self.dir_structure_dict[folder_name][0] = self.dir_structure_dict_copy[folder_name][0].copy()
                file_name = self.dir_structure_dict[folder_name][0][0]

                if folder_name != "train" and folder_name != "val":
                    image, label = utils.get_image_label(img_path=os.path.join(self.images_path,
                                                                         os.path.join(folder_name, file_name)),
                                                         label_path=os.path.join(self.labels_path,
                                                                         os.path.join(folder_name, file_name)),
                                                         model_input_dimension=self.config.model_input_dimension,
                                                         input_image_dim=self.config.image_dimension)
                else:
                    image, label = utils.get_image_label(os.path.join(self.images_path, file_name),
                                                         os.path.join(self.labels_path, file_name),
                                                         model_input_dimension=self.config.model_input_dimension,
                                                         input_image_dim=self.config.image_dimension)

                if self.config.augment and not self.using_val_generator:
                    if self.config.augment_frequency % random.randint(1, 10) == 0:
                        image, label = data_aug.random_augmentation(image, label, self.config.augmentation,
                                                                    self.config.augmentation_type)

                label = preprocessing.perform_preprocessing_label(label, file_name,
                                                                  image_dimension=label.shape,
                                                                  num_of_multi_label_classes=
                                                                  self.config.num_of_multi_label_classes,
                                                                  Multi_label=self.config.multi_label,
                                                                  training_classes=self.class_for_training)

                image = preprocessing.perform_normalization(image, self.config.normalization)

                if self.using_val_generator is False:
                    generate_test_set(self, i, test_images_list, test_images_file_name_list)

                images.append(image)
                labels.append(label)

                file_name_list.append(file_name)
                self.dir_structure_dict[folder_name][0].remove(file_name)

                if self.initial_index == len(self.dir_structure) - 1:
                    self.initial_index = 0
                else:
                    self.initial_index = self.initial_index + 1

            if self.config.multi_label:
                try:
                    yield np.array(images).reshape(self.config.batch_size, self.height, self.width, self.bands), np.array(labels). \
                              reshape(self.config.batch_size, self.height, self.width,
                                      self.config.number_multilabel_classes)
                except ValueError as ex:
                    print(ex)
                    print("Exception occured in Data-Generator while trying to return")
                    sys.tracebacklimit = None
                    raise SystemExit
            else:
                try:
                    yield np.array(images).reshape(self.config.batch_size, self.height, self.width, self.bands), \
                          np.array(labels).reshape(self.config.batch_size, self.height, self.width, 1)
                except ValueError as ex:
                    print(ex)
                    print("Exception occured in Data-Generator while trying to return ")
                    sys.tracebacklimit = None
                    raise SystemExit


