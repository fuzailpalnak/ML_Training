from utility.print_handler import colored_dual_string_print
import train_utility.train_metrics as metric
from data_utlity.semantic_segmentation_data_gen import DataGenerator
from data_utlity import data_normalization
from train_utility import train_loss_function
import models
import os
import sys

from keras import optimizers

from tensorflow.python.client import device_lib


def run_mandatory_check(config):
    if not hasattr(data_normalization, config.normalization):
        colored_dual_string_print("Sanity Check Failed", "method {} not implemented in "
                                                         "data_utility/data_normalization.py".
                                  format(config.normalization), "red", "yellow", attrs=['bold'])
        sys.exit()

    if not hasattr(train_loss_function, config.loss_function):
        colored_dual_string_print("Sanity Check Failed", "method {} not implemented in model_utility/loss_function.py"
                                  .format(config.loss_function), "red", "yellow", attrs=['bold'])
        sys.exit()

    if config.model_input_dimension > config.image_dimension:
        colored_dual_string_print("Sanity Check Failed", "Rescaling not supported model dimension should either be"
                                                         " equal or less than image dimension"
                                  .format(config.loss_function), "red", "yellow", attrs=['bold'])
        sys.exit()

    if not hasattr(models, config.model_name):
        colored_dual_string_print("Sanity Check Failed", "method {} not implemented in models"
                                  .format(config.model_name),
                                  "red", "yellow", attrs=['bold'])
        sys.exit()

    if config.test_loader_batch_size > config.batch_size:
        config.test_loader_batch_size = config.batch_size


def configure_training(model_name, optimizer_to_use, loss_to_use, config):
    try:
        model_object = getattr(models, model_name)()
        model = model_object(config)

        optimizer = getattr(optimizers, optimizer_to_use)
        optimizer = optimizer(lr=config.lr)

        loss = getattr(train_loss_function, loss_to_use)
        return model, optimizer, loss
    except Exception as ex:
        colored_dual_string_print("Exception", ex, "red", "yellow", attrs=['blink'])
        colored_dual_string_print("Function", "confiure_training", "red", "yellow", attrs=['blink'])
        sys.exit()


def configure_model_complie(model, optimizer, loss, metrics):
    try:
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model
    except Exception as ex:
        colored_dual_string_print("Exception", ex, "red", "yellow", attrs=['blink'])
        colored_dual_string_print("Function", "configure_model_complie", "red", "yellow", attrs=['blink'])
        sys.exit()


def configure_metrics(list_of_metric):
    final_metirc = []
    try:
        for value in list_of_metric:
            final_metirc.append(getattr(metric, value))
        return final_metirc
    except Exception as ex:
        colored_dual_string_print("Exception", ex, "red", "yellow", attrs=['blink'])
        colored_dual_string_print("Function", "configure_metrics", "red", "yellow", attrs=['blink'])
        sys.exit()


def get_available_gpus_count():
    local_device_protos = device_lib.list_local_devices()
    gpu_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return len(gpu_list)


def configure_data_gen(config):
    try:
        data_gen_train = DataGenerator(config=config, images_path=config.train_images_dir,
                                       labels_path=config.train_labels_dir,
                                       shuffle=True, using_val_generator=False)

        data_gen_val = DataGenerator(config=config, images_path=config.val_images_dir,
                                     labels_path=config.val_labels_dir,
                                     shuffle=True, using_val_generator=True)
        return data_gen_train, data_gen_val
    except Exception as ex:
        colored_dual_string_print("Exception", ex, "red", "yellow", attrs=['blink'])
        colored_dual_string_print("Function", "configure_data_gen", "red", "yellow", attrs=['blink'])
        sys.exit()


def create_neccesary_folder():
    if not os.path.exists(os.path.join(os.getcwd(), "logs")):
        os.mkdir(os.path.join(os.getcwd(), "logs"))

