from model_utility.models import ModelInit
import model_utility.metrics as metric
from data_utlity.semantic_segmentation_data_gen import DataGenerator
from data_utlity import data_normalization, data_augmentation
from model_utility import loss_function, models
from keras import optimizers

from tensorflow.python.client import device_lib


def run_mandatory_check(config):
    if not hasattr(data_normalization, config.normalization):
        print("method {} not implemented in data_utility/data_normalization.py".format(config.normalization))
        return False
    if not hasattr(loss_function, config.loss_function):
        print("method {} not implemented in model_utility/loss_function.py".format(config.loss_function))
        return False
    if config.model_input_dimension > config.image_dimension:
        print("Rescaling not supported model dimension should either be equal or less than image dimension"
              .format(config.loss_function))
        return False
    if not hasattr(data_augmentation, config.augmentation):
        print("method {} not implemented in data_utility/data_augmentation.py".format(config.augmentation))
        return False
    if not hasattr(models, config.model_name):
        print("method {} not implemented in model_utility/models.py".format(config.model_name))
        return False
    if config.test_loader_batch_size > config.batch_size:
        config.test_loader_batch_size = config.batch_size
        return True


def configure_training(model_name, optimizer_to_use, loss_to_use, image_dimension, classes, **kwargs):
    model_init = ModelInit(image_dimension, classes)
    model = getattr(model_init, model_name)()

    optimizer = getattr(optimizers, optimizer_to_use)
    optimizer = optimizer(**kwargs)

    loss = getattr(loss_function, loss_to_use)
    return model, optimizer, loss


def compile_model(model, optimizer, loss, metrics):
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def get_metrics(list_of_metric):
    final_metirc = []
    for value in list_of_metric:
        final_metirc.append(getattr(metric, value))
    return final_metirc


def get_available_gpus_count():
    local_device_protos = device_lib.list_local_devices()
    gpu_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
    return len(gpu_list)


def configure_data_gen(config):
    data_gen_train = DataGenerator(config=config, images_path=config.train_images_dir,
                                   labels_path=config.train_labels_dir,
                                   shuffle=True, using_val_generator=False)

    data_gen_val = DataGenerator(config=config, images_path=config.val_images_dir,
                                 labels_path=config.val_labels_dir,
                                 shuffle=True, using_val_generator=False)
    return data_gen_train, data_gen_val

