import os
import datetime
import numpy as np
from scipy.misc import imsave
from skimage import exposure, img_as_uint
from utility.getter_setter import get_test_images, get_test_images_file_name

from keras import callbacks


class DrawPredictionCallback(callbacks.Callback):

    def __init__(self, checkpoints_folder, multi_label, training_class):

        self.test_filenames = None
        self.test_imgs = None
        self.test_image_save_path = checkpoints_folder
        self.multi_label = multi_label
        self.training_class = training_class

    def on_epoch_end(self, epoch, logs={}):

        filelist = [f for f in os.listdir(self.test_image_save_path) if f.endswith(".png")]
        for f in filelist:
            os.remove(os.path.join(self.test_image_save_path, f))
        self.test_filenames = get_test_images_file_name()
        self.test_imgs = get_test_images()
        i = 0
        prediction_list = []
        prediction_file_names = []
        for test_img in self.test_imgs:
            prediction_file_names.append(self.test_filenames[i])
            i += 1
            prediction_list.append(test_img)
            prediction_image = np.array(prediction_list)

            if len(prediction_list) % 1 == 0:
                p = self.model.predict(prediction_image)
                _, height, width, bands = np.array(p).shape

                if self.multi_label:
                    image = p[0].reshape(height, width, bands)
                    image = image.argmax(axis=-1)

                    image = np.expand_dims(image, axis=-1)
                    image = np.concatenate(3 * (image,), axis=-1)
                    for count in range(0, bands):
                        image[np.where((image == [(self.training_class[count][2],
                                                   self.training_class[count][2],
                                                   self.training_class[count][2])]).all(axis=2))] = \
                            [self.training_class[count][1]]

                    save_multi_label_output(self.test_image_save_path, prediction_file_names, image, epoch)

                else:
                    save_output(p, len(prediction_list), prediction_file_names,
                                height, width, self.test_image_save_path, epoch)

                prediction_list = []
                prediction_file_names = []


def save_multi_label_output(checkpoints_folder,prediction_file_names, result_image, epoch):
    imsave(checkpoints_folder + "/{}_epoch_{}.png".format(prediction_file_names[0], epoch), result_image)


def save_output(p, len_prediction_list, prediction_file_names, height, width, checkpoints_folder, epoch):

    for i in range(0, len_prediction_list):
        result_image = p[i].reshape((height, width))
        result_image = exposure.rescale_intensity(result_image, out_range='float')
        result_image = img_as_uint(result_image)

        imsave(checkpoints_folder+"/{}_epoch_{}.png".format(prediction_file_names[i],epoch), result_image)


def model_checkpoint(weights_path):
    model_checkpoint_cback = callbacks.ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True)
    return model_checkpoint_cback


def reduce_lr():
    reduce_lr_cback = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,
        patience=5,
        verbose=1,
        min_lr=1e-3)
    return reduce_lr_cback


def tensorboard(tboard_path):
    tensor_cback = callbacks.TensorBoard(log_dir='{}'.format(tboard_path), histogram_freq=0,
                                         write_grads=True,
                                         write_graph=True, write_images=True)
    return tensor_cback


def get_callbacks(multi_label, training_class):

    log_folder = os.path.join(os.getcwd(), "logs")
    save_images_folder = os.path.join(log_folder, "test_images_"+datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    current_running_folder = os.path.join(log_folder, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    tboard_path = os.path.join(log_folder, "tboard_"+datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    weights_path = os.path.join(log_folder, "model_weights"+datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    try:
        os.makedirs(current_running_folder)
    except OSError:
        pass

    model_checkpoint_cback = model_checkpoint(weights_path)
    reduce_lr_cback = reduce_lr()
    tensor_cback = tensorboard(tboard_path)
    draw_prediction_cback = DrawPredictionCallback(save_images_folder, multi_label, training_class)

    return [model_checkpoint_cback, reduce_lr_cback, draw_prediction_cback, tensor_cback]



