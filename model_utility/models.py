import sys

import keras
from keras.layers import Input, MaxPooling2D, UpSampling2D, Conv2D, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization


class ModelInit(object):
    def __init__(self, image_dimension, classes):
        self.image_width, self.image_height, self.bands = image_dimension
        self.classes = classes

    def unet(self):
        inputs = Input((self.image_width, self.image_height, self.bands))
        conv1 = Conv2D(32, (3, 3), padding="same", kernel_initializer="he_uniform")(inputs)
        conv1 = BatchNormalization(axis=3)(conv1)
        conv1 = keras.layers.advanced_activations.ELU()(conv1)

        conv1 = Conv2D(32, (3, 3), padding="same", kernel_initializer="he_uniform")(conv1)
        conv1 = BatchNormalization(axis=3)(conv1)
        conv1 = keras.layers.advanced_activations.ELU()(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), padding="same", kernel_initializer="he_uniform")(pool1)
        conv2 = BatchNormalization(axis=3)(conv2)
        conv2 = keras.layers.advanced_activations.ELU()(conv2)

        conv2 = Conv2D(64, (3, 3), padding="same", kernel_initializer="he_uniform")(conv2)
        conv2 = BatchNormalization(axis=3)(conv2)
        conv2 = keras.layers.advanced_activations.ELU()(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), padding="same", kernel_initializer="he_uniform")(pool2)
        conv3 = BatchNormalization(axis=3)(conv3)
        conv3 = keras.layers.advanced_activations.ELU()(conv3)

        conv3 = Conv2D(128, (3, 3), padding="same", kernel_initializer="he_uniform")(conv3)
        conv3 = BatchNormalization(axis=3)(conv3)
        conv3 = keras.layers.advanced_activations.ELU()(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), padding="same", kernel_initializer="he_uniform")(pool3)
        conv4 = BatchNormalization(axis=3)(conv4)
        conv4 = keras.layers.advanced_activations.ELU()(conv4)
        conv4 = Conv2D(256, (3, 3), padding="same", kernel_initializer="he_uniform")(conv4)
        conv4 = BatchNormalization(axis=3)(conv4)
        conv4 = keras.layers.advanced_activations.ELU()(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), padding="same", kernel_initializer="he_uniform")(pool4)
        conv5 = BatchNormalization(axis=3)(conv5)
        conv5 = keras.layers.advanced_activations.ELU()(conv5)
        conv5 = Conv2D(512, (3, 3), padding="same", kernel_initializer="he_uniform")(conv5)
        conv5 = BatchNormalization(axis=3)(conv5)
        conv5 = keras.layers.advanced_activations.ELU()(conv5)

        up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), padding="same", kernel_initializer='he_uniform')(up6)
        conv6 = BatchNormalization(axis=3)(conv6)
        conv6 = keras.layers.advanced_activations.ELU()(conv6)
        conv6 = Conv2D(256, (3, 3), padding="same", kernel_initializer='he_uniform')(conv6)
        conv6 = BatchNormalization(axis=3)(conv6)
        conv6 = keras.layers.advanced_activations.ELU()(conv6)

        up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_uniform')(up7)
        conv7 = BatchNormalization(axis=3)(conv7)
        conv7 = keras.layers.advanced_activations.ELU()(conv7)
        conv7 = Conv2D(128, (3, 3), padding="same", kernel_initializer='he_uniform')(conv7)
        conv7 = BatchNormalization(axis=3)(conv7)
        conv7 = keras.layers.advanced_activations.ELU()(conv7)

        up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_uniform')(up8)
        conv8 = BatchNormalization(axis=3)(conv8)
        conv8 = keras.layers.advanced_activations.ELU()(conv8)
        conv8 = Conv2D(64, (3, 3), padding="same", kernel_initializer='he_uniform')(conv8)
        conv8 = BatchNormalization(axis=3)(conv8)
        conv8 = keras.layers.advanced_activations.ELU()(conv8)

        up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_uniform')(up9)
        conv9 = BatchNormalization(axis=3)(conv9)
        conv9 = keras.layers.advanced_activations.ELU()(conv9)
        conv9 = Conv2D(32, (3, 3), padding="same", kernel_initializer='he_uniform')(conv9)

        # crop9 = Cropping2D(cropping=((16, 16), (16, 16)))(conv9)
        conv9 = BatchNormalization(axis=3)(conv9)
        conv9 = keras.layers.advanced_activations.ELU()(conv9)

        if self.classes > 1:
            conv10 = Conv2D(self.classes, (1, 1), activation='softmax')(conv9)
        else:
            conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
        model = Model(inputs, conv10)

        return model

