from keras import losses
import tensorflow as tf

import keras.backend as K


def binary_crossentropy(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred)
    return loss


def focal_loss(y_true, y_pred, alpha=0.25, gamma=0.5):
    bce_loss = binary_crossentropy(y_true, y_pred)
    pt_1 = tf.where(tf.equal(y_true, 1), 1 - y_pred, tf.zeros_like(y_pred))
    pt_1 = tf.where(tf.equal(y_true, 0), y_pred, pt_1)
    focal_weights = (alpha * K.pow(pt_1, gamma))
    return K.mean(focal_weights, axis=-1) * bce_loss


def dice_coef_loss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0))


def jaccard_coef(y_true, y_pred):
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, 2, 1])
    sum_ = K.sum(y_true, axis=[0, 2, 1]) + K.sum(y_pred, axis=[0, 2, 1])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return 1 - K.mean(jac)


