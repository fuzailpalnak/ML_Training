from keras.metrics import binary_accuracy
import keras.backend as K


def precision(y_true, y_pred):
    """

    :param y_true: true label
    :param y_pred: predicted label
    :return: precision
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    score = true_positives / (predicted_positives + K.epsilon())
    return score


def recall(y_true, y_pred):
    """

    :param y_true: true label
    :param y_pred: predicted label
    :return: recall
    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    score = true_positives / (possible_positives + K.epsilon())
    return score


def f_beta_score(y_true, y_pred, beta=1):

    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return score


def f_score(y_true, y_pred):
    """

    :param y_true: true label
    :param y_pred: predicted label
    :return: fscore
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    return 2*((p*r)/(p+r+K.epsilon()))


def jaccard_coef(y_true, y_pred):
    smooth = 1e-12

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, 2, 1])
    sum_ = K.sum(y_true, axis=[0, 2, 1]) + K.sum(y_pred_pos, axis=[0, 2, 1])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac, axis=-1)


def accuracy(y_true, y_pred):
    return binary_accuracy(y_true, y_pred)

