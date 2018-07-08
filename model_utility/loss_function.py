from keras import losses


def binary_crossentropy(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred)
    return loss