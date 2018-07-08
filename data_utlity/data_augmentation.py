from imgaug import augmenters as iaa
import random
import numpy as np


def horizontal_flip(image, label):
    return iaa.Fliplr(1).augment_image(image), iaa.Fliplr(1).augment_image(label)


def vertical_flip(image, label):
    return iaa.Flipud(1).augment_image(image), iaa.Flipud(1).augment_image(label)


def rotate(image, label):
    number = np.random.randint(0, 45)
    return iaa.Affine(rotate=number).augment_image(image), iaa.Affine(rotate=number).augment_image(label)


def salt_pepper_noise(image, label, amount=0.4):
    return iaa.SaltAndPepper(p=amount, per_channel=True).augment_image(image), label


def gaussian_noise(image, label):
    return iaa.AdditiveGaussianNoise(scale=(0.0, 0.05*255), per_channel=0.5).augment_image(image), label


def perspective_transform(image, label):
    return iaa.PerspectiveTransform(scale=(0.01, 0.1), keep_size=True).augment_image(image), \
           iaa.PerspectiveTransform(scale=(0.01, 0.1), keep_size=True).augment_image(label)


def contrast_normalization(image, label):
    return iaa.ContrastNormalization((0.75, 1.5)).augment_image(image), label


def multiply(image, label):
    return iaa.Multiply((0.8, 1.2), per_channel=0.2).augment_image(image), label


def emboss(image, label):
    return iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)).augment_image(image), label


def dropout(image, label):
    return iaa.Dropout((0.01, 0.1), per_channel=0.5).augment_image(image), label


def hue_saturation(image, label):
    return iaa.AddToHueAndSaturation(-45).augment_image(image), label


def translate(image, label):
    randx = random.uniform(-0.2, 0.2)
    randy = random.uniform(-0.2, 0.2)
    aug = iaa.Affine(translate_percent={"x": randx, "y": randy}, mode="reflect")
    return aug.augment_image(image), aug.augment_image(label)


def random_augmentation(image, label, augmentation, augmentation_type):

    augmentation_copy = augmentation.copy()
    assert augmentation_type in ("AllOf", "OneOf", "SomeOf"), \
        "Augmentation Type should be 'OneOf', 'AllOf' or 'SomeOf'"

    if augmentation_type == "AllOf":
        image, label = all_of(image, label, augmentation_copy)

    elif augmentation_type == "OneOf":
        image, label = one_of(image, label, augmentation_copy)

    elif augmentation_type == "SomeOf":
        image, label = some_of(image, label, augmentation_copy)

    return image, label


def some_of(image, label, augmentation_kwargs_copy):
    random_some = random.randint(0, len(augmentation_kwargs_copy) - 2)
    for i in range(0, random_some):
        rand = random.randint(0, len(augmentation_kwargs_copy) - 2)
        value = augmentation_kwargs_copy[rand]
        image, label = eval(value)(image, label)

    return image, label


def one_of(image, label, augmentation_kwargs_copy):
    rand = random.randint(0, len(augmentation_kwargs_copy) - 2)
    value = augmentation_kwargs_copy[rand]
    image, label = eval(value)(image, label)

    return image, label


def all_of(image, label, augmentation_kwargs_copy):
    for value in augmentation_kwargs_copy:
        image, label = eval(value)(image, label)

    return image, label
