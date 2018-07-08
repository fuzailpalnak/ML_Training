mean_over_dataset = None
std_dev_over_dataset = None
calcualted_over_dataset = None
test_images_array = []


def set_mean_over_dataset(mean):
    global mean_over_dataset
    mean_over_dataset=mean


def set_std_dev_over_dataset(std):
    global std_dev_over_dataset
    std_dev_over_dataset=std


def get_mean_over_dataset():
    return mean_over_dataset


def get_std_dev_over_dataset():
    return std_dev_over_dataset


def set_calculate_over_dataset(calcualte):
    global calcualted_over_dataset
    calcualted_over_dataset=calcualte


def get_calculate_over_dataset():
    return calcualted_over_dataset


def set_test_images(test_images):
    global test_images_array
    test_images_array = test_images


def get_test_images():
    return test_images_array


def set_test_images_file_name(test_images_file_name):
    global file_names
    file_names = test_images_file_name


def get_test_images_file_name():
    return file_names