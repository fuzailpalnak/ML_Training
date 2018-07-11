# semantic_segmentation_training


CNN Training Keras 


# Getting Started

### Dataset structure


    ├── train
    │   ├── images          # images to train
    │   ├── labels          # corresponding ground truth
    |
    ├── val
    │   ├── images          # images to validate
    │   ├── labels          # corresponding ground truth
    │
    ├── test
    │   ├── images          # Test files


### Dataset structure if train on seperate region


    ├── train
    │   ├── images 
    |   |    ├── region1          
    |   |    ├── region2
    |   |    └── ...
    │   ├── labels
    |   |    ├── region1          
    |   |    ├── region2
    |   |    └── ...
    ├── val
    │   ├── images 
    |   |    ├── region1          
    |   |    ├── region2
    |   |    └── ...
    │   ├── labels
    |   |    ├── region1          
    |   |    ├── region2
    |   |    └── ...
    ├── test
    │   ├── images 



### Config

    model_name = ''                         # model to run
    optimizer = ''                          # optimizer to choose during model training
    loss_function = ''                      # loss for training
    train_images_dir = "train/images"       # train images dir
    train_labels_dir = "train/labels"       # train labels dir
    val_images_dir = "val/images"           # val images dir
    val_labels_dir = "val/labels"           # val labels dir
    test_images_dir = "test/images"         # test images dir
    normalization = ""                      # normalization to perform on input image
    augment = True                          # to perform augmentation on input image
    augment_frequency = 2
    test_loader_batch_size = 2              # images to load for testing after completion of each epoch
    num_of_multi_label_classes = 1          # number of classes to train
    batch_size = 2                          # batch size to train on
    model_input_dimension = (480, 480, 3)   # input dimension for model
    image_dimension = (500, 500, 3)         # image dimension
    augmentation = []                       # augmentation to perform
    multi_label = False                     # is the training multilabel
    augmentation_type = ''
    features = []                           # if multi_label then what feature to train on
    color_code_for_features = []            # color code of features in multi_label
    epochs = 200                            # number of epoch to train on


### Adding new Architecture

    # Add new architecture in model_utility/models.py

    class ModelInit():
        def __init__(self, ..., ...):
            ...
            ....

        def existing_arch(self):
            ...
            ....
            return model

        # new arch method
        def new_arch(self):
            ...
            ....
            return model

     # For calling the model during training, change the 'model_name' parameter in config.py
     model_name = "new_arch"

### Adding new loss function

    # Add new loss function in model_utility/loss_function.py

    def new_loss_function(y_true, y_pred):
        ....
        .....
        return loss

     # For calling the loss function during training, change the 'loss_function' parameter in config.py
     loss_function = "new_loss_function"

### Adding new normalization

    # Add new normalization in data_utility/data_normalization.py

    def new_normalization(img_array):
        ....
        .....
        return img_array

     # For calling the normalization during training, change the 'normalization' parameter in config.py
     normalization = "new_normalization"


### Train

    python train.py

    To copy terminal output in a log file run following command-
    python train.py | tee terminal_log.log

### Visualizing Training

    tensorboard --logdir="path_to_stored_tboard"

    NOTE - For running tensorboard on aws and visualizing it on local system
           run on local - ssh -i /path/to/your/AWS/key/file -NL 6006:localhost:6006 user@host
           After that you can browse to http://localhost:6006/