# ML_Training

Semantic Segmentation training Keras


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

        model_name = ''                         # model to run form
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


### Train

    python train.py

    To copy terminal output in a log file run following command-
    python train.py | tee terminal_log.log

### Visualizing Training

    tensorboard --logdir="path_to_stored_tboard"

    NOTE - For running tensorboard on aws and visualizing it on local system
           run on local - ssh -i /path/to/your/AWS/key/file -NL 6006:localhost:6006 user@host
           After that you can browse to http://localhost:6006/