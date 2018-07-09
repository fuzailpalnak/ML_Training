class Config:
    def __init__(self):
        self.model_name = 'unet'
        self.optimizer = 'Adam'
        self.loss_function = 'binary_crossentropy'
        self.train_images_dir = ""
        self.train_labels_dir = ""
        self.val_images_dir = ""
        self.val_labels_dir = ""
        self.test_images_dir = ""
        self.normalization = "std_dev_normalization"
        self.augment = True
        self.augment_frequency = 2
        self.test_loader_batch_size = 2
        self.val_batch_size = 4
        self.num_of_multi_label_classes = 1
        self.batch_size = 2
        self.model_input_dimension = (480, 480, 3)
        self.image_dimension = (500, 500, 3)
        self.augmentation = ['horizontal_flip', 'vertical_flip']
        self.multi_label = False
        self.augmentation_type = 'OneOf'
        self.features = ["Building", "Border"]
        self.color_code_for_features = [(0, 255, 0), (255, 0, 0)]
        self.epochs = 200
