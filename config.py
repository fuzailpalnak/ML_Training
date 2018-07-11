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
        self.num_of_multi_label_classes = 1
        self.batch_size = 2
        self.model_input_dimension = (480, 480, 3)
        self.image_dimension = (500, 500, 3)
        self.augmentation = ['horizontal_flip', 'vertical_flip']
        self.multi_label = False
        self.augmentation_type = 'OneOf'
        self.features = ["void", "Building", "Border"]
        self.color_code = [(0, 0, 0), (0, 255, 0), (255, 0, 0)]
        self.epochs = 200
        self.existing_model_weight = ""
        self.metric = ['accuracy', 'precision', 'recall', 'f_beta_score', 'f_score', 'jaccard_coef']
