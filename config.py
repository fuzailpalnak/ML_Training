class Config:
    def __init__(self):
        self.model_name = 'unet'
        self.optimizer = 'Adam'
        self.loss_function = ''
        self.train_images_dir = "/home/palnak/Dataset/inria/sample_train/images"
        self.train_labels_dir = "/home/palnak/Dataset/inria/sample_train/labels"
        self.test_images_dir = "/home/palnak/Dataset/inria/sample_train/images"
        self.normalization = "mean_std_over_dataset"
        self.augment = False
        self.augment_frequency = 20
        self.test_loader_batch_size = 4
        self.num_of_multi_label_classes = 2
        self.train_batch_size = 4
        self.model_input_dimension = (500, 500, 3)
        self.image_dimension = (500, 500, 3)
        self.augmentation_kwargs = []
        self.multi_label = False
