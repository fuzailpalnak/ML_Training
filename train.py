from utility.configure import compile_model, configure_training, get_metrics, get_available_gpus_count, \
    configure_data_gen, run_mandatory_check
from utility.parallelizer import make_parallel
from config import Config
from model_utility.model_callbacks import get_callbacks


def main():
    config = Config()
    if not run_mandatory_check(config):
        raise SystemExit
    model, optimizer, loss = configure_training(config.model_name, config.optimizer,
                                                config.loss_function, config.model_input_dimension,
                                                config.num_of_multi_label_classes, lr=1e-4)
    metrics = get_metrics(config.metric)
    model = compile_model(model, optimizer, loss, metrics)
    gpu_count = get_available_gpus_count()

    if gpu_count > 1:
        model = make_parallel(model, gpu_count)
        
    model.summary()
    if config.existing_model_weight.strip():
        model.load_weights(config.existing_model_weight)

    train_data_gen, val_data_gen = configure_data_gen(config)
    train_steps_per_epoch = train_data_gen.get_steps_per_epoch()
    val_steps_per_epoch = val_data_gen.get_steps_per_epoch()

    try:
        model.fit_generator(train_data_gen.generate(), steps_per_epoch=train_steps_per_epoch, workers=gpu_count,
                            epochs=config.epochs, validation_data=val_data_gen.generate(),
                            validation_steps=val_steps_per_epoch, callbacks=get_callbacks(
                multi_label=config.multi_label, training_class=train_data_gen.class_for_training))
    except Exception as ex:
        print(ex)


if __name__ == '__main__':
    main()
