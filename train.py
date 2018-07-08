from utility.configure import compile_model, configure_training, get_metrics, get_available_gpus_count
from utility.parallelizer import make_parallel
from config import Config


def main():
    config = Config()
    model, optimizer, loss = configure_training(config.model_name, config.optimizer, config.loss_function, lr=1e-4)
    metrics = get_metrics(['precision'])
    model = compile_model(model, optimizer, loss, metrics)

    if get_available_gpus_count() > 1:
        model = make_parallel(model, get_available_gpus_count())
        
    print(model.summary())


if __name__ == '__main__':
    main()
