from os.path import exists, join
from os import remove
import logging

logger = None


def create_a_logger_file():
    global logger
    # TODO add path
    log_path = join("", "error.log")
    if exists(log_path):
        remove(log_path)
    logging.basicConfig(filename=log_path,
                        format='%(asctime)s:%(levelname)s:%(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)


def get_logger_object():
    return logger
