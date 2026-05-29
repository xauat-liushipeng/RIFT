import logging
import os

def get_logger(process_floder_path, name):
    logger = logging.getLogger(name)
    filename = os.path.join(process_floder_path, f"{name}.log")

    fh = logging.FileHandler(filename, mode=''
                                            'w+', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    logger.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
