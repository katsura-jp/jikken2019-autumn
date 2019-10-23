import os
import random
import numpy as np
import torch
import logging

# TODO: Timerの実装

def seed_setting(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class Logger(object):
    def __init__(self, path=None):
        self.logger = logging.getLogger("Log")
        self.level = logging.DEBUG
        self.logger.setLevel(self.level)
        self.handler_format = logging.Formatter('')
        # self.handler_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setLevel(self.level)
        self.stream_handler.setFormatter(self.handler_format)

        self.logger.addHandler(self.stream_handler)

        if path is None:
            self.file_handler = None
        else:
            self.filehandler(path)

    def filehandler(self, path):
        self.file_handler = logging.FileHandler(path)
        self.file_handler.setLevel(self.level)
        self.file_handler.setFormatter(self.handler_format)
        self.logger.addHandler(self.file_handler)

    def log(self, text):
        self.logger.info(text)



