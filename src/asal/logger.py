import logging
import sys
from simple_colors import *
import os

class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d - %(funcName)s())"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
_formatter = logging.Formatter(
    '%(asctime)s | ''%(filename)s:%(lineno)s - %(funcName)s()'' | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
# stdout_handler.setFormatter(_formatter)
stdout_handler.setFormatter(CustomFormatter())

file_handler = logging.FileHandler('run.log', mode='w')
file_handler.setLevel(logging.DEBUG)
# file_handler.setFormatter(_formatter)
stdout_handler.setFormatter(CustomFormatter())

file_handler.setFormatter(CustomFormatter())

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)
