"""
Configure Python logger to work properly in AzureML.
"""

import os
import pathlib
import logging
import sys


_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def _configure_logger(logger, level=None, log_format=_LOG_FORMAT):
    handler_stderr = logging.StreamHandler()
    handler_stderr.setFormatter(logging.Formatter(log_format))
    handler_stderr.setLevel(logging.NOTSET)

    logger.handlers = [handler_stderr]  # Reset the handlers!

    if level is not None:
        logger.setLevel(level)

    return logger


def _log_to_file(logger, fname, level=logging.NOTSET, log_format=_LOG_FORMAT):
    "Add handler that writes log to the file. Creates log path if it does not exist."
    log_dir = os.path.dirname(fname)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

    handler_file = logging.FileHandler(fname)
    handler_file.setFormatter(logging.Formatter(log_format))
    handler_file.setLevel(level)

    logger.addHandler(handler_file)


def get_logger(name, level=logging.DEBUG):
    "Get stderr logger for AzureML tasks"
    logger = logging.getLogger(name)
    type(logger).log_to_file = _log_to_file
    logger.setLevel(level)

    formater = logging.Formatter(fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
                                  datefmt="%Y-%m-%d - %H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    console.setFormatter(formater)

    logger.addHandler(console)

    return logger