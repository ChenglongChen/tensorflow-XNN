
import datetime
import logging
import logging.handlers
import os
import shutil


def _timestamp():
    now = datetime.datetime.now()
    now_str = now.strftime("%Y%m%d%H%M")
    return now_str


def _get_logger(logdir, logname, loglevel=logging.INFO):
    fmt = "[%(asctime)s] %(levelname)s: %(message)s"
    formatter = logging.Formatter(fmt)

    handler = logging.handlers.RotatingFileHandler(
        filename=os.path.join(logdir, logname),
        maxBytes=2 * 1024 * 1024 * 1024,
        backupCount=10)
    handler.setFormatter(formatter)

    logger = logging.getLogger("")
    logger.addHandler(handler)
    logger.setLevel(loglevel)
    return logger


def _makedirs(dir, force=False):
    if os.path.exists(dir):
        if force:
            shutil.rmtree(dir)
            os.makedirs(dir)
    else:
        os.makedirs(dir)