import logging
import logging.handlers as handlers
import utils.file_tool as file_tool
import math
import sys


def get_model_result_logger():

    name = 'model_result_logger'
    if name not in logging.Logger.manager.loggerDict:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        file_handler = handlers.RotatingFileHandler(filename=file_tool.PathManager.model_running_data_log_file, maxBytes=math.pow(2,30), backupCount=10)
        console_handler = logging.StreamHandler(stream=sys.stderr)

        file_handler.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)

        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    else:
        logger = logging.getLogger(name)

    return logger

model_result_logger =get_model_result_logger()
# logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)