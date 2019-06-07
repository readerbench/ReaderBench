import logging
import logging.config
import os 
import json


class Logger():
    logger = None

    @staticmethod
    def get_logger():
        return Logger.set_up()

    @staticmethod   
    def set_up():
        if Logger.logger is None:
            Logger.setup_logging()
            Logger.logger = logging.getLogger() 
            Logger.logger.info('Logger set up done.')
        return Logger.logger

    @staticmethod
    def setup_logging(
        default_path='logging.json',
        default_level=logging.INFO,
        env_key='LOG_CFG'):
        """Setup logging configuration
        """
        path = default_path
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = json.load(f)
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)
