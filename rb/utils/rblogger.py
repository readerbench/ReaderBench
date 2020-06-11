import json
import logging
import logging.config
import os


class Logger():
    logger = None

    @staticmethod
    def get_logger() -> logging.Logger:
        return Logger.set_up()

    @staticmethod   
    def set_up() -> logging.Logger:
        if Logger.logger is None:
            Logger.setup_logging()
            Logger.logger = logging.getLogger(__name__) 
            Logger.logger.info('Logger set up done.')
            
        for v in logging.Logger.manager.loggerDict.values():
            if isinstance(v, logging.Logger) and not v.name.startswith('rb'):
                v.disabled = True

        return Logger.logger

    @staticmethod
    def setup_logging(
        default_path='logging.json',
        default_level=logging.INFO,
        env_key='LOG_CFG'):
        """Setup logging configuration """
        path = default_path
        if os.path.exists(path):
            with open(path, 'rt') as f:
                config = json.load(f)
            logging.config.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)
        
        

            #if v.name.startswith('rb'):
            # 0   v.disabled = True
