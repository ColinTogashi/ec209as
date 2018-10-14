# TODO: script header
import os
import json
import logging.config

def setup_logging(
    default_path='logging.json', default_level=logging.DEBUG, env_key='LOG_CFG'):
    # setup logging configuration
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

setup_logging()
logger = logging.getLogger('hw2')

if __name__ == "__main__":
    logger.debug('There be debugs in ma code!')
    logger.info('Just for your information.')
    logger.warning('This is a warning.')
    logger.error('Error away!')
    