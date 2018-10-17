#!/usr/bin/env python
'''Module to set up logging functionality based on logging.json.'''

import os
import json
import logging.config

def setupLogging(
    default_path='logging.json', default_level=logging.DEBUG, env_key='LOG_CFG'):
    '''Setup logging configuration
    
    Arguments:
        default_path - string: path/name of .json config file
        default_level - int: the default level of output to the file
        env_key - string: operating system variable to globally define logging location
    '''
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



if __name__ == "__main__":
    # create logger
    setupLogging()
    logger = logging.getLogger('main')

    # test logging capabilities
    logger.debug('There be debugs in ma code!')
    logger.info('Just for your information.')
    logger.warning('This is a warning.')
    logger.error('Error away!')
    