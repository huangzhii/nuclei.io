#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 07/28/2022
#############################################################################

import traceback
import logging
import os
from datetime import datetime

class Log:
    def __init__(self, parentObject):
        curr_datetime = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        if not os.path.exists(os.path.join(parentObject.wd,'logs')):
            os.makedirs(os.path.join(parentObject.wd,'logs'))
        logging.basicConfig(filename=os.path.join(parentObject.wd,'logs','%s.log' % curr_datetime),
                            format='%(asctime)s.%(msecs)03d *** %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.DEBUG
                            )
    def write(self, message, mode='info'):
        if mode == 'info':
            logging.info(message)
        elif mode == 'debug':
            logging.debug(message)
        elif mode == 'warning':
            logging.warning(message)
        elif mode == 'error':
            logging.error(message)


            
def print_error(e):
    print('Error: ', e)
    traceback.print_exc()