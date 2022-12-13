
#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 08/09/2022
#############################################################################

import pandas as pd
import numpy as np
import os
import ast
import time


class User():

    def __init__(self,
                MainWindow
                ):
        super(User, self).__init__()
        self.MainWindow = MainWindow
        self.isLogin = False
        self.ID = -1
        self.email = ''
        self.firstname = ''
        self.lastname = ''
        self.affiliation = ''
        

    def login(self, jsondata):
        print(jsondata)
        self.isLogin = True
        self.ID = int(jsondata['id'])
        self.email = jsondata['email']
        self.firstname = jsondata['firstname']
        self.lastname = jsondata['lastname']
        self.affiliation = jsondata['affiliation']
        self.MainWindow.log.write('User Login *** id=%d; email=%s; firstname=%s; lastname=%s.' % \
                                    (self.ID,
                                    self.email,
                                    self.firstname,
                                    self.lastname)
                                    )

        # update navbar from MainWindow.backend.

        dict2send = {'action':'update login navbar',
                    'userid': str(self.ID),
                     'firstname': jsondata['firstname'],
                     'lastname': jsondata['lastname']}
        self.MainWindow.backend.py2js(dict2send)

        if self.MainWindow.imageOpened:
            self.MainWindow.annotation.load_annotation_from_database()
        


    def logout(self):
        print('User log out.')
        self.isLogin = False
        self.ID = -1
        self.email = ''
        self.firstname = ''
        self.lastname = ''
        self.affiliation = ''