
#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 09/07/2022
#############################################################################

import pandas as pd
import numpy as np
import os
import ast
import time
import random
from datetime import datetime

class SlideAnnotations():

    def __init__(self,
                MainWindow
                ):
        super(SlideAnnotations, self).__init__()
        self.MainWindow = MainWindow
        
        self.columns=['id','userid','slideDescription','objectClass','shapeType',
                        'coordinates','colorContour','colorFill','fillOpacity',
                        'createTime','isFilled','isPrvate','isDelete']
        self.annotation = pd.DataFrame(columns=self.columns)


    
    def load_annotation_from_database(self):
        if self.MainWindow.imageOpened:
            if hasattr(self.MainWindow, 'user') and hasattr(self.MainWindow.user, 'ID'):
                userID = self.MainWindow.user.ID
            else:
                userID = -1

            slideDescription = self.MainWindow.slideDescription
            mydb = self.MainWindow.connect_to_DB()
            print('nucfinder database connected.')
            mycursor = mydb.cursor()
            sql = "SELECT * FROM Annotation WHERE slideDescription = %s AND (userid = %s OR userid = %s) AND isPrivate = 0 AND isDelete = 0"
            val = (slideDescription, userID, 100000001) # 622084982
            mycursor.execute(sql, val)
            
            for values in mycursor:
                values = np.array(values).reshape(1,-1)
                row = pd.DataFrame(values, columns = self.columns, index=[0])
                self.annotation = pd.concat([self.annotation, row], ignore_index=True)
            mydb.commit()

            print('init_annotation_dataframe', self.annotation)


    def add_annotation(self, ROI_dict):
        '''
        ROI_dict = {'type': self.ui.mode,
                    'points': points,
                    'points_global': points_global,
                    'rotation': self.rotation,
                    'class_ID': int,
                    'class_name': str,
                    'class_rgbcolor': list,
                    }
        '''
        if hasattr(self.MainWindow, 'user') and hasattr(self.MainWindow.user, 'ID'):
            userID = self.MainWindow.user.ID
        else:
            userID = -1
        
        id = ''.join(random.choice('0123456789ABCDEF') for i in range(16))
        coordinates = np.array2string(np.array(ROI_dict['points_global']))
        color = '(%d,%d,%d)' % tuple(ROI_dict['class_rgbcolor'])
        newrow_dict = {'id':id,
                        'userid': userID,
                        'slideDescription': self.MainWindow.slideDescription,
                        'objectClass': ROI_dict['class_name'],
                        'shapeType': ROI_dict['type'],
                        'coordinates': coordinates,
                        'colorContour': color,
                        'colorFill': color,
                        'fillOpacity': 0,
                        'createTime': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'isFilled': 0,
                        'isPrivate': 0,
                        'isDelete': 0}
        newrow = pd.DataFrame(newrow_dict, index=[0])
        self.annotation = pd.concat([self.annotation, newrow], ignore_index=True)

        if userID != -1:
            # sync annotation to database
            mydb = self.MainWindow.connect_to_DB()
            mycursor = mydb.cursor(buffered=True)
            sql = "INSERT INTO Annotation (id, userid, slideDescription, objectClass, shapeType, "+\
                    "coordinates, colorContour, colorFill, fillOpacity, createTime, isFilled, isPrivate, isDelete)"+\
                    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), %s, %s, %s)"
            val = (newrow_dict['id'], newrow_dict['userid'], newrow_dict['slideDescription'], \
                    newrow_dict['objectClass'], newrow_dict['shapeType'], newrow_dict['coordinates'], newrow_dict['colorContour'], \
                    newrow_dict['colorFill'], float(newrow_dict['fillOpacity']), int(newrow_dict['isFilled']), int(newrow_dict['isPrivate']), int(newrow_dict['isDelete'])
                    )
            mycursor.execute(sql, val)
            mydb.commit()
            print('Database:', mycursor.rowcount, "record inserted.")

        pass


    def delete_annotation(self,
                          ROI_dict,
                          ):
        '''
        ROI_dict = {'type': self.ui.mode,
                    'points': points,
                    'points_global': points_global,
                    'rotation': self.rotation,
                    'class_ID': int,
                    'class_name': str,
                    'class_rgbcolor': list,
                    }
        '''
        if hasattr(self.MainWindow, 'user') and hasattr(self.MainWindow.user, 'ID'):
            userID = self.MainWindow.user.ID
        else:
            userID = -1

        if ROI_dict['type'] == 'doubleclick':
            slideDescription = self.MainWindow.slideDescription
            coordinates = np.array2string(np.array(ROI_dict['points_global']))
            idx_to_delete = (self.annotation['slideDescription'].values == slideDescription) & \
                            (self.annotation['coordinates'].values == coordinates)
            print('Found %d annotation to delete.' % np.sum(idx_to_delete))
            self.annotation = self.annotation[~idx_to_delete]

        else:
            pass
        