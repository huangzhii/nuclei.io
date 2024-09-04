#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 07/25/2022
#############################################################################

from PySide6.QtCore import QObject, Property, Signal, QDir, QFile, QIODevice, QUrl, Qt, Slot
from PySide6.QtWidgets import (QWidget, QDialog, QFileDialog, QMainWindow, QMessageBox)

import numpy as np
import pandas as pd
import json

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb

class Backend(QObject):
    #tutorial:https://www.cnblogs.com/aloe-n/p/10052830.html
    sendjson_to_js = Signal(str)

    def __init__(self, engine=None, MainWindow=None):
        super().__init__(engine)
        self.MainWindow = MainWindow

    @Slot(str)
    def log(self, logstring):
        print(logstring)

    @Slot()
    def open_local_slide_dialog(self):
        self.MainWindow.onFileOpen()

    @Slot()
    def open_local_case_dialog(self):
        self.MainWindow.onCaseFolderOpen()

    @Slot(str, str)
    def open_local_slide(self, case_id, slide_id):
        self.MainWindow.SVS_clicked(case_id, slide_id)

    @Slot()
    def clear_user_profile(self):
        self.MainWindow.user.logout()
    

    @Slot(str)
    def py2js(self, dict):
        if 'action' in dict:
            if dict['action'] == 'update_workspace':
                dict2js = {}
                dict2js['case_id'] = dict['case_id']
                dict2js['action'] = 'update_workspace'
                dict2js['case_dict'] = {}
                if 'assist_type' in dict:
                    dict2js['assist_type'] = dict['assist_type']
                for d in dict['case_dict']:
                    dict2js['case_dict'][d] = {}
                    for val in ['svs_fname','thumbnail_small_base64','label_base64','magnitude','height','width',
                                'month','day','year','createtime','timezone']:
                        dict2js['case_dict'][d][val] = dict['case_dict'][d][val]
                jsonarray = json.dumps(dict2js)

            elif dict['action'] == 'apply2case_done':
                new_dict = {}
                new_dict['action'] = 'apply2case_done'
                new_dict['accuracy'] = '--'
                if 'data_dict' in dict: # self.ML_result
                    if 'test_acc' not in dict['data_dict'] or 'Apply_all' not in dict['data_dict']['test_acc'] or dict['data_dict']['test_acc']['Apply_all'] == 0:
                        new_dict['accuracy'] = '--'
                    else:
                        new_dict['accuracy'] = '%.2f%%' % (dict['data_dict']['test_acc']['Apply_all']*100)
                print('-----------------', new_dict['accuracy'])
                jsonarray = json.dumps(new_dict)

            else:
                jsonarray = json.dumps(dict)
        return self.sendjson_to_js.emit(jsonarray)


    @Slot(str, str, str)
    def set_active_class(self, class_id, class_name, class_color):
        print('----------------------')
        print('Active class: ')
        print('class_id:',class_id)
        print('class_name:',class_name)
        print('class_color:', class_color)
        print('----------------------')
        class_id = int(class_id)
        self.MainWindow.datamodel.setActiveClassID(class_id)

    @Slot(str, str)
    def update_classinfo(self, jsondata, action):
        jsondata = json.loads(jsondata)
        classinfo = pd.DataFrame(jsondata).T
        classinfo.index = classinfo.index.astype(int)
        self.MainWindow.datamodel.updateClassInfo(classinfo, action)


    @Slot()
    def apply_to_case(self):
        self.MainWindow.datamodel.apply_to_case()

    @Slot(str)
    def update_vfc(self, dim_info_json):
        dim_info = json.loads(dim_info_json)

        if 'dim_1_select' in dim_info and 'dim_2_select' in dim_info:
            dim1_text = dim_info['dim_1_select'].split(' | ')
            dim2_text = dim_info['dim_2_select'].split(' | ')
            self.MainWindow.datamodel.send_dim_info_for_VFC(dim1_text, dim2_text)
        else:
            self.MainWindow.datamodel.send_dim_info_for_VFC()

    @Slot(str)
    def import_nucfinder(self, dataset_id):
        self.MainWindow.datamodel.import_nucfinder_dataset(dataset_id)

    @Slot()
    def open_user_login_window(self):
        self.MainWindow.webdialog.show_dialog('login')


    @Slot(str)
    def sync(self, jsondata):
        print('**********************')
        print('Start syncing ...')
        print('**********************')
        jsondata = json.loads(jsondata)
        self.MainWindow.log.write('Interactive label: Sync start.')
        if not hasattr(self.MainWindow.datamodel, 'data_info'):
            exception = QMessageBox.information(self.MainWindow,
                                                        'Notice',
                                                        'self.MainWindow.datamodel.data_info is not exist!',
                                                        QMessageBox.Ok)
            return
        status = self.MainWindow.datamodel.sync_to_db(self.MainWindow, jsondata)
        self.MainWindow.log.write('Interactive label: Sync finished.')
        if status == 'success':
            print('Insert to DB success.')
        self.MainWindow.backend.py2js({'action': 'syncing_done'})


    @Slot()
    def review_annotations(self):
        self.MainWindow.review_annotations()
        
        '''
        if hasattr(self.MainWindow.datamodel, 'data_info') and \
            (len(self.MainWindow.datamodel.data_info) > 0):
            self.MainWindow.review_annotations()
        else:
            exception = QMessageBox.information(self.MainWindow,
                                                'Notice',
                                                'No labeled dataset.',
                                                QMessageBox.Ok)
            return
        '''

    @Slot(str)
    def force_update(self, status):
        if status == 'Enable':
            self.MainWindow.datamodel.force_update_ML = True
        else:
            self.MainWindow.datamodel.force_update_ML = False
        return
    
    @Slot()
    def save_model_offline(self):
        print('save_model_offline')
        self.MainWindow.datamodel.save_model_offline()
        return
    
    @Slot()
    def use_offline_model(self):
        print('use_offline_model')
        self.MainWindow.datamodel.load_offline_model()
        return