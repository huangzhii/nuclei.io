#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 08/10/2022
#############################################################################
from PySide6.QtCore import (QDir, QFile, QIODevice, QUrl, Qt, Slot, Signal, QPointF,
    QPoint, QEvent, QObject, QThread)
    
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient, QWheelEvent,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient, QPen,
    QTransform)
    
from PySide6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QGridLayout, QStackedLayout, QMainWindow, QMenu, QLabel,
    QMenuBar, QPlainTextEdit, QSizePolicy, QSplitter, QScrollBar, QProgressBar, QPushButton, QDial,
    QStatusBar, QWidget, QMessageBox)

from matplotlib import path
import pandas as pd
import numpy as np
import os
import re
import ast
import time
import random
from datetime import datetime
from gui.utils import im_2_b64
#from gui.uitils import parmap
from PIL import Image
from collections import Counter

import pickle
import copy
import xgboost as xgb
import sklearn
from tqdm import tqdm
import paramiko
import platform
import traceback
opj = os.path.join


class AsyncApplytoCase(QObject):
    finished = Signal()
    progress = Signal(int)
    def __init__(self, MainWindow):
        super(AsyncApplytoCase, self).__init__()
        self.MainWindow = MainWindow
        
    def run(self):
        """Long-running task."""
        self.MainWindow.datamodel.apply2case()
        print('Apply async job finished.')
        self.finished.emit()

    def stop(self):
        self.terminate()
        print("QThread terminated")
        self.MainWindow.log.write('Async apply to case *** QThread terminated.')



class DataModel():
    
    classinfo = None
    activeClassID = None
    activeClassName = None
    activeClassRGB = None
    force_update_ML = True

    data_X = None
    data_info = None
    data_info_columns = ['case_id', # str
                        'slide_id', # str
                        'filepath', # str
                        'original_class_id', # int
                        'nuclei_index', # int
                        'centroid_x', # int
                        'centroid_y', # int
                        'contour', # str: [[1,2], [3,4], [5,6]]
                        'label_type', # doubleclick, active_learning, region_annotation
                        'AL_epoch', # int
                        'label_toggle_count', # int
                        'label_init_value', # int
                        'label_final_value', # int
                        'label_init_datetime', # str: %Y-%m-%d %H:%M:%S.%f
                        'label_final_datetime', # str: %Y-%m-%d %H:%M:%S.%f
                        'base64string_40x_256x256', # str
                        'base64string_20x_256x256', # str
                        'base64string_10x_256x256', # str
                        'base64string_5x_256x256', # str
                        'base64string_2.5x_256x256', # str
                        'base64string_1.25x_256x256', # str
                        'base64string_0.6125x_256x256', # str
                        'other_information' # str
                        ]

    def __init__(self,
                MainWindow,
                ):
        self.MainWindow = MainWindow

        self.initData()

    def initData(self):
        self.data_info = pd.DataFrame(columns = self.data_info_columns)
        self.dataset_id = ''.join(random.choice('0123456789ABCDEF') for i in range(16))

    def updateClassInfo(self,
                        classinfo: pd.DataFrame,
                        action: str):
        print('****** update_classinfo ******')
        print(classinfo)
        self.classinfo = classinfo

        # When class name/color changed, we should also update active class name and RGB:
        if self.activeClassID is not None:
            self.setActiveClassID(self.activeClassID)
        else:
            # maybe just initialize the classinfo, there is no active class ID import yet.
            pass

        # If a class is deleted, we need to remove the associated data_info and data_X:
        previous_uniq_id = self.data_info['original_class_id'].unique().astype(int)
        for id in previous_uniq_id:
            if id not in self.classinfo.index:

                print(id, self.classinfo.index)
                bool_idx_exist = (self.data_info['original_class_id'].values == id)
                self.data_info = self.data_info.loc[~bool_idx_exist,:]
                self.data_X = self.data_X.loc[~bool_idx_exist,:]
                print('Remove ID=%d (total %d nuclei annotation removed).' % (id, np.sum(bool_idx_exist)))

        if action == 'update color':
            # after update class info, refresh ML overlay.
            self.ML_apply_all()
            self.overlay_create()

        return
    
    def setActiveClassID(self,
                         class_id: int):
        if hasattr(self, 'classinfo'):
            classname, rgbcolor, isVisible = self.classinfo.loc[class_id,:]
            self.activeClassID = class_id
            self.activeClassName = classname
            self.activeClassRGB = rgbcolor
        else:
            print('Warning: No classinfo found.')

    def annotate_ROI_from_prediction(self,
                                     ROI_dict = None,
                                     testing_ratio = 0.3
                                     ):
        if not hasattr(self.MainWindow, 'nucstat') or self.MainWindow.nucstat is None:
            print('No nucstat! Annotation exit.')
            return
        print('Start annotate ROI from current prediction.')

        ROI_type = ROI_dict['type']
        ROI_points = ROI_dict['points']
        ROI_points_global = ROI_dict['points_global']
        ROI_rotation = ROI_dict['rotation']
        
        if ROI_type == 'Polygon':
            # clear current polygon drawing
            if hasattr(self.MainWindow.ui, 'drawPolygonPoints'):
                delattr(self.MainWindow.ui, 'drawPolygonPoints')
            self.MainWindow.ui.polygon_in_progress = False

        if ROI_type == 'Rect':

            [[x1, y1], [x2, y2]] = ROI_points_global
            x1, x2 = np.sort([x1, x2])
            y1, y2 = np.sort([y1, y2])

            subset_index = (self.MainWindow.nucstat.centroid[:,0] > x1) & (self.MainWindow.nucstat.centroid[:,0] < x2) & \
                            (self.MainWindow.nucstat.centroid[:,1] > y1) & (self.MainWindow.nucstat.centroid[:,1] < y2)

        elif ROI_type == 'Polygon':
            p = path.Path(ROI_points_global)
            subset_index = p.contains_points(self.MainWindow.nucstat.centroid)

        subset_index = np.where(subset_index)[0]
        subset_index_vals = self.MainWindow.nucstat.index[subset_index]

        if len(subset_index) > 10000:
            reply = QMessageBox.information(self, 'Notification',
                                        'Warning: You have selected a very large region with over 10,000 nuclei, we suggest you to choose a smaller ROI for annotation.',
                                        QMessageBox.Ok)
            return


        bool_idx_exist = (self.data_info['case_id'].values == self.MainWindow.case_id) & \
                        (self.data_info['slide_id'].values == self.MainWindow.slide_id) & \
                        np.isin(self.data_info['nuclei_index'].values, subset_index_vals)

        if np.sum(bool_idx_exist) > 0:
            '''
            If ROI selection overlaps with previous nuclei index, then just overwrite it.
            Before overwrite, remove all previous nuclei annotations.
            '''
            self.data_X = self.data_X.loc[~bool_idx_exist,:]
            self.data_info = self.data_info.loc[~bool_idx_exist,:]


        label_type_list = np.repeat('ROI_training', len(subset_index)).astype(object)

        if len(subset_index) > 20:
            # only enable testing when more than 20 nuclei annotated.
            if testing_ratio > 0:
                boollist_isTesting = np.random.sample(len(subset_index)) < testing_ratio
            else:
                boollist_isTesting = np.ones(len(subset_index)).astype(bool)
            label_type_list[boollist_isTesting] = 'ROI_random_testing'
        
        self.MainWindow.ui.statusBar.statusbar_pbar.setValue(0)
        # parmap seems not working well.
        #new_rows = parmap(lambda tuple_in: self.add_data_row(tuple_in[0], tuple_in[1], class_id),
        #                    [(ix, lbl) for ix, lbl in zip(subset_index, label_type_list)]
        #                    )
        new_rows = []
        class_ids = []
        for idx, lbl in zip(subset_index, label_type_list):
            # Get curren prediction
            class_id = int(self.MainWindow.nucstat.prediction.loc[idx,'label'])
            class_ids.append(class_id)
            row = self.add_data_row(idx, lbl, class_id)
            new_rows.append(row)

        new_rows = pd.concat(new_rows, axis=0, ignore_index=True)
        


        self.data_info = pd.concat([self.data_info, new_rows], axis=0, ignore_index=True)

        learning_feature_idx = []
        for v in self.MainWindow.nucstat.feature_columns:
            learning_feature_idx.append(v in self.MainWindow.nucstat.learning_features)
        learning_feature_idx = np.array(learning_feature_idx)
        tuple_indices = [(self.MainWindow.case_id, self.MainWindow.slide_id, class_id, idx) for idx, class_id in zip(subset_index, class_ids)]
        feat = pd.DataFrame(self.MainWindow.nucstat.feature[np.isin(self.MainWindow.nucstat.index, subset_index_vals), :][:, learning_feature_idx], index=tuple_indices) # M_nuclei x N_features            
        
        if not hasattr(self, 'data_X') or self.data_X is None:
            self.data_X = feat
        else:
            self.data_X = pd.concat([self.data_X, feat], axis=0)


        print('%d new nuclei annotated.' % len(new_rows))

        self.MainWindow.ui.statusBar.statusbar_pbar.setValue(100)


        for idx, class_id in zip(subset_index, class_ids):
            new_ROI_dict = {}
            new_ROI_dict['type'] = 'doubleclick'
            new_ROI_dict['class_ID'] = class_id
            new_ROI_dict['points_global'] = self.MainWindow.nucstat.contour[idx]
            new_ROI_dict['class_name'] = self.classinfo.loc[class_id, 'classname']
            new_ROI_dict['class_rgbcolor'] = self.classinfo.loc[class_id, 'rgbcolor']
            self.MainWindow.annotation.add_annotation(new_ROI_dict)
        self.updateAnnotationToWebEngine()

        # clear drawing
        pixmap = QPixmap(self.MainWindow.ui.DrawingOverlay.size())
        pixmap.fill(Qt.transparent)
        self.MainWindow.ui.DrawingOverlay.setPixmap(pixmap)
        if self.force_update_ML:
            self.apply_to_case()

    def activeLearning(self,
                        ROI_dict = None):
        if not hasattr(self.MainWindow, 'nucstat') or self.MainWindow.nucstat is None:
            print('No nucstat! Annotation exit.')
            return
        print('Start active learning.')

        pass


    def analyzeROI(self,
                        ROI_dict = None):
        if not hasattr(self.MainWindow, 'nucstat') or self.MainWindow.nucstat is None:
            print('No nucstat! Annotation exit.')
            return
        print('Start analyze ROI.')

        ROI_type = ROI_dict['type']
        ROI_points = ROI_dict['points']
        ROI_points_global = ROI_dict['points_global']
        ROI_rotation = ROI_dict['rotation']
        
        if ROI_type == 'Polygon':
            # clear current polygon drawing
            if hasattr(self.MainWindow.ui, 'drawPolygonPoints'):
                delattr(self.MainWindow.ui, 'drawPolygonPoints')
            self.MainWindow.ui.polygon_in_progress = False


        if ROI_type == 'Rect':

            [[x1, y1], [x2, y2]] = ROI_points_global
            x1, x2 = np.sort([x1, x2])
            y1, y2 = np.sort([y1, y2])

            index_bool = (self.MainWindow.nucstat.centroid[:,0] > x1) & (self.MainWindow.nucstat.centroid[:,0] < x2) & \
                            (self.MainWindow.nucstat.centroid[:,1] > y1) & (self.MainWindow.nucstat.centroid[:,1] < y2)

        elif ROI_type == 'Polygon':
            p = path.Path(ROI_points_global)
            index_bool = p.contains_points(self.MainWindow.nucstat.centroid)

        # get all nuclei within ROI

        self.MainWindow.nucstat.isSelected_to_VFC = index_bool
        self.send_dim_info_for_VFC()
        # clear drawing
        pixmap = QPixmap(self.MainWindow.ui.DrawingOverlay.size())
        pixmap.fill(Qt.transparent)
        self.MainWindow.ui.DrawingOverlay.setPixmap(pixmap)

        pass
    
    def IHC_evaluation(self,
                       annotation_dict=None, 
                       tissue_type=None,
                       cell_type=None,
                       antibody=None,
                       additional_info=None,
                       ):
        print('Start INTERPRET IHC ROI.')

        ROI_type = annotation_dict['type']
        ROI_points = annotation_dict['points']
        ROI_points_global = annotation_dict['points_global']
        ROI_rotation = annotation_dict['rotation']

        if ROI_type != 'Rect':
            return

        [[x1, y1], [x2, y2]] = ROI_points_global
        x1, x2 = np.sort([x1, x2])
        y1, y2 = np.sort([y1, y2])

        if hasattr(self.MainWindow, 'nucstat'):
            # get all nuclei within ROI
            index_bool = (self.MainWindow.nucstat.centroid[:,0] > x1) & (self.MainWindow.nucstat.centroid[:,0] < x2) & \
                            (self.MainWindow.nucstat.centroid[:,1] > y1) & (self.MainWindow.nucstat.centroid[:,1] < y2)

            self.MainWindow.nucstat.isSelected_to_VFC = index_bool

        time.sleep(10)
        breakpoint()
        # the slide object: self.MainWindow.slide
        # the selected region: x1, x2, y1, y2
        selected_region = self.MainWindow.slide.read_region(location=(x1,y1), level=0, size=(x2-x1,y2-y1), as_array=True)
        print("Selected region shape: ", selected_region.shape)
        # @Jake: Now, given the selected_region, run IHC evaluation, return embeddding, dict of results.
        # below is fake output:
        # Run vector database search, get the closest top 10 matches in weblinks.

        similar_weblinks = [
            {
                'image_url': 'https://images.proteinatlas.org/60655/137304_B_7_5.jpg',
                'page_url': 'https://www.proteinatlas.org/ENSG00000111602-TIMELESS/tissue/cerebral+cortex#img'
            },
            {
                'image_url': 'https://images.proteinatlas.org/3387/11301_B_6_3.jpg', 
                'page_url': 'https://www.proteinatlas.org/ENSG00000170312-CDK1/cancer/pancreatic+cancer#img'
            },
            # Add more similar results as needed
        ]

        whole_region_embedding = np.random.rand(7*7, 512)
        whole_region_results = {'staining_intensity': "moderate",
                                'staining_location': "nuclear",
                                'staining_quantity': "25-75%",
                                'tissue_type': "Breast",
                                'cancerous': 'cancer',
                                'similar_weblinks': similar_weblinks,
                                }

        
        print('IHC evaluation done.')
        # Finally, send the evaluation result to the web engine for visualization.
        dict2send = {"action": "show_IHC_evaluation_result",
                     "data": whole_region_results}
        self.MainWindow.backend.py2js(whole_region_results) # send total nuclei count, active nuclei count, accuracy, etc.



        # clear drawing
        pixmap = QPixmap(self.MainWindow.ui.DrawingOverlay.size())
        pixmap.fill(Qt.transparent)
        self.MainWindow.ui.DrawingOverlay.setPixmap(pixmap)

    def annotate_all_nuclei_within_ROI(self,
                                        ROI_dict = None,
                                        class_id = None,
                                        testing_ratio=0.3
                                        ):
        '''
        ROI_dict = {'type': self.ui.mode,
                            'points': points,
                            'points_global': points_global,
                            'rotation': self.rotation}
        '''
        if not hasattr(self.MainWindow, 'nucstat') or self.MainWindow.nucstat is None:
            print('No nucstat! Annotation exit.')
            return

        ROI_type = ROI_dict['type']
        points_global = ROI_dict['points_global']

        if ROI_type == 'Polygon':
            # clear current polygon drawing
            if hasattr(self.MainWindow.ui, 'drawPolygonPoints'):
                delattr(self.MainWindow.ui, 'drawPolygonPoints')
            self.MainWindow.ui.polygon_in_progress = False


        if ROI_type == 'Rect':
            [(x1_temp, y1_temp), (x2_temp, y2_temp)] = points_global
            
            x1 = np.min((x1_temp, x2_temp))
            x2 = np.max((x1_temp, x2_temp))
            y1 = np.min((y1_temp, y2_temp))
            y2 = np.max((y1_temp, y2_temp))

            subset_index = (self.MainWindow.nucstat.centroid[:,0] > x1) & (self.MainWindow.nucstat.centroid[:,0] < x2) & \
                            (self.MainWindow.nucstat.centroid[:,1] > y1) & (self.MainWindow.nucstat.centroid[:,1] < y2)

        elif ROI_type == 'Ellipse':
            #TODO
            pass
            
        elif ROI_type == 'Polygon':
            p = path.Path(points_global)
            subset_index = p.contains_points(self.MainWindow.nucstat.centroid)

        subset_index = np.where(subset_index)[0]
        subset_index_vals = self.MainWindow.nucstat.index[subset_index]

        if len(subset_index) > 10000:
            reply = QMessageBox.information(self, 'Notification',
                                        'Warning: You have selected a very large region with over 10,000 nuclei, we suggest you to choose a smaller ROI for annotation.',
                                        QMessageBox.Ok)
            return

        bool_idx_exist = (self.data_info['case_id'].values == self.MainWindow.case_id) & \
                        (self.data_info['slide_id'].values == self.MainWindow.slide_id) & \
                        np.isin(self.data_info['nuclei_index'].values, subset_index_vals)

        if np.sum(bool_idx_exist) > 0:
            '''
            If ROI selection overlaps with previous nuclei index, then just overwrite it.
            Before overwrite, remove all previous nuclei annotations.
            '''
            self.data_X = self.data_X.loc[~bool_idx_exist,:]
            self.data_info = self.data_info.loc[~bool_idx_exist,:]


        label_type_list = np.repeat('ROI_training', len(subset_index)).astype(object)

        if len(subset_index) > 20:
            # only enable testing when more than 20 nuclei annotated.
            if testing_ratio > 0:
                boollist_isTesting = np.random.sample(len(subset_index)) < testing_ratio
            else:
                boollist_isTesting = np.ones(len(subset_index)).astype(bool)
            label_type_list[boollist_isTesting] = 'ROI_random_testing'
        
        self.MainWindow.ui.statusBar.statusbar_pbar.setValue(0)
        # parmap seems not working well.
        #new_rows = parmap(lambda tuple_in: self.add_data_row(tuple_in[0], tuple_in[1], class_id),
        #                    [(ix, lbl) for ix, lbl in zip(subset_index, label_type_list)]
        #                    )
        new_rows = []
        for idx, lbl in zip(subset_index, label_type_list):
            row = self.add_data_row(idx, lbl, class_id)
            new_rows.append(row)

        new_rows = pd.concat(new_rows, axis=0, ignore_index=True)
        


        self.data_info = pd.concat([self.data_info, new_rows], axis=0, ignore_index=True)

        learning_feature_idx = []
        for v in self.MainWindow.nucstat.feature_columns:
            learning_feature_idx.append(v in self.MainWindow.nucstat.learning_features)
        learning_feature_idx = np.array(learning_feature_idx)
        tuple_indices = [(self.MainWindow.case_id, self.MainWindow.slide_id, class_id, idx) for idx in subset_index]
        feat = pd.DataFrame(self.MainWindow.nucstat.feature[np.isin(self.MainWindow.nucstat.index, subset_index_vals), :][:, learning_feature_idx], index=tuple_indices) # M_nuclei x N_features            
        
        if not hasattr(self, 'data_X') or self.data_X is None:
            self.data_X = feat
        else:
            self.data_X = pd.concat([self.data_X, feat], axis=0)


        print('%d new nuclei annotated.' % len(new_rows))

        self.MainWindow.ui.statusBar.statusbar_pbar.setValue(100)

        ROI_dict['class_ID'] = class_id
        ROI_dict['class_name'] = self.classinfo.loc[class_id, 'classname']
        ROI_dict['class_rgbcolor'] = self.classinfo.loc[class_id, 'rgbcolor']
        self.MainWindow.annotation.add_annotation(ROI_dict)
        self.updateAnnotationToWebEngine()

        # clear drawing
        pixmap = QPixmap(self.MainWindow.ui.DrawingOverlay.size())
        pixmap.fill(Qt.transparent)
        self.MainWindow.ui.DrawingOverlay.setPixmap(pixmap)

        if self.force_update_ML:
            self.apply_to_case()
        return


    def add_data_row(self,
                    idx: int, # a value index of nucstat. Not the nucstat.index.
                    label_type: str,
                    class_id: int,
                    ):
        idx_real = self.MainWindow.nucstat.index[idx]
        centroid = self.MainWindow.nucstat.centroid[idx,:]
        contour = self.MainWindow.nucstat.contour[idx,:]
        contour = contour[np.sum(contour, axis=1)>0,:] # keep valid contour, remove 0
        contour_str = np.array2string(contour)
        
        new_row = {'case_id': self.MainWindow.case_id,
                    'slide_id': self.MainWindow.slide_id,
                    'filepath': self.MainWindow.slide_filepath,
                    'original_class_id': class_id,
                    'nuclei_index': idx_real,
                    'centroid_x': centroid[0],
                    'centroid_y': centroid[1],
                    'contour': contour_str,
                    'label_type': label_type,
                    'AL_epoch': np.nan,
                    'label_toggle_count': 0,
                    'label_init_value': np.nan,
                    'label_final_value': 1,
                    'label_init_datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    'label_final_datetime': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                    'other_information': np.nan
                    }

        '''
        It'd be nice to store nuclei image into base64 string, for cross-site machine learning.
        Is it okay to store nuclei image into base64 string?

        256x256 base64 string is about 18000 - 20000 characters. When 4 bytes per char, then total 0.08 MB per nuclei.
        I tried to store 17 nuclei with base64string_40x_256x256 on record only, result in 330 KB in pickle file.
        If 1K nuclei, then data_info size should be 19 MB.
        If 10K nuclei, then data_info size should be 190 MB.
        If 100K nuclei, then data_info size should be 1.9 GB.
        Overall I think it is affordable for system memory if less than 100K nuclei annotated.
        '''

        '''
        width, height = 256, 256
        for zoom_str in ['40x']: #, '20x','10x','5x','2.5x','1.25x','0.6125x']:
            zoom_val = float(zoom_str.rstrip('x'))
            x1, y1 = int(np.round(centroid[0] - width/2)), int(np.round(centroid[1] - height/2))
            img_patch = self.MainWindow.slide.read_region(location=(x1,y1), level=0, size=(width,height), as_array=True)
            #img_patch = Image.fromarray(img_patch[..., 0:3])
            #img_base64str = im_2_b64(img_patch)
            #new_row['base64string_%s_256x256' % zoom_str] = img_base64str

            #Alternatively, wrap image into a list, and save to pandas DataFrame:
            new_row['base64string_%s_256x256' % zoom_str] = [img_patch[..., 0:3]]
        '''
        new_row = pd.DataFrame(new_row, index=[0])
        return new_row




    def annotate_single_nuclei(self,
                                posx,
                                posy,
                                cx,
                                cy,
                                class_id = None):
        '''
        posx, posy: coordinates on the canvas.
        cx, cy: coordinates on the WSI.
        '''
        if not hasattr(self.MainWindow, 'nucstat') or self.MainWindow.nucstat is None:
            print('No nucstat! Annotation exit.')
            return
        
        if not hasattr(self, 'activeClassID') or self.activeClassID is None:
            print('No active class selected.')
            return

        if class_id is None:
            class_id = self.activeClassID
        # find nearest nucleus:

        distance_to_mouse = (self.MainWindow.nucstat.centroid[:,0]-cx)**2 + (self.MainWindow.nucstat.centroid[:,1]-cy)**2
        if np.min(distance_to_mouse) > 400: # 400 pixel, about 100 micron.
            print('No closest nuclei within 400 pixel, return.')
            return

        # Indices have been reset, and are continuous from 0 to N-1.
        idx = np.argmin(distance_to_mouse)
        mindist_idx = self.MainWindow.nucstat.index[idx]

        global_pts = self.MainWindow.nucstat.contour[idx,:]
        local_pts = [self.MainWindow.slideToScreen((posx, posy)) for posx, posy in global_pts]
        ROI_dict = {'type': 'doubleclick', 'points': local_pts, 'points_global': global_pts, 'rotation': self.MainWindow.rotation}
        ROI_dict['class_ID'] = class_id
        ROI_dict['class_name'] = self.classinfo.loc[class_id, 'classname']
        ROI_dict['class_rgbcolor'] = self.classinfo.loc[class_id, 'rgbcolor']


        print('double clicked for interactive labeling. screen: (%d, %d); slide: (%d, %d). Nuclei index selected: %d' % (posx, posy, cx, cy, mindist_idx))
        self.MainWindow.log.write('Interactive label *** (double click) click on screen. screen: (%d, %d); slide: (%d, %d). case_id: %s, slide_id: %s, Nuclei index selected: %d' % (posx, posy, cx, cy, self.MainWindow.case_id, self.MainWindow.slide_id, mindist_idx))

        tuple_idx = (self.MainWindow.case_id, self.MainWindow.slide_id, class_id, mindist_idx)
        bool_idx_exist = (self.data_info['case_id'].values == self.MainWindow.case_id) & \
                        (self.data_info['slide_id'].values == self.MainWindow.slide_id) & \
                        (self.data_info['nuclei_index'].values == mindist_idx)
    
        if np.sum(bool_idx_exist) > 0:
            if class_id == self.data_info.loc[bool_idx_exist, 'original_class_id'].values[0]:
                '''
                If that existed annotation has same labeling class, then we remove that nuclei.
                '''
                self.data_X = self.data_X.loc[~bool_idx_exist,:]
                self.data_info = self.data_info.loc[~bool_idx_exist,:]
                # Delete annotation (polygon)
                self.MainWindow.annotation.delete_annotation(ROI_dict)
                self.MainWindow.showOverlayAnnotation()
            else:
                '''
                If that existed annotation has different labeling class, then we update that nuclei.
                '''
                self.data_info.loc[bool_idx_exist,'original_class_id'] = class_id
                self.data_info.loc[bool_idx_exist,'label_toggle_count'] += 1
                self.data_info.loc[bool_idx_exist,'label_final_datetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                # Modify annotation (polygon)
                self.MainWindow.annotation.delete_annotation(ROI_dict)
                self.MainWindow.annotation.add_annotation(ROI_dict)
                self.MainWindow.showOverlayAnnotation()

        else:
            new_row = self.add_data_row(idx, label_type='doubleclick', class_id=class_id)
            self.data_info = pd.concat([self.data_info, new_row], axis=0, ignore_index=True)
            learning_feature_idx = []
            for v in self.MainWindow.nucstat.feature_columns:
                learning_feature_idx.append(v in self.MainWindow.nucstat.learning_features)
            learning_feature_idx = np.array(learning_feature_idx)
            feat = pd.DataFrame(self.MainWindow.nucstat.feature[mindist_idx == self.MainWindow.nucstat.index, :][:, learning_feature_idx], index=[tuple_idx]) # 1 x N_features            
            if not hasattr(self, 'data_X') or self.data_X is None:
                self.data_X = feat
            else:
                self.data_X = pd.concat([self.data_X, feat], axis=0)
            
            # Add annotation (polygon)
            self.MainWindow.annotation.add_annotation(ROI_dict)
            self.MainWindow.showOverlayAnnotation()


        self.updateAnnotationToWebEngine()
        if self.force_update_ML:
            self.apply_to_case()
    
    def updateAnnotationToWebEngine(self):
        dict_to_send = {'action': 'update annotation count',
                        'value': dict(Counter(self.data_info['original_class_id']))
                        }

        self.MainWindow.backend.py2js(dict_to_send)



    def get_merged_lbl_data(self,
                            stage='',
                            force_retrain=False
                            ):

        '''
        replace NA by 0
        '''
        X = self.data_X.values.astype(float)
        y = copy.deepcopy(self.data_info[['case_id','slide_id','original_class_id','nuclei_index','label_type','label_final_value']])
        
        if hasattr(self, 'get_merged_lbl_data_X_lastused') & \
            hasattr(self, 'get_merged_lbl_data_y_lastused') & \
            hasattr(self, 'datadict_lastused'):
            is_X_equal = np.array_equal(X, self.get_merged_lbl_data_X_lastused)
            is_y_equal = y.equals(self.get_merged_lbl_data_y_lastused)
            is_data_equal = is_X_equal & is_y_equal
            print('############################################')
            print('get_merged_lbl_data: Is data equal?', is_X_equal, is_y_equal)
            print('############################################')

            if is_data_equal:
                datadict = self.datadict_lastused
                print('---------------------------')
                print('get_merged_lbl_data: found last data used identical, skip processing.')
                print('---------------------------')
                return datadict
            else:
                pass

        # just to have a first look, do we have any data?
        if len(y) == 0:
            dict = {'interactive_labeling_proceed_status': 'no sufficient training data'}
            return dict
        
        self.get_merged_lbl_data_X_lastused = X
        self.get_merged_lbl_data_y_lastused = y
        
        idx_isBinary = np.array([str(v) in ['0','1'] for v in y['label_final_value']]) # remove "not sure" and "incorrect segmentation"
        
        y = y.loc[idx_isBinary]
        X = X[idx_isBinary, :]

        # After filtering out the non-binary lables, then we have a second look, do we have any data?
        if len(y) == 0:
            dict = {'interactive_labeling_proceed_status': 'no sufficient training data'}
            return dict

        
        y_class = y['original_class_id'].values.astype(int)
        if stage != 'apply2case':
            y_active_idx = y_class == self.activeClassID
        y_type = y['label_type'].values.astype(str)
        y_class_new = np.zeros(y_class.shape)
        class_id_list = np.unique(self.data_info['original_class_id'].values)
        
        class_id_offset = 0
        if 0 not in class_id_list: # this is to accommodate old label dataset.
            class_id_offset = 1

        for i, cls in enumerate(class_id_list):
            y_class_new[y_class == cls] = i + class_id_offset
        y['original_class_id'] = y_class_new.astype(int)
        del y_class

        y_all = (y['original_class_id'].values * y['label_final_value'].values).astype(int) # any value contains zero is zero.


        datadict = {}
        datadict['All_class'] = {}
        datadict['All_class']['train'] = {}

        training_idx = ['test' not in v for v in y_type]
        testing_idx = ['test' in v for v in y_type]
        

        datadict['All_class']['train']['X'] = X[training_idx, :]
        datadict['All_class']['train']['y'] = y_all[training_idx]#.reshape(-1,1)

        datadict['All_class']['test'] = {}
        datadict['All_class']['test']['X'] = X[testing_idx, :]
        datadict['All_class']['test']['y'] = y_all[testing_idx]#.reshape(-1,1)
        
        if stage != 'apply2case':
            datadict['Active_class'] = {}
            datadict['Active_class']['index'] = y.index
            datadict['Active_class']['train'] = {}
            datadict['Active_class']['train']['X'] = X[training_idx & y_active_idx, :]
            y_active_train = y_all[training_idx & y_active_idx]
            y_active_train[y_active_train > 0] = 1
            datadict['Active_class']['train']['y'] = y_active_train#.reshape(-1,1)
            datadict['Active_class']['test'] = {}
            datadict['Active_class']['test']['X'] = X[testing_idx & y_active_idx, :]
            y_active_test = y_all[testing_idx & y_active_idx]
            y_active_test[y_active_test > 0] = 1
            datadict['Active_class']['test']['y'] = y_active_test#.reshape(-1,1)
        
            if (len(np.unique(datadict['Active_class']['train']['y'])) < 2) or (len(np.unique(datadict['Active_class']['test']['y'])) < 2):
                # negative label has not been generated yet.
                datadict['interactive_labeling_proceed_status'] = 'no positive/negative label'
            else:
                datadict['interactive_labeling_proceed_status'] = 'pass'
        else:
            datadict['interactive_labeling_proceed_status'] = 'pass'
        self.datadict_lastused = datadict
        return datadict


    def ML_train(self,
                    datadict,
                    modelclass='All_class',
                    trained_for='All_class',
                    warmstart=False,
                    force_retrain=False,
                    ):
        print('---- ML train ----')

        if force_retrain:
            # force retrain, then we need to clear all previous model, and update data.
            if hasattr(self, 'model'):
                delattr(self, 'model')
            if hasattr(self, 'ML_result'):
                delattr(self, 'ML_result')
            
        
        if not hasattr(self, 'model'):
            self.model = {}
            self.model['classinfo'] = self.classinfo

        if not hasattr(self, 'ML_result'):
            self.ML_result = {}

        
        if platform.system() == 'Darwin': # mac OS
            tree_method = None
        else:
            tree_method = "gpu_hist"
        clf = xgb.XGBClassifier(max_depth = 8, tree_method=tree_method, eval_metric='logloss', random_state=0)
        
        X_train = datadict[trained_for]['train']['X']
        y_train = datadict[trained_for]['train']['y']
            
        if not force_retrain and hasattr(self, 'ML_X_train_last') and np.array_equal(X_train, self.ML_X_train_last) and np.array_equal(y_train, self.ML_y_train_last):
            st = time.time()
            clf = self.ML_clf_last
            et = time.time()
            print('Retrieve unchanged model cost %.2f seconds' %(et-st))
        else:
            st = time.time()

            clf.fit(X_train, y_train, xgb_model=None, sample_weight=None)
            et = time.time()
            print('training cost %.2f seconds' %(et-st))
            self.ML_X_train_last = copy.deepcopy(X_train)
            self.ML_y_train_last = copy.deepcopy(y_train)
            self.ML_clf_last = copy.deepcopy(clf)



        self.model[modelclass] = clf
        y_train_pred = clf.predict(X_train)
        train_acc = sklearn.metrics.accuracy_score(y_train, y_train_pred)
        train_cmat = sklearn.metrics.confusion_matrix(y_train, y_train_pred)

        if modelclass == 'Apply_all':
            self.X_train_last_used = X_train
            self.y_train_last_used = y_train
            print('---- ML train done ----')
            #return clf

        if len(datadict[trained_for]['test']['y']) > 0:
            X_test = datadict[trained_for]['test']['X']
            y_test = datadict[trained_for]['test']['y']
            y_test_pred = clf.predict(X_test)
            test_acc = sklearn.metrics.accuracy_score(y_test, y_test_pred)
            test_cmat = sklearn.metrics.confusion_matrix(y_test, y_test_pred)
            print('[Model for %s] train elapsed %.2f seconds' % (trained_for, time.time()-st))
            print('train acc = %.2f' % train_acc)
            print('test acc = %.2f' % test_acc)

            if 'train_acc' not in self.ML_result:
                self.ML_result['train_acc'] = {}
            if 'test_acc' not in self.ML_result:
                self.ML_result['test_acc'] = {}


            self.ML_result['train_acc'][modelclass] = train_acc
            self.ML_result['test_acc'][modelclass] = test_acc

            if 'train_acc_per_stage' not in self.ML_result:
                self.ML_result['train_acc_per_stage'] = {}
                self.ML_result['train_acc_per_stage'][modelclass] = {}
            if 'test_acc_per_stage' not in self.ML_result:
                self.ML_result['test_acc_per_stage'] = {}
                self.ML_result['test_acc_per_stage'][modelclass] = {}
            
            if hasattr(self, 'interactive_curr_epoch'):
                self.ML_result['train_acc_per_stage'][modelclass][self.interactive_curr_epoch] = train_acc
                self.ML_result['test_acc_per_stage'][modelclass][self.interactive_curr_epoch] = test_acc


            if 'train_confusion_matrix' not in self.ML_result:
                self.ML_result['train_confusion_matrix'] = {}
            if 'test_confusion_matrix' not in self.ML_result:
                self.ML_result['test_confusion_matrix'] = {}

            self.ML_result['train_confusion_matrix'][modelclass] = train_cmat
            self.ML_result['test_confusion_matrix'][modelclass] = test_cmat
        
            print('---- ML test done ----')

        return clf

    def ML_apply_all(self):
        print('---- ML apply all ----')
        if not (hasattr(self, 'ML_result') and hasattr(self, 'model') and 'Apply_all' in self.model):
            print('No Apply_all model in self.model. Skip.')
            print('This issue is because you read a slide but there is no available model.')
        else:
            st = time.time()
            clf = self.model['Apply_all']
            # apply classfier to all other nuclei in WSI
            learning_feature_idx = []
            for v in self.MainWindow.nucstat.feature_columns:
                learning_feature_idx.append(v in self.MainWindow.nucstat.learning_features)
            learning_feature_idx = np.array(learning_feature_idx)
            X_test = self.MainWindow.nucstat.feature[:, learning_feature_idx]
            

            #TODO: Could this be improved? In another experiment, I found xgboost predict_proba is already paralleled.
            
            if clf.n_features_in_ == X_test.shape[1]:
                proba_ = np.array(clf.predict_proba(X_test))
            else:
                print("Number of features mismatch. Maybe the nuclei features increased?")
                print("Start training again.")
                self.apply2case(force_retrain=True)
                
            y_pred = np.argmax(proba_, axis=1)
            y_pred_proba = np.max(proba_, axis=1)#np.array(clf.predict_proba(X_test))[:,1]

            pdin = np.c_[y_pred, y_pred_proba]
            print(Counter(y_pred))
            print('Apply to case test for the rest of nuclei elapsed %.2f seconds.' % (time.time()-st) )
            self.MainWindow.nucstat.prediction = pd.DataFrame(pdin, index=self.MainWindow.nucstat.index, columns=['label','proba'])

            self.MainWindow.nucstat.prediction.loc[:,'class_name'] = 'Other'
            # background nuclei is grey
            RGB_Other = self.classinfo.loc[0,'rgbcolor']
            self.MainWindow.nucstat.prediction.loc[:,'color_r'] = RGB_Other[0]
            self.MainWindow.nucstat.prediction.loc[:,'color_g'] = RGB_Other[1]
            self.MainWindow.nucstat.prediction.loc[:,'color_b'] = RGB_Other[2]
            


            for cls_id in self.classinfo.index:
                rgbcolor = self.classinfo.loc[cls_id, 'rgbcolor']
                class_name = self.classinfo.loc[cls_id, 'classname']
                
                boolean_idx = self.MainWindow.nucstat.prediction['label'].values.reshape(-1) == cls_id
                self.MainWindow.nucstat.prediction.loc[boolean_idx,'class_name'] = class_name
                self.MainWindow.nucstat.prediction.loc[boolean_idx,'color_r'] = rgbcolor[0]
                self.MainWindow.nucstat.prediction.loc[boolean_idx,'color_g'] = rgbcolor[1]
                self.MainWindow.nucstat.prediction.loc[boolean_idx,'color_b'] = rgbcolor[2]

                
            fi = clf.feature_importances_
            learning_feature_idx = []
            for v in self.MainWindow.nucstat.feature_columns:
                learning_feature_idx.append(v in self.MainWindow.nucstat.learning_features)
            learning_feature_idx = np.array(learning_feature_idx)
            index = pd.MultiIndex.from_arrays(np.array(self.MainWindow.nucstat.feature_columns)[learning_feature_idx].T)
            feature_importance = pd.DataFrame(fi, index=index, columns=['feature_importance'])
            feature_importance = feature_importance.sort_values('feature_importance', ascending=False)
            self.feature_importance = feature_importance
            # for i in feature_importance.index: print(i, feature_importance.loc[i,'feature_importance'])
            # print(feature_importance)
            self.send_dim_info_for_VFC()
            # TODO: also update classinfo to frontend
            # self.MainWindow.backend.py2js({'action': 'apply2case_done', 'data_dict': self.ML_result})
            print('Interactive label: Apply to case done.')
            self.MainWindow.log.write('Interactive label *** Apply to case done.')

            '''
            if hasattr(self, 'login_user_ID') and self.login_user_ID is not None:
                #save dict file to temp_annotation_file dir, in case the system crashes.
                st = time.time()
                temp_dir = os.path.join(self.wd, 'Cache', 'Temp_annotation_file')
                os.makedirs(temp_dir, exist_ok=True)
                curr_datetime = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
                save_path = os.path.join(temp_dir, 'model_userid=%d_timestamp=%s.pickle' % (self.login_user_ID, curr_datetime))
                with open(save_path, 'wb') as f:
                    pickle.dump(self.model, f)
                et = time.time()
                self.MainWindow.log.write('Interactive label *** Save current model to %s, cost %.2f seconds.' % (temp_dir, et-st))
            '''
            #save dict file to temp_annotation_file dir, in case the system crashes.
            save_ = True
            if save_:
                self.model["data_info"] = self.data_info
                self.model["classinfo"] = self.classinfo
                self.model["dataX"] = self.data_X
                st = time.time()
                temp_dir = os.path.join(self.MainWindow.wd, 'Cache', 'Temp_annotation_file')
                os.makedirs(temp_dir, exist_ok=True)
                curr_datetime = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
                save_path = os.path.join(temp_dir, 'model_timestamp=%s.pickle' % (curr_datetime))
                with open(save_path, 'wb') as f:
                    pickle.dump(self.model, f)
                et = time.time()
                self.MainWindow.log.write('Interactive label *** Save current model to %s, cost %.2f seconds.' % (temp_dir, et-st))


            
    def load_offline_model(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self.MainWindow,
            "Load Model",
            "",
            "Pickle Files (*.pkl);;All Files (*)",
            options=options
        )
        
        if file_name:
            try:
                with open(file_name, 'rb') as f:
                    loaded_model = pickle.load(f)
                
                # Update the current model with the loaded data
                self.classinfo = loaded_model['classinfo']
                self.data_info = loaded_model['data_info']
                self.data_X = loaded_model['data_X']
                if 'model' in loaded_model and loaded_model['model'] is not None:
                    self.model = loaded_model['model']
                if 'ML_result' in loaded_model and loaded_model['ML_result'] is not None:
                    self.ML_result = loaded_model['ML_result']
                
                print(f"Model loaded successfully from {file_name}")
                self.MainWindow.log.write(f"Interactive label *** Model loaded offline from {file_name}")
                
                self.MainWindow.backend.py2js({'action': 'update_classinfo', 'data': self.classinfo.to_dict()})
                # Update the UI and apply the model to the current case
                self.updateAnnotationToWebEngine()
                if self.MainWindow.imageOpened:
                    self.apply_to_case()
            except Exception as e:
                print(f"Error loading model: {e}")
                self.MainWindow.log.write(f"Interactive label *** Error loading model offline: {e}")        
            

    def save_model_offline(self):
        options = QFileDialog.Options()
        
        # Get the total number of cells
        total_cells = len(self.data_info) if hasattr(self, 'data_info') else 0
        
        # Generate default filename with current date and time, and number of cells
        default_filename = f"nuclei.io-model-{total_cells}cells-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        
        file_name, _ = QFileDialog.getSaveFileName(
            self.MainWindow,
            "Save Model",
            default_filename,
            "Pickle Files (*.pkl);;All Files (*)",
            options=options
        )
        
        if file_name:
            if not file_name.endswith('.pkl'):
                file_name += '.pkl'
            
            model_to_save = {
                'classinfo': self.classinfo,
                'data_info': self.data_info,
                'data_X': self.data_X,
                'model': self.model if hasattr(self, 'model') else None,
                'ML_result': self.ML_result if hasattr(self, 'ML_result') else None
            }
            
            try:
                with open(file_name, 'wb') as f:
                    pickle.dump(model_to_save, f)
                print(f"Model saved successfully to {file_name}")
                self.MainWindow.log.write(f"Interactive label *** Model saved offline to {file_name}")
            except Exception as e:
                print(f"Error saving model: {e}")
                self.MainWindow.log.write(f"Interactive label *** Error saving model offline: {e}")
        

    def closeMLThread(self):
        if hasattr(self.MainWindow, 'ML_thread'):
            try:
                #self.ML_thread.stop()
                self.MainWindow.ML_thread.quit()
                self.MainWindow.ML_thread.wait()
            except Exception as e:
                print('closeMLThread function has error:', e)


    def apply_to_case(self):
        if not hasattr(self.MainWindow, 'nucstat') or self.MainWindow.nucstat is None:
            print('No nucstat! Annotation exit.')
            return
        
        if not hasattr(self, 'data_X') or self.data_X is None:
            print('No training data.')
            return

        ML_async = False
        if ML_async:
            self.closeMLThread()
            # Step 2: Create a QThread object
            self.MainWindow.ML_thread = QThread()
            # Step 3: Create a worker object
            self.MainWindow.ML_worker = AsyncApplytoCase(self.MainWindow)
            # Step 4: Move worker to the thread
            self.MainWindow.ML_worker.moveToThread(self.MainWindow.ML_thread)
            # Step 5: Connect signals and slots
            self.MainWindow.ML_thread.started.connect(self.MainWindow.ML_worker.run)
            self.MainWindow.ML_worker.finished.connect(self.MainWindow.ML_thread.quit)
            self.MainWindow.ML_worker.finished.connect(self.MainWindow.ML_worker.deleteLater)
            self.MainWindow.ML_thread.finished.connect(self.MainWindow.ML_thread.deleteLater)
            # Step 6: Start the thread
            self.MainWindow.ML_thread.start()
        else:
            self.apply2case()
        


    def apply2case(self,
                   force_retrain=False):
        
        self.MainWindow.ui.statusBar.statusbar_pbar.setValue(0)
        try:
            datadict = self.get_merged_lbl_data(stage='apply2case', force_retrain=force_retrain)
        except Exception as e:
            print(e)
            self.MainWindow.backend.py2js({'action': 'apply2case_done'})
            return

        self.MainWindow.ui.statusBar.statusbar_pbar.setValue(10)
        
        try:
            self.MainWindow.log.write('Interactive label *** Apply to case start.')

            QApplication.setOverrideCursor(Qt.WaitCursor)
            X_train = np.concatenate([datadict['All_class']['train']['X'], datadict['All_class']['test']['X']], axis=0)
            y_train = np.concatenate([datadict['All_class']['train']['y'], datadict['All_class']['test']['y']], axis=0)
            
            if len(np.unique(y_train)) <= 1:
                print('Only 0 or 1 class annotated. skip.')
                self.MainWindow.ui.statusBar.statusbar_pbar.setValue(100)
                return

            if len(X_train) == 0:
                print('Empty data.')
                # reset previous labeled colors to color: Others
                RGB_Other = self.classinfo.loc[0,'rgbcolor']

                if hasattr(self, 'interactive_prediction'):
                    self.MainWindow.nucstat.prediction.loc[:,'color_r'] = RGB_Other[0]
                    self.MainWindow.nucstat.prediction.loc[:,'color_g'] = RGB_Other[1]
                    self.MainWindow.nucstat.prediction.loc[:,'color_b'] = RGB_Other[2]

                self.MainWindow.backend.py2js({'action': 'apply2case_done'})
                return
            

            is_data_equal = False
            if hasattr(self, 'X_train_apply2case_last_used') and hasattr(self, 'y_train_apply2case_last_used'):
                is_X_equal = np.array_equal(X_train, self.X_train_apply2case_last_used)
                is_y_equal = np.array_equal(y_train, self.y_train_apply2case_last_used)
                is_data_equal = is_X_equal & is_y_equal
                
                print('-----------------------')
                print('Is data equal? ', is_data_equal)
                print('-----------------------')

            self.MainWindow.ui.statusBar.statusbar_pbar.setValue(30)
            if hasattr(self, 'X_train_last_used') and hasattr(self, 'y_train_last_used') and\
                'Apply_all' in self.model and is_data_equal:
                print('previous training model existed, and the data for training did not change. So we skip re-training.')
                clf = self.model['Apply_all']
            else:
                clf = self.ML_train(datadict,
                                    modelclass = 'Apply_all',
                                    trained_for = 'All_class',
                                    force_retrain=force_retrain)
            

            self.MainWindow.ui.statusBar.statusbar_pbar.setValue(40)

            # now, check if image we are looking at is identical, otherwise, we don't have to update.
            if hasattr(self, 'X_train_last_used') and hasattr(self, 'y_train_last_used') and is_data_equal and \
                hasattr(self, 'slideDescription_last_train') and self.slideDescription_last_train == self.MainWindow.slideDescription:
                print('Identical, pass.')
                self.MainWindow.backend.py2js({'action': 'apply2case_done'})
                pass
            else:

                self.ML_apply_all()
                self.MainWindow.ui.statusBar.statusbar_pbar.setValue(50)
                self.slideDescription_last_train = self.MainWindow.slideDescription
                self.X_train_apply2case_last_used = X_train
                self.y_train_apply2case_last_used = y_train
                self.overlay_create()

            self.MainWindow.ui.statusBar.statusbar_pbar.setValue(100)

            
            print('Interactive label *** Apply to case finish overlay create.')
            self.MainWindow.log.write('Interactive label *** Apply to case finish overlay create.')
            
            
            self.MainWindow.showOverlayML()
            self.MainWindow.showOverlayAnnotation()
            QApplication.setOverrideCursor(Qt.ArrowCursor)
            return
        
        except Exception as e:
            print('Apply2case Error:', e)
            print(traceback.format_exc())
            self.MainWindow.backend.py2js({'action': 'apply2case_done'})
            QApplication.setOverrideCursor(Qt.ArrowCursor)
            return
    



    def overlay_create(self,
                        patch=512,
                        threshold=10,
                        init=False
                        ):

        if init:
            self.ui.backend_workspace.py2js({'action':'update_slide_progress',
                                            'value':'create low-magnification overlay',
                                            'progress': '0'})
        st = time.time()
        c_x = self.MainWindow.nucstat.centroid[:,0]
        c_y = self.MainWindow.nucstat.centroid[:,1]
        dim = self.MainWindow.slide.dimensions
        dim_downsample = (int(np.round(dim[0]/patch)), int(np.round(dim[1]/patch)))
        
        c_x_downsample = np.round(c_x/patch)
        c_y_downsample = np.round(c_y/patch)

        n_tumor_cells = np.sum(self.MainWindow.nucstat.prediction['label'].values.reshape(-1) == 1)
        
        if n_tumor_cells < 2000:
            threshold = 3
        if n_tumor_cells < 200:
            threshold = 2

        if not hasattr(self.MainWindow.nucstat, 'overlay'):
            slide_path = os.path.dirname(os.path.realpath(self.MainWindow.slide_filepath))
            overlay_filepath = os.path.join(slide_path, 'temp',
                                '%s_%s_nucstat_overlay_%d_%d.pkl' % \
                                (self.MainWindow.case_id, self.MainWindow.slide_id, patch, threshold))
            if os.path.exists(overlay_filepath):
                with open(overlay_filepath, 'rb') as f:
                    self.MainWindow.nucstat.overlay = pickle.load(f)
            else:
                os.makedirs(os.path.join(slide_path, 'temp'), exist_ok=True)
                self.MainWindow.nucstat.overlay = {'index_x_valid':[],
                                        'index_y_valid':[],
                                        'subset_index':{},
                                        'idx1_map':{},
                                        'idx2_map':{},
                                        'overlay': None}
                for i in range(dim_downsample[0]):
                    self.MainWindow.nucstat.overlay['idx1_map'][i] = np.where(c_x_downsample==i)[0]
                    if len(self.MainWindow.nucstat.overlay['idx1_map'][i]) > 0:
                        self.MainWindow.nucstat.overlay['index_x_valid'].append(i)
                for j in range(dim_downsample[1]):
                    self.MainWindow.nucstat.overlay['idx2_map'][j] = np.where(c_y_downsample==j)[0]
                    if len(self.MainWindow.nucstat.overlay['idx2_map'][j]) > 0:
                        self.MainWindow.nucstat.overlay['index_y_valid'].append(j)
                for i in tqdm(self.MainWindow.nucstat.overlay['index_x_valid']):
                    idx1 = self.MainWindow.nucstat.overlay['idx1_map'][i]
                    for j in self.MainWindow.nucstat.overlay['index_y_valid']:
                        idx2 = self.MainWindow.nucstat.overlay['idx2_map'][j]
                        subset_idx = idx1[np.in1d(idx1, idx2, assume_unique=True)]
                        self.MainWindow.nucstat.overlay['subset_index'][(i,j)] = subset_idx

                with open(overlay_filepath, 'wb') as f:
                    pickle.dump(self.MainWindow.nucstat.overlay, f)
        
        overlay = np.zeros((dim_downsample[0], dim_downsample[1], 4), dtype=np.uint8)
        interactive_prediction_label = self.MainWindow.nucstat.prediction['label'].values.astype(int)

        colors_df = self.MainWindow.nucstat.prediction.drop_duplicates('label').set_index('label')
        
        for i in tqdm(self.MainWindow.nucstat.overlay['index_x_valid']):
            '''
            if i % 10 == 0:
                self.ui.backend_workspace.py2js({'action':'update_slide_progress',
                                                'value':'create low-magnification overlay',
                                                'progress': str(i/len(self.MainWindow.nucstat.overlay['index_x_valid']))})
            '''
            idx1 = self.MainWindow.nucstat.overlay['idx1_map'][i]
            for j in self.MainWindow.nucstat.overlay['index_y_valid']:
                idx2 = self.MainWindow.nucstat.overlay['idx2_map'][j]
                subset_idx = self.MainWindow.nucstat.overlay['subset_index'][(i,j)]
                label = interactive_prediction_label[subset_idx]
                if len(label) == 0: continue
                n_pos = label[label>0]
                if len(n_pos) >= threshold:
                    major_label, count = Counter(n_pos).most_common(1)[0]
                    rgb = colors_df.loc[major_label,['color_r','color_g','color_b']].values.astype(np.uint8)
                    
                    fix_constant = 500
                    if n_tumor_cells < 2000:
                        fix_constant = 1500
                    if n_tumor_cells < 200:
                        fix_constant = 5000
                    
                    alpha = np.min((count/len(label)*fix_constant, 255)) * 0.8
                    alpha = int(alpha)
                    
                    '''
                    # alpha = np.min((count*2, 255))
                    # alpha = 255
                    '''
                    rgba = (rgb[0],rgb[1],rgb[2],alpha)
                    overlay[i,j,:] = rgba
                    
        self.MainWindow.nucstat.overlay['overlay'] = overlay

        et = time.time()
        print('Get overlay cost %.2f seconds' % (et-st))
        return


    def send_dim_info_for_VFC(self,
                                dim1_text = None,
                                dim2_text = None,
                                max_number_cell = 10000
                                ):


        if hasattr(self, 'feature_importance'):
            feature_importance = copy.deepcopy(self.feature_importance)
            feature_importance.index = [' | '.join(v) for v in feature_importance.index]

        if dim1_text is not None:
            self.dim1_text = dim1_text
        elif hasattr(self, 'feature_importance'):
            # choose first important feature
            self.dim1_text = self.feature_importance.index[0]
        else:
            self.dim1_text = ('Morphology', 'area')

        if dim2_text is not None:
            self.dim2_text = dim2_text
        elif hasattr(self, 'feature_importance'):
            # choose second important feature
            self.dim2_text = self.feature_importance.index[1]
        else:
            self.dim2_text = ('Haralick', 'heterogeneity')

        dim1_text_flat = ' | '.join(self.dim1_text)
        dim2_text_flat = ' | '.join(self.dim2_text)
        
        idx1 = np.array([ix for ix, v in enumerate(self.MainWindow.nucstat.feature_columns) if self.dim1_text[0] == v[0] and self.dim1_text[1] == v[1]])
        idx2 = np.array([ix for ix, v in enumerate(self.MainWindow.nucstat.feature_columns) if self.dim2_text[0] == v[0] and self.dim2_text[1] == v[1]])

        dim1 = self.MainWindow.nucstat.feature[:, idx1].astype(np.float64).reshape(-1)
        dim2 = self.MainWindow.nucstat.feature[:, idx2].astype(np.float64).reshape(-1)

        self.VFC_dim1_val = dim1
        self.VFC_dim2_val = dim2

        # 1: selected; 0: not selected.
        if hasattr(self.MainWindow.nucstat, 'isSelected_to_VFC'): # previously isSelected
            idx_selected = self.MainWindow.nucstat.isSelected_to_VFC.reshape(-1).astype(float) # Object of type int32 is not JSON serializable
        else:
            idx_selected = np.array([1] * len(dim1)).astype(float)

        
        if hasattr(self.MainWindow.nucstat, 'prediction') and len(self.MainWindow.nucstat.prediction) > 0:
            classname = self.MainWindow.nucstat.prediction.loc[:,'class_name'].values.reshape(-1)
            color_r = self.MainWindow.nucstat.prediction.loc[:,'color_r'].values.astype(float)
            color_g = self.MainWindow.nucstat.prediction.loc[:,'color_g'].values.astype(float)
            color_b = self.MainWindow.nucstat.prediction.loc[:,'color_b'].values.astype(float)
        else:
            classname = np.array(['Other']*len(dim1))
            color_r = np.array([120] * len(dim1)).astype(float)
            color_g = np.array([120] * len(dim1)).astype(float)
            color_b = np.array([120] * len(dim1)).astype(float)


        # get unique class information for tumor burden stat
        uniq_classname = np.unique(classname)
        uniq_classcolor = []
        uniq_class_nuclei_count = []
        for c in uniq_classname:
            rgb = 'rgb(%d, %d, %d)' % (color_r[classname == c][0], color_g[classname == c][0], color_b[classname == c][0])
            uniq_classcolor.append(rgb)
            n_nuclei = np.sum(classname == c)
            uniq_class_nuclei_count.append(n_nuclei)


        # subsample index
        idx_out_select = np.where(idx_selected == 0)[0]
        idx_in_select = np.where(idx_selected == 1)[0]
        random.shuffle(idx_out_select)
        random.shuffle(idx_in_select)
        idx_out_select = idx_out_select[:np.min((max_number_cell,len(idx_out_select)))]
        idx_in_select = idx_in_select[:np.min((max_number_cell,len(idx_in_select)))]

        if len(idx_out_select) > 0:
            uniq_classname_selected = np.unique(classname[idx_in_select])
            uniq_classname_notselected = np.unique(classname[idx_out_select])
            uniq_classcolor_selected = np.array(uniq_classcolor)[np.isin(uniq_classname, uniq_classname_selected)]
            uniq_classcolor_notselected = np.array(uniq_classcolor)[np.isin(uniq_classname, uniq_classname_notselected)]
            
            uniq_class_nuclei_count_selected = []
            for c in uniq_classname_selected:
                uniq_class_nuclei_count_selected.append(np.sum((classname == c) & (idx_selected == 1)))

            uniq_class_nuclei_count_notselected = []
            for c in uniq_classname_notselected:
                uniq_class_nuclei_count_notselected.append(np.sum((classname == c) & (idx_selected == 0)))


        subsample_idx = np.array(list(idx_out_select) + list(idx_in_select))


        dim1_subset = dim1[subsample_idx]
        dim2_subset = dim2[subsample_idx]
        classname = classname[subsample_idx]
        idx_selected = idx_selected[subsample_idx]
        color_r = color_r[subsample_idx]
        color_g = color_g[subsample_idx]
        color_b = color_b[subsample_idx]



        dict2send = {}
        dict2send['action'] = 'plot_virtual_flow_stat'
        dict2send['data'] = {}
        dict2send['data']['dim1'] = list(dim1_subset) # cannot json np array
        dict2send['data']['dim2'] = list(dim2_subset)
        dict2send['axis_title'] = {}
        dict2send['axis_title']['dim1'] =  dim1_text_flat
        dict2send['axis_title']['dim2'] =  dim2_text_flat
        dict2send['class_name'] =  list(classname)
        dict2send['class_color_r'] =  list(color_r)
        dict2send['class_color_g'] =  list(color_g)
        dict2send['class_color_b'] =  list(color_b)
        dict2send['main_window_selected'] =  list(idx_selected)

        if hasattr(self, 'feature_importance'):
            dict2send['feature_importance'] = feature_importance.to_dict()

        dict2send['barplot'] = {}
        dict2send['barplot']['selected'] = {}
        dict2send['barplot']['selected']['class_name_unique'] = list(uniq_classname)
        dict2send['barplot']['selected']['class_color_unique'] = list(uniq_classcolor)
        dict2send['barplot']['selected']['class_count_unique'] = list(np.array(uniq_class_nuclei_count).astype(float)) # cannot json int32 array
        
        if len(idx_out_select) > 0:
            dict2send['barplot']['selected']['class_name_unique'] = list(uniq_classname_selected)
            dict2send['barplot']['selected']['class_color_unique'] = list(uniq_classcolor_selected)
            dict2send['barplot']['selected']['class_count_unique'] = list(np.array(uniq_class_nuclei_count_selected).astype(float)) # cannot json int32 array
        
            dict2send['barplot']['not_selected'] = {}
            dict2send['barplot']['not_selected']['class_name_unique'] = list(uniq_classname_notselected)
            dict2send['barplot']['not_selected']['class_color_unique'] = list(uniq_classcolor_notselected)
            dict2send['barplot']['not_selected']['class_count_unique'] = list(np.array(uniq_class_nuclei_count_notselected).astype(float)) # cannot json int32 array
        

        
        self.MainWindow.backend.py2js(dict2send) # send total nuclei count, active nuclei count, accuracy, etc.


    def show_highlight_nucleus_from_VFC(self, dim1, dim2):
        idx_order = np.argmin((self.VFC_dim1_val - dim1)**2 + (self.VFC_dim2_val - dim2)**2)

        x, y = self.nucstat['centroid'][idx_order,:]
        idx_raw = self.nucstat['index'][idx_order]
        self.highlight_nuclei = {}
        self.highlight_nuclei['index'] = idx_raw

        self.log.write('Virtual Flow *** show single-clicked nuclei in WSI. Centroid on slide: (%d, %d).' % (x, y))

        slide_dimension = self.slide.level_dimensions[0] # highest resolution, size around 80000*40000
        self.setZoomValue(1) # the smaller, the larger cell looks like
        self.relativeCoords = np.array([x/slide_dimension[0],
                                                y/slide_dimension[1]]).astype(float)
        if (self.relativeCoords[1]>1.0):
            self.relativeCoords[1]=1.0

        self.relativeCoords -= 0.5
        
        image_dims=self.slide.level_dimensions[0]
        self.ui.statusbar_text.setText('Position: (%d,%d)' % (int((self.relativeCoords[0]+0.5)*image_dims[0]), int((self.relativeCoords[1]+0.5)*image_dims[1])))
        self.showImage()
        self.updateScrollbars()
    
    def show_selected_nuclei_from_VFC(self, selectdict):
        if selectdict['type'] == 'lasso':
            x = np.array(selectdict['points_x']).astype(float)
            y = np.array(selectdict['points_y']).astype(float)
            polypoints = np.c_[x,y]
            poly_path = path.Path(polypoints)
            index_bool = poly_path.contains_points(np.c_[self.VFC_dim1_val, self.VFC_dim2_val])


        elif selectdict['type'] == 'rect':
            x = np.array(selectdict['points_x']).astype(float)
            y = np.array(selectdict['points_y']).astype(float)
            xmin, xmax = np.min(x), np.max(x)
            ymin, ymax = np.min(y), np.max(y)
            index_bool = (self.VFC_dim1_val > xmin) & (self.VFC_dim1_val < xmax) & (self.VFC_dim2_val > ymin) & (self.VFC_dim2_val < ymax)
            
        self.nucstat.isSelected_from_VFC = index_bool
        self.processingStep = self.id+1
        self.showImage_part2(self.npi, self.processingStep)


    def import_nucfinder_dataset(self, dataset_id):
        status = self.import_from_db(dataset_id)
        if status == 'success':
            if self.MainWindow.imageOpened:
                self.apply_to_case()
        return
                
    def rgb_to_hex(self, rgb):
        return '%02x%02x%02x' % tuple(rgb)

    def import_from_db(self, dataset_id):
        print('Retrieve dataset with id:',dataset_id)
        
        mydb = self.MainWindow.connect_to_DB()
        mycursor = mydb.cursor(buffered=True)
        sql = "SELECT * FROM Dataset_nuclei WHERE id = '%s'" % dataset_id
        mycursor.execute(sql)
        mydb.commit()
        print('Database:', mycursor.rowcount, "record found.")
        data = mycursor.fetchall()
        if len(data) != 1: raise Exception('Multiple ID found!')
        data = data[0]
        user_id = data[1]
        remotepath = data[2]
        create_datetime = data[3]
        lastedit_datetime = data[4]
        tissueType = data[5]
        description = data[6]
        n_class = data[7]
        n_nuclei = data[8]
        test_acc = data[9]
        isPrivate = data[10]
        

        cachedir = os.path.join(self.MainWindow.wd, 'Cache', 'Receive', dataset_id)
        os.makedirs(cachedir, exist_ok=True)
        
        cli, REMOTE_DIR = self.MainWindow.connect_to_SSH()

        print('Start sftp')
        st = time.time()
        sftp = cli.open_sftp()
        fname='interactive_nuclei_dict.pickle'
        try:
            fname='interactive_nuclei_dict.pickle'
            sftp.get(os.path.join(remotepath, fname), os.path.join(cachedir, fname))
        except:
            fname='model.pickle'
            sftp.get(os.path.join(remotepath, fname), os.path.join(cachedir, fname))
            
        with open(os.path.join(cachedir, fname), 'rb') as f:
            model_in = pickle.load(f)

        if fname == 'interactive_nuclei_dict.pickle':
            model_in['classinfo'] = model_in['class_info']
            model_in['classinfo'].rename(columns={"class_name": "classname",
                                                "class_color": "rgbcolor"},
                                                inplace=True)
            del model_in['class_info']
            model_in['data_info']['original_class_id'] = model_in['data_info']['class_id']
        
        print('Import finished! Time elapsed = %.2f seconds.' % (time.time()-st))
        self.classinfo = model_in['classinfo']
        self.model = model_in

        self.data_X = self.model['dataX']
        self.data_info = self.model['data_info']

        jsdict = {'action': 'import: load_class_info'}
        jsdict['classinfo'] = {}
        
        y_labels = (self.model['data_info']['original_class_id'].values * self.model['data_info']['label_final_value'].values).astype(int) # any value contains zero is zero.

        for c in self.model['classinfo'].index:
            class_name = self.model['classinfo'].loc[self.model['classinfo'].index==c, 'classname'].values[0]
            class_color = self.model['classinfo'].loc[self.model['classinfo'].index==c, 'rgbcolor'].values[0]
            n_curr_nuclei = np.sum(y_labels == c)
            jsdict['classinfo'][c] = {'classname': class_name,
                                      'n_curr_nuclei': float(n_curr_nuclei),
                                      'hexcolor': self.rgb_to_hex(class_color)}

        self.MainWindow.backend.py2js(jsdict)

        dict2send = {}
        dict2send['action'] = 'import_class_status'
        dict2send['n_class'] = n_class
        dict2send['n_nuclei'] = float(n_nuclei)
        dict2send['test_acc'] = '%.2f%%' % (test_acc*100)
        dict2send['creatorID'] = user_id
        dict2send['create_datetime'] = str(create_datetime)
        dict2send['tissueType'] = tissueType
        dict2send['description'] = description
        self.MainWindow.backend.py2js(dict2send)

        return "success"


    def sync_to_db(self, MainWindow, jsondata):
        #print(jsondata)
        userid = int(jsondata['id'])
        isPrivate_int = int(bool(jsondata['isPrivate']))
        tissueType = jsondata['tissueType']
        studyDescription = jsondata['studyDescription']
        


        # SFTP
        cli, REMOTE_DIR = self.MainWindow.connect_to_SSH()
        dataset_id = self.dataset_id
        remotepath = REMOTE_DIR + dataset_id + '/'
        cachedir = opj(MainWindow.wd, 'Cache', 'Send', dataset_id)
        os.makedirs(cachedir, exist_ok=True)
        model_save = {'classinfo': self.classinfo,
                    'data_info': self.data_info,
                    'dataX': self.data_X}
        with open(opj(cachedir, 'model.pickle'), 'wb') as f:
            pickle.dump(model_save, f)
        stdin_, stdout_, stderr_ = cli.exec_command("mkdir %s" % remotepath)
        stdout_.channel.recv_exit_status()
        lines = stdout_.readlines()
        for line in lines: print(line)
        print('Start sftp')
        sftp = cli.open_sftp()
        fname='model.pickle'
        sftp.put(opj(cachedir, fname), remotepath + fname)
        sftp.close()
        print('End sftp')
        cli.close()



        # Database
        mydb = self.MainWindow.connect_to_DB()
        number_of_class = 0
        class_count = []
        for cls in self.classinfo.index:
            number_of_class += 1
            n_of_nuclei = np.sum((self.data_info['original_class_id'] == cls).values)
            class_count.append(n_of_nuclei)
        n_total_nuclei = np.sum(class_count)

        if hasattr(self, 'ML_result') and ('test_acc' in self.ML_result):
            if 'All_class' in self.ML_result['test_acc']:
                accuracy = self.ML_result['test_acc']['All_class']
            else:
                accuracy = 0
        else:
            accuracy = 0
        
        mycursor = mydb.cursor()
        sql = "INSERT INTO Dataset_nuclei (id, creator_id, filepath, createtime, lastupdatetime, cancertype, "+\
                "studydescription, numberofclass, numberofnuclei, accuracy, isPrivate) VALUES (%s, %s, %s, NOW(), NOW(), %s, %s, %s, %s, %s, %s)"
        val = (dataset_id, userid, remotepath, tissueType, studyDescription, str(number_of_class), str(n_total_nuclei), str(accuracy), str(isPrivate_int))
        mycursor.execute(sql, val)
        mydb.commit()
        print('Database:', mycursor.rowcount, "record inserted.")
        return 'success'
