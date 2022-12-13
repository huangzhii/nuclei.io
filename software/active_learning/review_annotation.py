#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 09/07/2022
#############################################################################


from PySide6.QtCore import (QDir, QFile, QIODevice, QUrl, Qt, Slot, Signal, QPointF,
    QPoint, QEvent, QObject, QThread)

from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)

from PySide6.QtWidgets import (QWidget, QDialog, QFileDialog, QMainWindow, QMessageBox, QSlider,
                                QDialogButtonBox)

from PySide6.QtWidgets import (QApplication, QHBoxLayout, QVBoxLayout, QGridLayout, QStackedLayout, QMainWindow, QMenu, QLabel,
    QMenuBar, QPlainTextEdit, QSizePolicy, QSplitter, QScrollBar, QProgressBar, QPushButton, QDial, QToolBar,
    QStatusBar, QWidget)
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineWidgets import QWebEngineView

import os, platform
import random
import numpy as np
import pandas as pd
from datetime import datetime
from ast import literal_eval
from PIL import Image, ImageDraw
from PIL.ImageQt import ImageQt
from functools import partial
import json
import copy
import time
from tqdm import tqdm
from collections import Counter
from gui.utils import im_2_b64
from slide_reader.slideReader import RotatableOpenSlide, SlideReader, SlideImageReceiverThread



class BackendReview(QObject):
    sendjson_to_js = Signal(str)

    def __init__(self, engine=None, MainWindow=None, interactiveObject=None):
        super().__init__(engine)
        self.MainWindow = MainWindow
        self.interactiveObject = interactiveObject

    @Slot(str)
    def log(self, logstring):
        print(logstring)
        
    @Slot(result=str)
    def py2js(self, dict):
        jsonarray = json.dumps(dict)
        self.sendjson_to_js.emit(jsonarray)

        
    @Slot()
    def load_nuclei_dict(self):
        self.interactiveObject.interactive_review_dialog_load_nuclei_dict()

    @Slot()
    def init_active_learning(self):
        self.interactiveObject.init_active_learning_from_python()

    @Slot(str)
    def review_show_nuclei(self, nuclei_id):
        self.interactiveObject.review_show_nuclei(nuclei_id, self.MainWindow)

    @Slot(str, str)
    def review_update_class(self, nuclei_id, new_class):
        self.interactiveObject.review_update_class(nuclei_id, new_class, self.MainWindow)

    @Slot(str)
    def review_update_filepath(self, new_paths_json):
        new_paths = json.loads(new_paths_json)
        new_filepaths = new_paths['path']
        data_info = self.MainWindow.datamodel.data_info
        '''
        Warning:
        Following steps just changed filepath. The case_id and slide_id
        remained as previous.
        '''
        for i, v in enumerate(new_filepaths):
            if len(v) > 0:
                old_filename = self.uniq_svs_files[i]
                data_info.loc[data_info['filepath'] == old_filename, 'filepath'] = new_filepaths[i]
                self.uniq_svs_files[i] = new_filepaths[i]
        self.MainWindow.datamodel.data_info = data_info

        dict2send = {'data_info': self.MainWindow.datamodel.data_info,
                    'classinfo': self.MainWindow.datamodel.classinfo,
                    'action': 'nuclei_dict_for_review'
                    }
        self.py2js(dict2send)

    @Slot(str)
    def active_learning_next_batch(self, jsonstr):
        al_changes = json.loads(jsonstr)
        print('-------AL changes-------')
        print(al_changes)
        if len(al_changes['nuclei_id']) == 0:
            print('No nuclei approved.')
        else:
            selected_nuclei_id = np.array(al_changes['nuclei_id']).astype(int)
            selected_nuclei_classname = al_changes['class_name']
            classinfo = self.MainWindow.datamodel.classinfo
            selected_nuclei_classid = [classinfo.index[classinfo['classname']==name][0] \
                                        for name in selected_nuclei_classname]
            selected_nuclei_classid = np.array(selected_nuclei_classid)
            print(selected_nuclei_classid)

            for i in range(len(selected_nuclei_id)):
                idx = selected_nuclei_id[i]
                class_id = selected_nuclei_classid[i]
                global_pts = self.MainWindow.nucstat.contour[idx,:]
                local_pts = [self.MainWindow.slideToScreen((posx, posy)) for posx, posy in global_pts]
                ROI_dict = {'type': 'activelearning', 'points': local_pts, 'points_global': global_pts, 'rotation': self.MainWindow.rotation}
                ROI_dict['class_ID'] = class_id
                ROI_dict['class_name'] = self.MainWindow.datamodel.classinfo.loc[class_id, 'classname']
                ROI_dict['class_rgbcolor'] = self.MainWindow.datamodel.classinfo.loc[class_id, 'rgbcolor']


                print('active learning. Nuclei index selected: %d' % (idx))
                self.MainWindow.log.write('Interactive label *** (active learning) click on screen. case_id: %s, slide_id: %s, Nuclei index selected: %d' % (self.MainWindow.case_id, self.MainWindow.slide_id, idx))

                tuple_idx = (self.MainWindow.case_id, self.MainWindow.slide_id, class_id, idx)
                bool_idx_exist = (self.MainWindow.datamodel.data_info['case_id'].values == self.MainWindow.case_id) & \
                                (self.MainWindow.datamodel.data_info['slide_id'].values == self.MainWindow.slide_id) & \
                                (self.MainWindow.datamodel.data_info['nuclei_index'].values == idx)
            
                if np.sum(bool_idx_exist) > 0:
                    if class_id == self.MainWindow.datamodel.data_info.loc[bool_idx_exist, 'original_class_id'].values[0]:
                        '''
                        If that existed annotation has same labeling class, then we ignore it.
                        '''
                        pass
                    else:
                        '''
                        If that existed annotation has different labeling class, then we update that nuclei.
                        '''
                        self.MainWindow.datamodel.data_info.loc[bool_idx_exist,'original_class_id'] = class_id
                        self.MainWindow.datamodel.data_info.loc[bool_idx_exist,'label_toggle_count'] += 1
                        self.MainWindow.datamodel.data_info.loc[bool_idx_exist,'label_final_datetime'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                        # Modify annotation (polygon)
                        self.MainWindow.annotation.delete_annotation(ROI_dict)
                        self.MainWindow.annotation.add_annotation(ROI_dict)
                        self.MainWindow.showOverlayAnnotation()
                else:
                    new_row = self.MainWindow.datamodel.add_data_row(idx, label_type='activelearning', class_id=class_id)
                    self.MainWindow.datamodel.data_info = pd.concat([self.MainWindow.datamodel.data_info, new_row], axis=0, ignore_index=True)
                    learning_feature_idx = []
                    for v in self.MainWindow.nucstat.feature_columns:
                        learning_feature_idx.append(v in self.MainWindow.nucstat.learning_features)
                    learning_feature_idx = np.array(learning_feature_idx)
                    feat = pd.DataFrame(self.MainWindow.nucstat.feature[idx == self.MainWindow.nucstat.index, :][:, learning_feature_idx], index=[tuple_idx]) # 1 x N_features            
                    if not hasattr(self.MainWindow.datamodel, 'data_X') or self.MainWindow.datamodel.data_X is None:
                        self.MainWindow.datamodel.data_X = feat
                    else:
                        self.MainWindow.datamodel.data_X = pd.concat([self.MainWindow.datamodel.data_X, feat], axis=0)
                    
                    # Add annotation (polygon)
                    self.MainWindow.annotation.add_annotation(ROI_dict)

            self.MainWindow.showOverlayAnnotation()
            self.MainWindow.datamodel.updateAnnotationToWebEngine()
            self.MainWindow.datamodel.apply_to_case()

            self.interactiveObject.init_active_learning_from_python()



                




        
class InteractiveReviewDialog(QDialog):
    def __init__(self, MainWindow):
        super().__init__()
        self.setWindowTitle("Review annotations")
        self.resize(1400, 800)
        self.other_slides = {}
        self.MainWindow = MainWindow

        self.layout = QVBoxLayout()
        self.horizontalSplitter = QSplitter()
        self.horizontalSplitter.setOrientation(Qt.Horizontal)
        self.horizontalSplitter.setStyleSheet("QSplitter::handle:vertical {{margin: 0px 0px; "
                           "background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
                           "stop:0.0 rgba(255, 255, 255, 0),"
                           "stop:0.5 rgba({0}, {1}, {2}, {3}),"
                           "stop:0.99 rgba(255, 255, 255, 0))}}"
                           "QSplitter::handle:horizontal {{margin: 0px 0px; "
                           "background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, "
                           "stop:0.0 rgba(255, 255, 255, 0),"
                           "stop:0.5 rgba({0}, {1}, {2}, {3}),"
                           "stop:0.99 rgba(255, 255, 255, 0));}}".format(64, 143, 255, 255))
        self.sliderbar = QSlider(Qt.Horizontal)
        self.sliderbar_init_value = 90
        self.patchsize = (102-self.sliderbar_init_value)*30
        self.windowsize = 360
        self.sliderbar.setFixedWidth(self.patchsize)
        self.sliderbar.setValue(self.sliderbar_init_value)
        self.sliderbar.setTickInterval(10)
        self.sliderbar.valueChanged.connect(self.sliderbar_zoom_review)


        image = Image.fromarray(np.zeros((self.windowsize, self.windowsize)).astype(np.uint8))
        pixmap = QPixmap.fromImage(ImageQt(image))
        pixmap = pixmap.scaled(self.windowsize, self.windowsize, Qt.KeepAspectRatio)
        self.img_patch = QLabel()
        self.img_patch.setAlignment(Qt.AlignCenter)
        self.img_patch.setPixmap(pixmap)
        
        layout_img_lbl = QVBoxLayout()
        self.img_lbl_title = QLabel('Current nucleus:')
        self.img_lbl_case = QLabel('Case:')
        self.img_lbl_slide = QLabel('Slide:')
        self.img_lbl_id = QLabel('ID:')
        layout_img_lbl.addWidget(self.img_lbl_title)
        layout_img_lbl.addWidget(self.img_lbl_case)
        layout_img_lbl.addWidget(self.img_lbl_slide)
        layout_img_lbl.addWidget(self.img_lbl_id)
        layout_img_lbl.setAlignment(Qt.AlignLeft)
        layout_img_lbl.setContentsMargins(0,0,0,0)

        self.interactive_column = QVBoxLayout()
        self.interactive_column.addWidget(self.sliderbar)
        self.interactive_column.addLayout(layout_img_lbl)
        self.interactive_column.addWidget(self.img_patch)
        self.interactive_column.setAlignment(Qt.AlignTop)
        self.interactive_column_parent_widget = QWidget()
        self.interactive_column_parent_widget.setLayout(self.interactive_column)


        self.web_review = QWebEngineView()
        self.web_review.setContextMenuPolicy(Qt.NoContextMenu)
        self.backend_review = BackendReview(self.web_review, self.MainWindow, self)
        self._channel = QWebChannel(self)
        self.web_review.page().setWebChannel(self._channel)
        self._channel.registerObject("backend_review", self.backend_review)
        filename = os.path.join(self.MainWindow.wd, 'HTML5_UI', 'html', 'nucfinder_review.html')
        url = QUrl.fromLocalFile(filename)
        self.web_review.load(url)

        self.buttonBox = QDialogButtonBox()
        self._btn_finish = QPushButton("OK", autoDefault=False)
        self._btn_finish.setFixedWidth(240)
        self._btn_finish.setFixedHeight(40)
        self._btn_finish.setStyleSheet("QPushButton{background-color: rgb(200,200,200); color: black; font-size: 16px;font-family: Arial}")
        self._btn_finish.clicked.connect(self.review_finish)
        self.buttonBox.addButton(self._btn_finish, QDialogButtonBox.RejectRole)

        self.horizontalSplitter.addWidget(self.interactive_column_parent_widget)
        self.horizontalSplitter.addWidget(self.web_review)
        self.horizontalSplitter.setSizes([self.windowsize, 1400-self.windowsize])

        self.layout.addWidget(self.horizontalSplitter)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

    def review_show_nuclei(self, nuclei_id, MainWindow):
        nucleus_info = nuclei_id.split('@')
        if len(nucleus_info) == 3:
            case_id, slide_id, nuc_idx = nucleus_info
        elif len(nucleus_info) == 1:
            case_id, slide_id, nuc_idx = self.MainWindow.case_id, self.MainWindow.slide_id, nucleus_info[0]
        else:
            print('nucleus_info', nucleus_info)
            case_id, slide_id, nuc_idx = 'unknown', 'unknown', '0'
        nuc_idx = int(nuc_idx)
        self.img_lbl_case.setText('Case: %s' % case_id)
        self.img_lbl_slide.setText('Slide: %s' % slide_id)
        self.img_lbl_id.setText('ID: %d' % nuc_idx)

        idx_bool = (MainWindow.nucstat.index == nuc_idx)
        slide = MainWindow.slide

        contour = MainWindow.nucstat.contour[idx_bool,...][0]
        
        self.for_sliderbar = {'slide': slide,
                                'contour': contour,
                                'idx_bool': idx_bool}

        self.img_patch.close()
        self.img_patch = self.plot_nuc_patch_review(slide, contour, contourtype=None)
        self.interactive_column.addWidget(self.img_patch)
        self.interactive_column.update()




    def sliderbar_zoom_review(self):
        '''
        We get the slierbar zoom value, and then update the image patch.
        '''
        MainWindow = self.MainWindow
        slide = self.for_sliderbar['slide']
        contour = self.for_sliderbar['contour']
        idx_bool = self.for_sliderbar['idx_bool']

        zoomval = self.sliderbar.value()
        
        self.patchsize = (102-zoomval)*30
        
        self.img_patch.close()
        self.img_patch = self.plot_nuc_patch_review(slide, contour, contourtype=None)
        self.interactive_column.addWidget(self.img_patch)
        self.interactive_column.update()

        # update behavior
        '''
        if self.patchsize < MainWindow.datamodel.data_info.loc[idx_bool, 'behavior_min_patch_size'].values[0]:
            MainWindow.datamodel.data_info.loc[idx_bool, 'behavior_min_patch_size'] = self.patchsize
        if self.patchsize > MainWindow.datamodel.data_info.loc[idx_bool, 'behavior_max_patch_size'].values[0]:
            MainWindow.datamodel.data_info.loc[idx_bool, 'behavior_max_patch_size'] = self.patchsize
        
        MainWindow.datamodel.data_info.loc[idx_bool, 'behavior_zoom_count'] += 1
        '''



    def plot_nuc_patch_review(self, slide, contour, contourtype = 'polygon'):
        
        rectwidth = 1
        offset_on_screen = 5
        if self.patchsize > 500:
            contourtype = 'rect'
            rectwidth = 5
            offset_on_screen = 10
        if self.patchsize > 1000:
            rectwidth = 10
            offset_on_screen = 15
        if self.patchsize > 2000:
            rectwidth = 20
            offset_on_screen = 20

        # Don't use ('Nuclei', 'bbox'), use this instead:
        coord = [np.min(contour[:,0]), np.min(contour[:,1]), np.max(contour[:,0]), np.max(contour[:,1])]
        
        w = coord[2]-coord[0]
        h = coord[3]-coord[1]

        offset_x = int(np.round((self.patchsize - w)/2))
        offset_y = int(np.round((self.patchsize - h)/2))
        new_coord = [coord[0]-offset_x, coord[1]-offset_y, coord[2]+offset_x, coord[3]+offset_y]
        image = slide.read_region(location=new_coord[0:2], size=[new_coord[2]-new_coord[0], new_coord[3]-new_coord[1]], level=0)
        image = Image.fromarray(image[..., :3])
        image = image.convert('RGBA')
        contour_relative = copy.deepcopy(contour)
        contour_relative[:,0] = contour[:,0] - coord[0]+offset_x
        contour_relative[:,1] = contour[:,1] - coord[1]+offset_y
        contour_tuples = [(contour_relative[ci,0],contour_relative[ci,1]) for ci in range(len(contour_relative))]

        transp = Image.new('RGBA', image.size, (0,0,0,0))
        draw = ImageDraw.Draw(transp,'RGBA')
        
        if contourtype == 'rect':
            bbox = np.zeros((2,2))
            bbox[0,0] = (np.min(contour_relative[:,0])) - offset_on_screen
            bbox[1,0] = (np.max(contour_relative[:,0])) + offset_on_screen
            bbox[0,1] = (np.min(contour_relative[:,1])) - offset_on_screen
            bbox[1,1] = (np.max(contour_relative[:,1])) + offset_on_screen
            draw.rectangle([bbox[0,0], bbox[0,1], bbox[1,0], bbox[1,1]],
                            fill=None, outline=(255, 255, 0, 255), width=rectwidth)
        elif contourtype == 'polygon':
            draw.polygon(contour_tuples,
                            outline =(255, 255, 0, 255),
                            )
        image.paste(Image.alpha_composite(image, transp))                


        pixmap = QPixmap.fromImage(ImageQt(image))
        pixmap = pixmap.scaled(self.windowsize, self.windowsize, Qt.KeepAspectRatio)
        nuc_ = QLabel()
        nuc_.setAlignment(Qt.AlignCenter)
        nuc_.setPixmap(pixmap)
        #nuc_.setStyleSheet("border: 3px solid darkred; border-radius: 5px;")

        return nuc_

    def review_update_class(self, nuclei_id, new_class, MainWindow):
        case_id, slide_id, nuc_idx = nuclei_id.split('@')
        nuc_idx = int(nuc_idx)
        
        # update MainWindow
        idx_bool = (MainWindow.datamodel.data_info['case_id'] == case_id) &\
                    (MainWindow.datamodel.data_info['slide_id'] == slide_id) &\
                    (MainWindow.datamodel.data_info['nuclei_index'] == nuc_idx)

        MainWindow.datamodel.data_info.loc[idx_bool, 'label_toggle_count'] += 1

        original_class_id = MainWindow.datamodel.data_info.loc[idx_bool, 'original_class_id'].values[0]
        original_class_name = MainWindow.datamodel.classinfo.loc[MainWindow.datamodel.classinfo.index == original_class_id, 'classname'].values[0]
        if new_class == original_class_name:
            MainWindow.datamodel.data_info.loc[idx_bool, 'label_final_value'] = 1
        elif new_class == 'Other':
            MainWindow.datamodel.data_info.loc[idx_bool, 'label_final_value'] = 0
        elif new_class == 'Not Sure':
            MainWindow.datamodel.data_info.loc[idx_bool, 'label_final_value'] = 'Not Sure'
        elif new_class == 'Incorrect Segmentation':
            MainWindow.datamodel.data_info.loc[idx_bool, 'label_final_value'] = 'Incorrect Segmentation'


    def review_finish(self):
        self.MainWindow.log.write('Interactive label: Review annotations finished.')
        print('Update testing accuracy ...')

        if hasattr(self.MainWindow, 'nucstat'):
            self.MainWindow.datamodel.apply2case()
        else:
            datadict = self.MainWindow.datamodel.get_merged_lbl_data(stage='apply2case')

        '''
        datadict = self.MainWindow.datamodel.get_merged_lbl_data(stage='apply2case')
        if datadict['interactive_labeling_proceed_status'] == 'pass':
            # MODEL for all class

            clf = self.MainWindow.datamodel.ML_train(datadict,
                                    modelclass = 'All_class',
                                    trained_for = 'All_class')
        '''
        dict2send = {'action': 'update annotation count',
                    'value': dict(Counter(self.MainWindow.datamodel.data_info['original_class_id']))
                    }
        self.MainWindow.backend.py2js(dict2send) # send total nuclei count, active nuclei count, accuracy, etc.
        self.close()



    def interactive_review_dialog_load_nuclei_dict(self):
        dict2send = {'data_info': self.MainWindow.datamodel.data_info,
                    'classinfo': self.MainWindow.datamodel.classinfo,
                    }
        # Those files were used to train that model
        svs_filepath_counter = Counter(dict2send['data_info']['filepath']).most_common()
        self.uniq_svs_files = [v[0] for v in svs_filepath_counter]
        number_of_nuclei_annotated = [float(v[1]) for v in svs_filepath_counter]
        
        file_exist_list = []
        for filepath in self.uniq_svs_files:
            file_exist_list.append(os.path.exists(filepath))

        '''
        Some files are not existed in this computer.
        Did you annotate nuclei from another machine and import to this computer?
        If yes, then you need to manualy change the filepath.
        '''
        newdict = {'action': 'need_update_file_locations',
                    'original_filepath_list': self.uniq_svs_files,
                    'number_of_nuclei_annotated': number_of_nuclei_annotated,
                    'file_exist_list': list(np.array(file_exist_list).astype(str)),
                    }
        self.backend_review.py2js(newdict)


        if np.all(file_exist_list):
            '''
            All file exists in this computer.
            '''
            newdict = {}
            newdict['action'] = 'nuclei_dict_for_review'
            newdict['class_id'] = list(dict2send['classinfo'].index.astype(float)) # must be float to send json
            newdict['class_name'] = list(dict2send['classinfo']['classname'].values.astype(str))

            self.svs_pointers = {}
            for p in self.uniq_svs_files:
                self.svs_pointers[p] = RotatableOpenSlide(p, rotation=0)
            newdict
            newdict['nuclei_id'] = []
            newdict['nuclei_img_small_paths'] = []
            newdict['nuclei_img_original_paths'] = []
            for i in tqdm(dict2send['data_info'].index):
                nuclei_id = '%s@%s@%d' % (dict2send['data_info'].loc[i,'case_id'],
                                            dict2send['data_info'].loc[i,'slide_id'],
                                            dict2send['data_info'].loc[i,'nuclei_index'])

                newdict['nuclei_id'].append(nuclei_id)
                
                # convert image patch to base64
                svs = self.svs_pointers[dict2send['data_info'].loc[i,'filepath']]
                centroid_x, centroid_y = dict2send['data_info'].loc[i,['centroid_x', 'centroid_y']]
                contour_str = dict2send['data_info'].loc[i,'contour'].replace('\n','').replace(' ',',')
                contour_str = contour_str.replace(',,',',').replace('[,','[').replace(',]',']')
                contour_str = contour_str.replace(',,',',')
                contour = np.array(eval(contour_str)).astype(np.uint32)
                
                patchsize = 128
                offset = 10
                x_min, y_min = np.min(contour, axis=0)
                x_max, y_max = np.max(contour, axis=0)
                x_min, x_max = x_min-(centroid_x-int(patchsize/2))-offset, x_max-(centroid_x-int(patchsize/2))+offset
                y_min, y_max = y_min-(centroid_y-int(patchsize/2))-offset, y_max-(centroid_y-int(patchsize/2))+offset
                x_min, y_min = np.max([0, x_min]), np.max([0, y_min])
                x_max, y_max = np.min([patchsize-1, x_max]), np.min([patchsize-1, y_max])

                patch = svs.read_region((centroid_x-int(patchsize/2), centroid_y-int(patchsize/2)),
                                        level=0,
                                        size=(patchsize, patchsize))
                patch = Image.fromarray(patch[..., :3])

                patch_draw = ImageDraw.Draw(patch)
                shape = [(x_min, y_min), (x_max, y_max)]
                patch_draw.rectangle(shape, fill=None, outline='red', width=2)


                thumbnail_small_base64 = str(im_2_b64(patch))
                thumbnail_small_base64 = thumbnail_small_base64.lstrip('b\'').rstrip('\'')
                newdict['nuclei_img_small_paths'].append(thumbnail_small_base64)

            newdict['nuclei_class_id'] = list(dict2send['data_info']['original_class_id'].values.astype(float))
            newdict['nuclei_label_original'] = list(dict2send['data_info']['label_final_value'].values.astype(str))
            newdict['nuclei_label_type'] = list(dict2send['data_info']['label_type'].values.astype(str))
            newdict['label_init_datetime'] = list(dict2send['data_info']['label_init_datetime'].values.astype(str))

        self.backend_review.py2js(newdict)
        





    def init_active_learning_from_python(self):
        dict2send = {'action': 'setup_active_learning'}
        dict2send['class_id'] = list(self.MainWindow.datamodel.classinfo.index.astype(float)) # must be float to send json
        dict2send['class_name'] = list(self.MainWindow.datamodel.classinfo['classname'].values.astype(str))
        
        # get top most uncertain nuclei
        top_n = 80
        if not hasattr(self.MainWindow, 'nucstat'):
            return
        elif (self.MainWindow.datamodel.data_info is None) or \
            (len(self.MainWindow.datamodel.data_info) == 0):
            print('data_info is none. Get random nuclei')
            random_i = np.random.choice(len(self.MainWindow.nucstat.index), top_n)
            list_of_nuclei_id = list(self.MainWindow.nucstat.index[list(random_i)].astype(float))
            list_of_proba = list([0]*len(list_of_nuclei_id))
            list_of_estimated_label = list([0]*len(list_of_nuclei_id))

        else:
            if not hasattr(self.MainWindow.nucstat, 'prediction'):
                self.MainWindow.datamodel.apply2case()

            # sort estimated_proba
            prediction = copy.deepcopy(self.MainWindow.nucstat.prediction)
            prediction['new_proba'] = (prediction['proba']-0.5).abs()
            top_uncertain = prediction.sort_values(by=['new_proba'], ascending=True).head(top_n)

            list_of_estimated_label = list(top_uncertain['label'].values)
            list_of_proba = list(top_uncertain['proba'].values)
            list_of_nuclei_id = list(top_uncertain.index.values.astype(float))

        nuclei_img_small_paths = []
        for idx in np.array(list_of_nuclei_id).astype(int):
            centroid = self.MainWindow.nucstat.centroid[self.MainWindow.nucstat.index==idx,...]
            centroid_x, centroid_y = centroid.reshape(-1)
            contour = self.MainWindow.nucstat.contour[self.MainWindow.nucstat.index==idx,...][0]

            patchsize = 128
            offset = 10
            x_min, y_min = np.min(contour, axis=0)
            x_max, y_max = np.max(contour, axis=0)
            x_min, x_max = x_min-(centroid_x-int(patchsize/2))-offset, x_max-(centroid_x-int(patchsize/2))+offset
            y_min, y_max = y_min-(centroid_y-int(patchsize/2))-offset, y_max-(centroid_y-int(patchsize/2))+offset
            x_min, y_min = np.max([0, x_min]), np.max([0, y_min])
            x_max, y_max = np.min([patchsize-1, x_max]), np.min([patchsize-1, y_max])

            patch = self.MainWindow.slide.read_region((centroid_x-int(patchsize/2), centroid_y-int(patchsize/2)),
                                    level=0,
                                    size=(patchsize, patchsize))
            patch = Image.fromarray(patch[..., :3])

            patch_draw = ImageDraw.Draw(patch)
            shape = [(x_min, y_min), (x_max, y_max)]
            patch_draw.rectangle(shape, fill=None, outline='red', width=2)

            thumbnail_small_base64 = str(im_2_b64(patch))
            thumbnail_small_base64 = thumbnail_small_base64.lstrip('b\'').rstrip('\'')
            nuclei_img_small_paths.append(thumbnail_small_base64)

        dict2send['estimated_nuclei_class_id'] = list_of_estimated_label
        dict2send['estimated_nuclei_proba'] = list_of_proba
        dict2send['list_of_nuclei_id'] = list_of_nuclei_id
        dict2send['nuclei_img_small_paths'] = nuclei_img_small_paths


        self.backend_review.py2js(dict2send)