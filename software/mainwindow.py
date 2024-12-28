#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 07/25/2022
#############################################################################

from PySide6.QtCore import (QDir, QFile, QIODevice, QUrl, Qt, Slot, Signal, QPointF,
    QPoint, QEvent, QObject, QThread)
from PySide6.QtWidgets import (QWidget, QDialog, QFileDialog, QMainWindow, QMessageBox)

from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QNativeGestureEvent,
    QTransform)
from ui_mainwindow import Ui_MainWindow

# Import necessary tools from folders
from slide_reader.slideReader import RotatableOpenSlide, SlideReader, SlideImageReceiverThread
from system_log.log_error import Log, print_error
from webengine.webengine_setup import webengine_setup
from gui import show_image
from gui import show_overlay_ML
from gui import show_overlay_annotation
from gui import thumbnail
from gui import scrollbar
from gui import mouseEvents
from gui import keyEvents
from gui import drawing
from gui import dialog

from machine_learning import slide_statistics, data_model
from annotation import slide_annotations
from webengine import connect_db_ssh
from user import user_manager

import os, sys
import numpy as np
import tiffslide
import cv2
from functools import partial
import traceback
from PIL import Image
from matplotlib import path
from natsort import os_sorted
from gui.utils import im_2_b64
opj = os.path.join


class MainWindow(QMainWindow):

    showImageRequest = Signal(np.ndarray, int)
    showImage3Request = Signal(np.ndarray, int)
    readRegionCompleted = Signal(np.ndarray, int)
    processingStep = 0

    def __init__(self, wd, parent=None):
        super().__init__(parent)
        self.wd = wd # working directory

        # logging
        self.log = Log(self)
        self.log.write('Start program *** Start program')

        #############################################
        # UI setup
        #############################################
        self.ui = Ui_MainWindow()
        self.ui.setupUI(self)
        self.mainImageSize = np.asarray([self.ui.MainImage.frameGeometry().width(),self.ui.MainImage.frameGeometry().height()])
        
        self.CHROME_DEBUG = False
        self.currentZoom = 1
        self.rotation = 0 # WSI rotation from dial knob. Initial value: 0
        self.region = [[0,0],[0,0]]
        self.mouse_offset_x = 0
        self.mouse_offset_y = 0
        self.mouse_loc_x = 0
        self.mouse_loc_y = 0
        self.case_id = ''
        self.slide_id = ''
        self.imageOpened = False

        #############################################
        # Initialize slide reader
        #############################################

        self.slideReaderThread = SlideReader()
        self.slideReaderThread.start()
        self.slideImageReceiverThread = SlideImageReceiverThread(self, readerqueue=self.slideReaderThread.outputQueue)
        self.slideImageReceiverThread.setDaemon(True)
        self.slideImageReceiverThread.start()
        
        self.showImageRequest.connect(partial(show_image.showImage_part2, self))
        self.showImage3Request.connect(partial(show_image.showImage_part3, self))
        self.readRegionCompleted.connect(partial(show_image.showImage_part2, self))

        #############################################
        # Initialize mouse and keyboard events
        #############################################
        self.toggle_ML_Overlay = False
        self.toggle_Annotation_Overlay = False

        self.ui.MainImage.setMouseTracking(True)

        self.keyPressEvent = partial(keyEvents.whenKeyPressed, self)
        self.keyReleaseEvent = partial(keyEvents.whenKeyReleased, self)
        self.wheelEvent = partial(mouseEvents.wheelEvent,self)
        self.mousePressEvent = partial(mouseEvents.pressImage, self)
        self.mouseReleaseEvent = partial(mouseEvents.releaseImage, self)
        self.mouseMoveEvent = partial(mouseEvents.moveImage,self)
        self.mouseDoubleClickEvent = partial(mouseEvents.doubleClick,self)

        self.installEventFilter(self) # https://doc.qt.io/qtforpython/overviews/eventsandfilters.html

        self.ui.OverviewLabel.mousePressEvent = partial(thumbnail.pressOverviewImage, self)

        #############################################
        # webengine setup
        #############################################

        self.backend = webengine_setup(self)
        

        #############################################
        # Menu action setup
        #############################################
        self.ui.actionOpen.triggered.connect(self.onFileOpen)
        self.ui.actionOpenCase.triggered.connect(self.onCaseFolderOpen)
        self.ui.actionExit.triggered.connect(self.close)


        #############################################
        # Tool action setup
        #############################################
        self.showImage = partial(show_image.showImage, self)
        self.showImage_part2 = partial(show_image.showImage_part2, self)
        self.showOverlayML = partial(show_overlay_ML.showOverlayML, self)
        self.showOverlayAnnotation = partial(show_overlay_annotation.showOverlayAnnotation, self)
        self.connect_to_DB = partial(connect_db_ssh.connect_to_DB, self)
        self.connect_to_SSH = partial(connect_db_ssh.connect_to_SSH, self)
        self.draw = drawing.Draw(self)
        self.user = user_manager.User(self)
        self.datamodel = data_model.DataModel(self)
        self.webdialog = dialog.WebDialog(self)

        self.ui.rotation_dial.valueChanged.connect(partial(self.ui.rotation_dial.value_changed, self))

        self.IHC_Eval = dialog.IHC_Evaulation_Dialog(self)

        self.changeScrollbars = partial(scrollbar.changeScrollbars, self)
        self.updateScrollbars = partial(scrollbar.updateScrollbars, self)
        self.ui.horizontalScrollBar.valueChanged.connect(self.changeScrollbars) # valueChanged passes (self, value)
        self.ui.verticalScrollBar.valueChanged.connect(self.changeScrollbars) # valueChanged passes (self, value)



        #############################################
        # Slide/case management setup
        #############################################
        self.all_case_folders = {}
        
        return


    def openLocalSlide(self, filepath):
        self.ui.statusBar.message.setText("Open file ...")
        self.slide = RotatableOpenSlide(filepath, rotation=self.rotation)
        #thumbnail = self.slide.associated_images['thumbnail']
        self.slideDescription = self.slide.properties['tiff.ImageDescription']
        self.slideDescription = self.slideDescription.replace('\r', ' ').replace('\n', ' ')
        if (tiffslide.PROPERTY_NAME_OBJECTIVE_POWER in self.slide.properties):
            self.slideMagnification = self.slide.properties[tiffslide.PROPERTY_NAME_OBJECTIVE_POWER]
            if self.slideMagnification is None:
                print('Slide magnification is None. now hard-coded to 60.')
                self.slideMagnification = 60
        else:
            self.slideMagnification = 1
        if (tiffslide.PROPERTY_NAME_MPP_X in self.slide.properties):
            self.slideMicronsPerPixel = self.slide.properties[tiffslide.PROPERTY_NAME_MPP_X]
        elif (hasattr(self.slide,'mpp_x')):
            self.slideMicronsPerPixel = self.slide.mpp_x
        else:
            self.slideMicronsPerPixel = 1E-6
        self.zPosition = 0
        self.imageOpened = True
        self.relativeCoords = np.asarray([0,0], np.float32)
        self.updateOverview()

        self.annotation = slide_annotations.SlideAnnotations(self)
        #self.annotation.load_annotation_from_database()

        self.nucstat = self.openProcessedData(filepath)

        if self.nucstat is not None:
            self.datamodel.apply_to_case()


        self.imageCenter=[0,0]
        if self.getMaxZoom() > 80:
            zoomv = 160
        else:
            zoomv = self.getMaxZoom()
        self.setZoomValue(zoomv)
        self.showImage()

    def openProcessedData(self, slide_filepath):
        self.log.write('Open processed data')
        data_folder = os.path.dirname(slide_filepath)
        slidename = os.path.basename(slide_filepath)
        self.stat_folder = None
        for d in os.listdir(data_folder):
            if d in ['stardist_results', 'statistics']:
                self.stat_folder = os.path.join(data_folder, d)
                break
        if self.stat_folder is None:
            print('No processed data (no stardist_results)')
            return None

        '''
        Another format to aggregate multiple slides' stat info into one folder
        '''
        slidename_no_svs = slidename.rstrip('.svs')
        if (slidename_no_svs in os.listdir(self.stat_folder)) and \
            os.path.isdir(os.path.join(self.stat_folder, slidename_no_svs)):
            self.stat_folder = os.path.join(self.stat_folder, slidename_no_svs)

        stat = slide_statistics.SlideStatistics(self, self.stat_folder, preload=True)

        
        if hasattr(stat, 'feature_columns'):
            dict2send = {}
            dict2send['action'] = 'update feature list'
            fl = ['%s | %s' % (v[0], v[1]) for v in stat.feature_columns]
            dict2send['feature_list'] = fl
            self.backend.py2js(dict2send)

        

        return stat


    def SVS_clicked(self, case_val, slide_val):
        self.case_id = case_val
        self.slide_id = slide_val
        self.slide_info = self.all_case_folders[case_val]['case_dict'][slide_val]
        self.slide_filepath = self.slide_info['svs_fname']
        self.openLocalSlide(self.slide_filepath)




    def keyPressEvent(self, event):
        if self.imageOpened:
            keyEvents.whenKeyPressed(self, event)
    def keyReleaseEvent(self, event):
        if self.imageOpened:
            keyEvents.whenKeyReleased(self, event)

    def eventFilter(self, source, event):
        
        if event.type() == QEvent.HoverMove:
            if event.buttons() == Qt.NoButton:
                self.mouseMoveEvent(event)
                #pos = event.pos()
                #print('Mouse hover on: x: %d, y: %d' % (pos.x(), pos.y()))
            else:
                pass

        if not (self.imageOpened):
            return QWidget.eventFilter(self, source, event)

        if (isinstance(event, QNativeGestureEvent)):
            
            if (event.gestureType()== Qt.BeginNativeGesture):
                self.eventIntegration=0
                self.eventCounter=0

            if (event.gestureType()== Qt.ZoomNativeGesture):
                self.eventIntegration+=event.value()
                self.eventCounter+= 1

            if ((event.gestureType()== Qt.EndNativeGesture) or
                ((event.gestureType() == Qt.ZoomNativeGesture) and (self.eventCounter>5))):
                self.setZoomValue(self.getZoomValue() * np.power(1.25, -self.eventIntegration*5))
                self.eventIntegration = 0
                self.eventCounter = 0
                self.showImage()

        return QWidget.eventFilter(self, source, event)



    def setCenter(self, target):
        self.setCenterTo(target[0], target[1])
        self.showImage()

    def setCenterTo(self,cx,cy):
        if (self.imageOpened):

            image_dims=self.slide.level_dimensions[0]
            self.relativeCoords = np.asarray([cx/image_dims[0], cy/image_dims[1]])

            if (self.relativeCoords[1]>1.0):
                self.relativeCoords[1]=1.0

            self.relativeCoords -= 0.5

            self.mouse_slide_x = cx
            self.mouse_slide_y = cy
    
    def updateCenterfromRelativeCoord(self):
        slidecenter = np.asarray(self.slide.level_dimensions[0])/2
        imgarea_p1 = slidecenter - self.mainImageSize * self.getZoomValue() / 2 + self.relativeCoords*slidecenter*2
        imgarea_w =  self.mainImageSize * self.getZoomValue()
        self.mouse_slide_x = imgarea_p1[0] + imgarea_w[0]/2
        self.mouse_slide_y = imgarea_p1[1] + imgarea_w[1]/2
        

    def updateOverview(self):
        self.thumbnail = thumbnail.Thumbnail(self.slide)

        # Read overview thumbnail from slide
        if 32 in self.slide.level_downsamples:
            level_overview = np.where(np.array(self.slide.level_downsamples)==32)[0][0] # pick overview at 32x
        else:
            level_overview = self.slide.level_count-1
        overview = self.slide.read_region(location=(0,0),
                                            level=level_overview,
                                            size=self.slide.level_dimensions[level_overview],
                                            zLevel=self.zPosition)
        self.slideOverview = np.asarray(overview)
        overview = cv2.cvtColor(np.asarray(overview), cv2.COLOR_BGRA2RGB)
        self.overview = overview

    def screenToSlide(self,co):
        """
            convert screen coordinates to slide coordinates
        """
        p1 = self.region[0]
        xpos = int(co[0] * self.getZoomValue() + p1[0])
        ypos = int(co[1] * self.getZoomValue() + p1[1])
        return (xpos,ypos)

    def slideToScreen(self,pos):
        """
            convert slide coordinates to screen coordinates
        """
        xpos,ypos = pos
        p1 = self.region[0]
        cx = int((xpos - p1[0]) / self.getZoomValue())
        cy = int((ypos - p1[1]) / self.getZoomValue())
        return (cx,cy)


    def mapToTruePos(self,
                    event_pos,
                    wheelEvent=False):
        '''
        Map from event position to canvas position
        '''
        '''
        if not hasattr(self, 'mouse_offset_x'):
            self.mouse_offset_x = 0
            self.mouse_offset_y = 0
        else:
        '''
        self.mouse_offset_x = 0
        self.mouse_offset_y = -self.ui.toolBar.size().height()#/2
        #TODO: exact mouse" zoom in not fixed.
        #TODO: Task for GUI folks.

        if wheelEvent:
            # event_pos = PySide6.QtGui.QWheelEvent
            new_x = event_pos.position().x() - (self.ui.MainImage.pos().x()+self.mouse_offset_x)
            new_y = event_pos.position().y() - (self.ui.MainImage.pos().y()+self.mouse_offset_y)
            new_pos = (new_x, new_y)
        else:
            event_pos = self.mapToGlobal(event_pos)
            #print("mouse_offset_y", self.mouse_offset_y)
            new_x = event_pos.x() + self.ui.MainImage.pos().x()+self.mouse_offset_x
            new_y = event_pos.y() + self.ui.MainImage.pos().y()+self.mouse_offset_y
            new_pos = QPoint(new_x, new_y)

        return new_pos

    def setZoomValue(self, zoomValue):
        """
            Sets the zoom of the current image.
        """

        self.currentZoom = zoomValue
        if (self.currentZoom < 0.25):
            self.currentZoom = 0.25
        maxzoom = self.getMaxZoom()
        
        if (self.currentZoom > maxzoom):
            self.currentZoom = maxzoom

        sliderVal = 100*np.log2(self.currentZoom/(maxzoom))/(np.log2(0.25/maxzoom))

        self.ui.zoomSlider.valueChanged.disconnect()
        self.ui.zoomSlider.setValue(sliderVal)
        self.ui.zoomSlider.valueChanged.connect(self.ui.zoomSlider.sliderChanged)
        if (self.currentZoom<1):
            self.ui.zoomSlider.setText('(%.1f x)' % (float(self.slideMagnification)/self.currentZoom))
        else:
            self.ui.zoomSlider.setText('%.1f x' % (float(self.slideMagnification)/self.currentZoom))

    def getZoomValue(self):
        """
            returns the current zoom value
        """
        return self.currentZoom


    def getMaxZoom(self):
        """
            Returns the maximum zoom available for this image.
        """
        return max(self.slide.level_dimensions[0][0] / self.mainImageSize[0],self.slide.level_dimensions[0][1] / self.mainImageSize[1]) * 1.2

    def setAsCenter(self, cx, cy):
        '''
        cx, cy: coordinates on WSI (global position)
        '''
        self.setCenterTo(cx, cy)
        self.showImage()

    def clear_subset_nuclei(self):
        '''
        Instead of showing the subset nuclei from the virtual flow cytometry,
        show all nuclei on the entire WSI.
        '''
        if hasattr(self, 'nucstat') and self.nucstat is not None:
            self.nucstat.isSelected_to_VFC[:] = True
            self.nucstat.isSelected_from_VFC[:] = True
            self.datamodel.send_dim_info_for_VFC()
            self.showImage()

    def open_visualization_setting(self):
        print("TODO: Open visualization setting.")


    def resizeEvent(self, event):
        QMainWindow.resizeEvent(self, event)
        if self.imageOpened:
            self.showImage()

    def onSplitterMoved(self):
        #print('Splitter moved')
        if self.imageOpened:
            self.showImage()

    def toggleOverlay(self, toggle_name):
        print('Toggle overlay', toggle_name)
        if toggle_name == 'ML Overlay':
            #self.ui.toggleBtn.toggle_ml.setCheckState(Qt.CheckState.Checked)
            self.showOverlayML()
        if toggle_name == 'Annotation Overlay':
            #self.toggle_Annotation_Overlay = isChecked
            print(self.ui.toggleBtn.toggle_a.checkState())
            self.showOverlayAnnotation()


    def review_annotations(self):
        from active_learning import review_annotation
        print('Review annotations and active learning')
        self.log.write('Interactive label *** Review annotations/active learning start.')
        self.interactive_review = review_annotation.InteractiveReviewDialog(self)
        self.interactive_review.show() # if use exec(), the webengine will not show properly.


    @Slot()
    def onFileOpen(self):
        dialog = QFileDialog(self)
        dialog.setWindowTitle("Open Slide File")
        dialog.setAcceptMode(QFileDialog.AcceptOpen)
        if dialog.exec() == QDialog.Accepted:
            self.openFile(dialog.selectedFiles()[0])

    @Slot()
    def onCaseFolderOpen(self):
        filename = QFileDialog.getExistingDirectory()
        if (len(filename)==0):
            return ''
        self.openCaseFolder(filename)
        return filename


    @Slot(str)
    def openFile(self, path):
        f = QFile(path)
        name = QDir.toNativeSeparators(path)
        if not f.open(QIODevice.ReadOnly):
            error = f.errorString()
            QMessageBox.warning(self, self.windowTitle(),
                                f"Could not open file {name}: {error}")
            return
        self.slide_filepath = path
        self.case_id = 'NO_CASE'
        self.slide_id = os.path.basename(path)
        self.openLocalSlide(self.slide_filepath)
        self.ui.statusBar.message.setText(f"Opened {name}")


    @Slot(str)
    def openCaseFolder(self, foldername):

        accept_file_types = ['svs', 'tiff', 'tif']
        self.case_id = os.path.basename(os.path.normpath(foldername))
        for c in self.all_case_folders:
            if self.all_case_folders[c]['foldername'] == foldername:
                print('Case already opened.')
                return
        
        self.all_case_folders[self.case_id] = {'foldername': foldername,
                                                'case_dict': {}}
        self.log.write('Open case *** %s' % foldername)

        svs_files_in_root = False
        for d in os.listdir(foldername):
            if d.lower().split('.')[-1] in accept_file_types:
                svs_files_in_root = True

        if svs_files_in_root:
            # multiple svs files, and all results are combined in stardist_results folder.
            cassette_list = [d for d in os.listdir(foldername) if d.lower().split('.')[-1] in accept_file_types]
        else:
            cassette_list = [d for d in os.listdir(foldername) if os.path.isdir(os.path.join(foldername,d))]
        
        cassette_list = os_sorted(cassette_list)


        case_dict = {}
        if svs_files_in_root:
            for f in cassette_list:
                try:
                    if f.lower().split('.')[-1] in accept_file_types:
                        fname = opj(foldername, f)
                        d = f
                        case_dict[d] = {}
                        slide = RotatableOpenSlide(fname, rotation=self.rotation)

                        try:
                            thumbnail = slide.associated_images['thumbnail']
                        except:
                            print('Slide does not have thumbnail.')
                            thumbnail = Image.fromarray(np.ones((100,100,3),dtype=np.uint8)*200)
                        try:
                            label = slide.associated_images['label']
                        except:
                            print('Slide does not have label.')
                            label = Image.fromarray(np.ones((100,100,3),dtype=np.uint8)*200)
                        
                        w, h = thumbnail.size
                        aspect_ratio = w/h
                        w_new, h_new = 50, int(np.round(50/aspect_ratio))
                        thumbnail_resize = thumbnail.resize((w_new, h_new))
                        thumbnail_small_base64 = str(im_2_b64(thumbnail_resize))
                        thumbnail_small_base64 = thumbnail_small_base64.lstrip('b\'').rstrip('\'')

                        # resize
                        w, h = label.size
                        aspect_ratio = w/h
                        w_new, h_new = 180, int(np.round(180/aspect_ratio))
                        label_resize = label.resize((w_new, h_new))
                        label_base64 = str(im_2_b64(label_resize))
                        label_base64 = label_base64.lstrip('b\'').rstrip('\'')
                        try:
                            magnitude = int(slide.properties['aperio.AppMag'])
                        except:
                            print('Slide does not have magnitude.')
                            print('Now temporarily hard-code magnitude = 40')
                            magnitude = 40

                        # check if slide has properties.
                        if hasattr(slide, 'properties'):
                            try:
                                height = int(slide.properties['tiffslide.level[0].height'])
                                width = int(slide.properties['tiffslide.level[0].width'])
                            except Exception as e:
                                print(e)
                                height = int(slide.properties['openslide.level[0].height'])
                                width = int(slide.properties['openslide.level[0].width'])
                        else:
                            raise Exception('Cannot access slide properties.')
                        
                        if 'aperio.Date' in slide.properties:
                            month, day, year = slide.properties['aperio.Date'].split('/')
                            if len(year) == 2: year = '20' + year
                            createtime = slide.properties['aperio.Time']
                            if 'aperio.Time Zone' in slide.properties:
                                timezone = slide.properties['aperio.Time Zone']
                            else:
                                timezone = 'nan'
                        else:
                            month, day, year, timezone = 0, 0, 0, 'nan'
                            createtime = 'nan'
                        
                        case_dict[d]['svs_fname'] = fname
                        case_dict[d]['thumbnail'] = thumbnail
                        case_dict[d]['label'] = label
                        case_dict[d]['thumbnail_small_base64'] = thumbnail_small_base64
                        case_dict[d]['label_base64'] = label_base64
                        case_dict[d]['magnitude'] = magnitude
                        case_dict[d]['height'] = height
                        case_dict[d]['width'] = width
                        case_dict[d]['month'] = int(month)
                        case_dict[d]['day'] = int(day)
                        case_dict[d]['year'] = int(year)
                        case_dict[d]['createtime'] = createtime
                        case_dict[d]['timezone'] = timezone
                    elif f.lower() in ['statistics', 'stardist_results']:
                        case_dict[d]['nuclei_stat'] = os.path.join(foldername, d, f)
                        if 'svs_fname' not in case_dict[d]:
                            case_dict.pop(d, None)
                except Exception as e:
                    print(traceback.format_exc())
                    print(e)
                    pass

        elif not svs_files_in_root:
            for d in cassette_list:
                case_dict[d] = {}
                files = os.listdir(os.path.join(foldername, d))
                for f in files:
                    try:
                        if f.lower().split('.')[-1] in accept_file_types:
                            case_dict[d] = {}
                            fname = foldername + os.sep + d + os.sep + f
                            slide = RotatableOpenSlide(fname, rotation=self.rotation)
                            thumbnail = slide.associated_images['thumbnail']
                            label = slide.associated_images['label']
                            
                            w, h = thumbnail.size
                            aspect_ratio = w/h
                            w_new, h_new = 50, int(np.round(50/aspect_ratio))
                            thumbnail_resize = thumbnail.resize((w_new, h_new))
                            thumbnail_small_base64 = str(im_2_b64(thumbnail_resize))
                            thumbnail_small_base64 = thumbnail_small_base64.lstrip('b\'').rstrip('\'')

                            # resize
                            w, h = label.size
                            aspect_ratio = w/h
                            w_new, h_new = 180, int(np.round(180/aspect_ratio))
                            label_resize = label.resize((w_new, h_new))
                            label_base64 = str(im_2_b64(label_resize))
                            label_base64 = label_base64.lstrip('b\'').rstrip('\'')

                            magnitude = int(slide.properties['aperio.AppMag'])
                            try:
                                height = int(slide.properties['tiffslide.level[0].height'])
                                width = int(slide.properties['tiffslide.level[0].width'])
                            except Exception as e:
                                print(e)
                                height = int(slide.properties['openslide.level[0].height'])
                                width = int(slide.properties['openslide.level[0].width'])
                            month,day,year = slide.properties['aperio.Date'].split('/')
                            if len(year) == 2: year = '20' + year
                            createtime = slide.properties['aperio.Time']
                            timezone = slide.properties['aperio.Time Zone']
                            case_dict[d]['svs_fname'] = fname
                            case_dict[d]['thumbnail'] = thumbnail
                            case_dict[d]['label'] = label
                            case_dict[d]['thumbnail_small_base64'] = thumbnail_small_base64
                            case_dict[d]['label_base64'] = label_base64
                            case_dict[d]['magnitude'] = magnitude
                            case_dict[d]['height'] = height
                            case_dict[d]['width'] = width
                            case_dict[d]['month'] = int(month)
                            case_dict[d]['day'] = int(day)
                            case_dict[d]['year'] = int(year)
                            case_dict[d]['createtime'] = createtime
                            case_dict[d]['timezone'] = timezone

                        if f.lower() in ['statistics', 'stardist_results']:
                            case_dict[d]['nuclei_stat'] = os.path.join(foldername, d, f)
                            if 'svs_fname' not in case_dict[d]:
                                case_dict.pop(d, None)
                    except Exception as e:
                        print(e)
                        pass
        self.all_case_folders[self.case_id]['case_dict'] = case_dict

        dict2send = {'action': 'update_workspace',
                    'case_id': self.case_id,
                    'case_dict': case_dict}
        self.backend.py2js(dict2send) # this is to update workspace layout
    
    
    
    
    def show_highlight_nucleus_from_VFC(self, dim1, dim2):
        idx_order = np.argmin((self.datamodel.VFC_dim1_val - dim1)**2 + (self.datamodel.VFC_dim2_val - dim2)**2)

        x, y = self.nucstat.centroid[idx_order,:]
        idx_raw = self.nucstat.index[idx_order]
        #self.highlight_nuclei = {}
        #self.highlight_nuclei['index'] = idx_raw

        self.log.write('Virtual Flow *** show single-clicked nuclei in WSI. Centroid on slide: (%d, %d).' % (x, y))

        slide_dimension = self.slide.level_dimensions[0] # highest resolution, size around 80000*40000
        self.setZoomValue(1) # the smaller, the larger cell looks like
        self.relativeCoords = np.array([x/slide_dimension[0],
                                                y/slide_dimension[1]]).astype(float)
        if (self.relativeCoords[1]>1.0):
            self.relativeCoords[1]=1.0

        self.relativeCoords -= 0.5
        
        image_dims=self.slide.level_dimensions[0]
        self.ui.statusBar.message.setText('Position: (%d,%d)' % (int((self.relativeCoords[0]+0.5)*image_dims[0]), int((self.relativeCoords[1]+0.5)*image_dims[1])))
        self.processingStep = self.id+1
        self.showImage()
        self.updateScrollbars()


    def show_selected_nuclei_from_VFC(self, selectdict):
        if selectdict['type'] == 'lasso':
            x = np.array(selectdict['points_x']).astype(float)
            y = np.array(selectdict['points_y']).astype(float)
            polypoints = np.c_[x,y]
            poly_path = path.Path(polypoints)
            index_bool = poly_path.contains_points(np.c_[self.datamodel.VFC_dim1_val, self.datamodel.VFC_dim2_val])


        elif selectdict['type'] == 'rect':
            x = np.array(selectdict['points_x']).astype(float)
            y = np.array(selectdict['points_y']).astype(float)
            xmin, xmax = np.min(x), np.max(x)
            ymin, ymax = np.min(y), np.max(y)
            index_bool = (self.datamodel.VFC_dim1_val > xmin) & (self.datamodel.VFC_dim1_val < xmax) & (self.datamodel.VFC_dim2_val > ymin) & (self.datamodel.VFC_dim2_val < ymax)
            
        self.nucstat.isSelected_from_VFC = index_bool
        self.processingStep = self.id+1
        self.showImage_part2(self.npi, self.processingStep)

    def closeEvent(self, event):
        print('Program closed.')
        self.log.write('Program closed.')
        '''
        if self.isModified():
            m = "You have unsaved changes. Do you want to exit anyway?"
            button = QMessageBox.question(self, self.windowTitle(), m)
            if button != QMessageBox.Yes:
                event.ignore()
            else:
                event.accept()
        '''
