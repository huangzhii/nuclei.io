#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 07/31/2022
#############################################################################

from matplotlib import widgets
from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)

from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient, QWheelEvent,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient, QPen,
    QTransform)

from PySide6.QtWidgets import (QApplication, QHBoxLayout, QGridLayout, QStackedLayout, QMainWindow, QMenu, QLabel,
    QMenuBar, QPlainTextEdit, QSizePolicy, QSplitter, QScrollBar, QProgressBar, QPushButton, QDial,
    QStatusBar, QWidget)
    
from functools import partial
import numpy as np
import matplotlib.path as path
import cv2
import copy
import os
import pandas as pd
from datetime import datetime
import time

from gui import show_image


def doubleClick(self, event):
    """
        Doubleclick event on the main image
    """
    print('pressImage')
    if (event.button() == Qt.LeftButton):
        leftDoubleClickImage(self,event)
    elif (event.button()==Qt.RightButton):
        rightDoubleClickImage(self,event)

def leftDoubleClickImage(self, event):

    if not (self.imageOpened):
        return
    
    print('leftDoubleClickImage')
    posx,posy = getMouseEventPosition(self, event)
    cx, cy = self.screenToSlide((posx,posy))

    if (self.ui.mode == 'Polygon'):
        newpoint = pd.DataFrame({'x_screen': posx,
                                'y_screen': posy,
                                'x_slide': cx,
                                'y_slide': cy}, index=[0])
        current_polygon = pd.concat([self.ui.drawPolygonPoints, newpoint], ignore_index=True)
        self.draw.completePolygon(current_polygon)

    if (self.ui.mode in ['View', 'Ellipse', 'Rect']):
        self.datamodel.annotate_single_nuclei(posx, posy, cx, cy)
        
    return

def rightDoubleClickImage(self,event):
    print('rightDoubleClickImage')

    return
    
def getMouseEventPosition(self,event):
    """
        Retrieves the current position of a mouse pointer event
    """
    pos = (int(event.pos().x()) + self.mouse_offset_x,
            int(event.pos().y()) + self.mouse_offset_y)
    return pos



def wheelEvent(self, event: QWheelEvent):
    """
        WheelEvent is the callback for wheel events of the operating system.
        Dependent on the OS, this can be a trackpad or a mouse wheel
    """
    self.mouse_loc_x, self.mouse_loc_y = self.mapToTruePos(event, wheelEvent=True)
    self.mouse_slide_x, self.mouse_slide_y = self.screenToSlide((self.mouse_loc_x, self.mouse_loc_y))

    if not (self.imageOpened):
        return
    
    # Disable wheel if x position is leaving image compartment
    if (self.mouse_loc_x < self.ui.MainImage.pos().x()):
        return
    if (self.mouse_loc_y < self.ui.MainImage.pos().y()):
        return
    if (self.mouse_loc_x > (self.ui.MainImage.pos().x() + self.ui.MainImage.size().width()) ):
        return
    if (self.mouse_loc_y > (self.ui.MainImage.pos().y() + self.ui.MainImage.size().height()) ):
        return

    #print('mouse_loc', self.mouse_loc_x, self.mouse_loc_y)
    #print('mouse_slide', self.mouse_slide_x, self.mouse_slide_y)

    self.lastZoomValue = self.getZoomValue()
    if (event.source() == Qt.MouseEventSynthesizedBySystem):
        # touch pad geasture - use direct scrolling, not zooming
        # this is usually indicative for Mac OS
        subs = np.asarray([float(event.pixelDelta().x()), float(event.pixelDelta().y())])/32000.0*self.getZoomValue()
        self.relativeCoords -= subs
        if (self.relativeCoords[0]<-0.5):
            self.relativeCoords[0]=-0.5

        if (self.relativeCoords[1]<-0.5):
            self.relativeCoords[1]=-0.5

    else: # mouse wheel - use scrolling
        inc = -1
        if (event.angleDelta().y()>0):
            inc = +1
        self.setZoomValue(self.getZoomValue() * np.power(1.75, -inc))
    
    self.updateScrollbars()
    self.showImage()



def moveImage(self, event):
    """
        Mouse move event on the main image

    if event.buttons() == Qt.NoButton:
        print("Simple mouse motion")
    elif event.buttons() == Qt.LeftButton:
        print("Left click drag")
    elif event.buttons() == Qt.RightButton:
        print("Right click drag")

    """

    if not (self.imageOpened):
        return

    posx,posy = getMouseEventPosition(self, event)
    cx,cy = self.screenToSlide((posx,posy))

    if event.buttons() == Qt.LeftButton:
        # Move image if shift+left click
        modifiers = QApplication.keyboardModifiers()
        mouse_coord_text = ' Mouse coords: (%d, %d) (%d, %d)' % (posx, posy, cx, cy)
        if 'Mouse coords' not in self.ui.statusBar.message.text():
            new_message = self.ui.statusBar.message.text() + mouse_coord_text
        else:
            new_message = self.ui.statusBar.message.text().split(' Mouse coords:')[0] + mouse_coord_text
        self.ui.statusBar.message.setText(new_message)


        self.log.write('Mouse movement (LeftButton) *** %s' % mouse_coord_text)

        if (modifiers == Qt.ControlModifier) and (self.dragPoint):
            cx,cy = self.screenToSlide((posx,posy))
            self.db.annotations[self.drag_id[0]].coordinates[self.drag_id[1],:] = [cx,cy]
            self.showImage()

        if (modifiers == Qt.ShiftModifier) or (self.ui.clickToMove):
            self.setCursor(Qt.ClosedHandCursor)
            cx,cy = self.screenToSlide((posx,posy))
            anno_abs = self.screenToSlide(self.ui.anno_pt1)
            offsetx = anno_abs[0]-cx
            offsety = anno_abs[1]-cy
            image_dims=self.slide.level_dimensions[0]
            self.relativeCoords += np.asarray([offsetx/image_dims[0], offsety/image_dims[1]])
            self.ui.anno_pt1 = (posx,posy)

            self.mouse_slide_x = cx
            self.mouse_slide_y = cy
            self.updateScrollbars()
            self.showImage()
        #else:
        #    self.setCursor(Qt.ArrowCursor)

        if self.ui.mode in ['Rect', 'Ellipse']:
            pt1 = self.ui.anno_pt1
            pt2 = (posx,posy)
            self.draw.drawing(shape=self.ui.mode,
                        coordinates=[pt1, pt2],
                        widget='DrawingOverlay')

    if event.buttons() == Qt.NoButton:
        if self.ui.mode in ['Polygon']:
            # simple mouse hovering motion

            if hasattr(self.ui, 'drawPolygonPoints'):
                '''
                If self.ui does not have drawPolygonPoints, it might not be start yet, or it has been canceled.
                '''
                pt2 = (posx,posy)
                newpoint = pd.DataFrame({'x_screen': posx,
                                        'y_screen': posy,
                                        'x_slide': cx,
                                        'y_slide': cy}, index=[0])
                current_polygon = pd.concat([self.ui.drawPolygonPoints, newpoint], ignore_index=True)

                self.draw.drawing(shape='Polygon',
                            coordinates=current_polygon,
                            widget='DrawingOverlay')
            




def leftClickImage(self, event):
    """
        Callback function for a left click in the main image
    """
    print('left click image')
    
    if not (self.imageOpened):
        return
    posx, posy = getMouseEventPosition(self, event)
    cx,cy = self.screenToSlide((posx,posy))
    
    self.ui.clickToMove = False
    self.ui.dragPoint = False
    modifiers = QApplication.keyboardModifiers()

    '''
    If mode is Polygon, move image if shift+left click
    If mode is View, then just move image.
    After mouse position recorded, return.
    '''
    if (self.ui.mode == 'View') or \
        ((self.ui.mode == 'Polygon') and (modifiers == Qt.ShiftModifier)):
        self.ui.clickToMove = True
        self.ui.anno_pt1 = (posx, posy)
        self.setCursor(Qt.ClosedHandCursor)
        return
    '''
    If mode is not Polygon or View, then the left click should be some actions.
    '''

    if self.ui.mode in ['Rect', 'Ellipse']:
        '''
        self.ui.anno_pt1 is the starting point (x,y) of Rect and Ellipse annotation.
        The coordinates are respect to the MainImage, not to the WSI.
        '''
        self.ui.anno_pt1 =  (posx, posy)
    
    if self.ui.mode == 'Polygon':
        self.ui.polygon_in_progress = True
        self.ui.anno_pt1 =  (posx, posy)
        if not hasattr(self.ui, 'drawPolygonPoints'):
            self.ui.drawPolygonPoints = pd.DataFrame(columns=['x_screen','y_screen', 'x_slide', 'y_slide'])
        
        # only append if position changed a little on slide:
        if (len(self.ui.drawPolygonPoints)>0) and \
            all([abs(a-p)<3 for a,p in zip((self.ui.drawPolygonPoints.iloc[-1]['x_slide'], self.ui.drawPolygonPoints.iloc[-1]['y_slide']),
                                            getMouseEventPosition(self,event))]):
            return
        
        newpoint = pd.DataFrame({'x_screen': posx,
                                'y_screen': posy,
                                'x_slide': cx,
                                'y_slide': cy}, index=[0])
        self.ui.drawPolygonPoints = pd.concat([self.ui.drawPolygonPoints, newpoint], ignore_index=True)
        
        self.draw.drawing(shape='Polygon',
                    coordinates=self.ui.drawPolygonPoints,
                    widget='DrawingOverlay')





def rightClickImage(self, event):
    """
        Global callback for right click events.
        Dependent on the current mode, the menu displayed has different options. 
    """
    menu = QMenu(self)
    iconfolder = os.path.join(self.wd, 'Artwork', 'canvas_menu')

    if not self.imageOpened:
        addmenu = menu.addAction(QIcon(os.path.join(iconfolder, 'folder-icon.png')), 'Open local slide', self.onFileOpen)
        action = menu.exec_(self.mapToTruePos(event.pos()))
        return


    posx,posy = getMouseEventPosition(self, event)
    cx,cy = self.screenToSlide((posx,posy))

    if (self.ui.mode == 'Polygon'):
        newpoint = pd.DataFrame({'x_screen': posx,
                                'y_screen': posy,
                                'x_slide': cx,
                                'y_slide': cy}, index=[0])
        current_polygon = pd.concat([self.ui.drawPolygonPoints, newpoint], ignore_index=True)

        points = current_polygon[['x_screen', 'y_screen']].values
        points_global = current_polygon[['x_slide', 'y_slide']].values
            
        annotation_dict = {'type': self.ui.mode,
                            'points': points,
                            'points_global': points_global,
                            'rotation': self.rotation}

        if hasattr(self.datamodel, 'classinfo') and self.datamodel.classinfo is not None:
            for cls_id in self.datamodel.classinfo.index:
                cls_name = self.datamodel.classinfo.loc[cls_id, 'classname']
                cls_color = self.datamodel.classinfo.loc[cls_id, 'rgbcolor']
                pixmap = QPixmap(100,100)
                pixmap.fill(QColor(cls_color[0],cls_color[1],cls_color[2]))
                icon = QIcon(pixmap)
                action = menu.addAction(icon,
                                        'Annotate as ' + cls_name,
                                        partial(self.datamodel.annotate_all_nuclei_within_ROI, annotation_dict, cls_id))

        menu.addSeparator()
        #addmenu = menu.addAction('Done', partial(self.draw.completePolygon, current_polygon))
        addmenu = menu.addAction('Cancel', self.draw.cancelPolygon)
        addmenu = menu.addAction('Remove last point', self.draw.removeLastPolygonPoint)
        
        addmenu = menu.addAction(QIcon(os.path.join(iconfolder, 'thumbs-up-line-icon.png')), 'Annotate ROI from prediction', partial(self.datamodel.annotate_ROI_from_prediction, annotation_dict))
        addmenu = menu.addAction(QIcon(os.path.join(iconfolder, 'circle-arrow-icon.png')), 'Select ROI for active learning', partial(self.datamodel.activeLearning, annotation_dict))
        addmenu = menu.addAction(QIcon(os.path.join(iconfolder, 'bar-chart-icon.png')), 'Select ROI for analysis', partial(self.datamodel.analyzeROI, annotation_dict))
        
        #menu.addSeparator()
        #addmenu = menu.addAction('Select ROI for active learning', partial(self.interactive_labeling_proceed, mydict))
        #addmenu = menu.addAction('Select ROI for analysis', partial(self.analyzeROI, mydict))
    
    if (self.ui.mode in ['View']):
        if hasattr(self.datamodel, 'classinfo') and self.datamodel.classinfo is not None:
            for cls_id in self.datamodel.classinfo.index:
                cls_name = self.datamodel.classinfo.loc[cls_id, 'classname']
                cls_color = self.datamodel.classinfo.loc[cls_id, 'rgbcolor']
                pixmap = QPixmap(100,100)
                pixmap.fill(QColor(cls_color[0],cls_color[1],cls_color[2]))
                icon = QIcon(pixmap)
                action = menu.addAction(icon,
                                        'Annotate as ' + cls_name,
                                        partial(self.datamodel.annotate_single_nuclei, posx, posy, cx, cy, cls_id))

    menu.addSeparator()
    iconfolder = os.path.join(self.wd, 'Artwork', 'canvas_menu')
    addmenu = menu.addAction(QIcon(os.path.join(iconfolder, 'target-focus-line-icon.png')), 'Set as center', partial(self.setAsCenter,cx,cy))
    addmenu = menu.addAction(QIcon(os.path.join(iconfolder, 'eye-icon.png')), 'Clear subsetting', partial(self.clear_subset_nuclei))
    addmenu = menu.addAction(QIcon(os.path.join(iconfolder, 'repair-fix-repairing-icon.png')), 'Visualization Setting', partial(self.open_visualization_setting))

    #addmenu = menu.addAction('Reset canvas', self.clear_all_MainWindow_selection)
    #addmenu = menu.addAction('Remove this ROI annotation', partial(self.annotation.removeAnnotation, (posx, posy), (cx, cy)))
    action = menu.exec_(self.mapToTruePos(event.pos()))


def releaseImage(self, event):
    """
        Callback function for a mouse release event in the main image
    """
    self.ui.clickToMove = False
    self.setCursor(Qt.ArrowCursor)

    if self.ui.mode in ['Rect', 'Ellipse', 'Polygon']:

        posx, posy = getMouseEventPosition(self, event)
        iconfolder = os.path.join(self.wd, 'Artwork', 'canvas_menu')

        if self.ui.mode in ['Rect', 'Ellipse']:
            # start/end corner on screen: [[x1, y1], [x2, y2]]
            points = [[self.ui.anno_pt1[0], self.ui.anno_pt1[1]], [posx, posy]]
            # start/end corner on screen: [[x1, y1], [x2, y2]]
            points_global = [self.screenToSlide((self.ui.anno_pt1[0], self.ui.anno_pt1[1])),
                            self.screenToSlide((posx, posy))]
            selected_width = np.abs(points_global[0][0] - points_global[1][0])
            selected_height = np.abs(points_global[0][1] - points_global[1][1])
            print("Selected area width: %d, height: %d" % (selected_width, selected_height))
            if max((selected_width, selected_height)) < 5:
                print('Selected area too small. Will not show menu.')
                return

        if self.ui.mode == 'Polygon':
            # all points of polygon
            points = self.ui.drawPolygonPoints[['x_screen', 'y_screen']].values
            points_global = self.ui.drawPolygonPoints[['x_slide', 'y_slide']].values
            
        annotation_dict = {'type': self.ui.mode,
                            'points': points,
                            'points_global': points_global,
                            'rotation': self.rotation}



        if not self.ui.polygon_in_progress:
            menu = QMenu(self)
            #menuitems = list()
            

            if hasattr(self.datamodel, 'classinfo') and self.datamodel.classinfo is not None:
                for cls_id in self.datamodel.classinfo.index:
                    cls_name = self.datamodel.classinfo.loc[cls_id, 'classname']
                    cls_color = self.datamodel.classinfo.loc[cls_id, 'rgbcolor']
                    pixmap = QPixmap(100,100)
                    pixmap.fill(QColor(cls_color[0],cls_color[1],cls_color[2]))
                    icon = QIcon(pixmap)
                    action = menu.addAction(icon,
                                            'Annotate as ' + cls_name,
                                            partial(self.datamodel.annotate_all_nuclei_within_ROI, annotation_dict, cls_id))

            menu.addSeparator()
            addmenu = menu.addAction(QIcon(os.path.join(iconfolder, 'thumbs-up-line-icon.png')), 'Annotate ROI from prediction', partial(self.datamodel.annotate_ROI_from_prediction, annotation_dict))
            addmenu = menu.addAction(QIcon(os.path.join(iconfolder, 'circle-arrow-icon.png')), 'Select ROI for active learning', partial(self.datamodel.activeLearning, annotation_dict))
            addmenu = menu.addAction(QIcon(os.path.join(iconfolder, 'bar-chart-icon.png')), 'Select ROI for analysis', partial(self.datamodel.analyzeROI, annotation_dict))
            action = menu.exec_(self.mapToTruePos(event.pos())) # after mouse release, auto pop up menu


def pressImage(self, event):
    """
        Callback function for a click on the main image
    """
    self.mouse_offset_x = -self.ui.gridLayout.contentsMargins().left()
    self.mouse_offset_y = -(self.ui.gridLayout.contentsMargins().top() + \
                            self.ui.menuBar.size().height() + \
                            self.ui.toolBar.size().height())

    print('pressImage')
    if (event.button() == Qt.LeftButton):
        leftClickImage(self,event)
    elif (event.button() == Qt.RightButton):
        rightClickImage(self,event)
