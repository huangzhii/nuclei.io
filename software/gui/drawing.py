#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 08/10/2022
#############################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect, QLine,
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
from gui.utils import array2d_to_qpolygonf

class Draw(QWidget):

    def __init__(self,
                    MainWindow):
        super(Draw, self).__init__()
        self.MainWindow = MainWindow
        self.coordinates = None
        self.shape = None

    def refresh(self,
                widget='DrawingOverlay'):
        print('refresh')
        print(self.coordinates)
        print(self.shape)
        
        if (self.coordinates is not None) and (self.shape is not None):
            self.drawing(self.shape,
                        self.coordinates,
                        widget)
        else:
            pixmap = QPixmap(self.MainWindow.ui.DrawingOverlay.size())
            pixmap.fill(Qt.transparent)
            self.MainWindow.ui.DrawingOverlay.setPixmap(pixmap)

    def drawing(self,
            shape,
            coordinates,
            widget='DrawingOverlay'):

        self.coordinates = coordinates
        self.shape = shape
        if shape == 'Ellipse':
            self.drawEllipse(coordinates)
        elif shape == 'Rect':
            self.drawRect(coordinates)
        elif shape == 'Polygon':
            self.drawPolygon(coordinates)

    def drawRect(self, coordinates):
        '''
        Coordinates: [(x1,y1), (x2,y2)]
        x1, y1: that is center of mouse start
        x2, y2: that is the current location of mouse
        '''
        pt1, pt2 = coordinates
        thickness = 1
        x1, x2 = min((pt1[0], pt2[0])), max((pt1[0], pt2[0]))
        y1, y2 = min((pt1[1], pt2[1])), max((pt1[1], pt2[1]))
        width = x2-x1
        height = y2-y1

        pixmap = QPixmap(self.MainWindow.ui.DrawingOverlay.size())
        pixmap.fill(Qt.transparent)
        painter = QPainter()
        painter.begin(pixmap)

        color_fill = QColor(255,0,0,100)

        painter.setPen(QPen(Qt.black, thickness, Qt.SolidLine))
        painter.setBrush(QBrush(color_fill))
        painter.drawRect(QRect(x1, y1, width, height))

        painter.end()
        self.MainWindow.ui.DrawingOverlay.setPixmap(pixmap)

    def drawEllipse(self, coordinates):
        '''
        Coordinates: [(x1,y1), (x2,y2)]
        x1, y1: that is center of mouse start
        x2, y2: that is the current location of mouse
        '''
        pt1, pt2 = coordinates
        thickness = 1
        width = int(np.abs(pt2[0]-pt1[0]))
        height = int(np.abs(pt2[1]-pt1[1]))

        x1 = pt1[0]-width
        y1 = pt1[1]-height

        pixmap = QPixmap(self.MainWindow.ui.DrawingOverlay.size())
        pixmap.fill(Qt.transparent)
        painter = QPainter()
        painter.begin(pixmap)

        color_fill = QColor(255,0,0,100)

        # Add auxilary line.
        painter.setPen(QPen(Qt.yellow, thickness, Qt.DashLine))
        painter.setBrush(QBrush(QColor(0,0,0,0)))
        painter.drawRect(QRect(x1, y1, width*2, height*2))
        # draw circle
        painter.setPen(QPen(Qt.black, thickness, Qt.SolidLine))
        painter.setBrush(QBrush(color_fill))
        painter.drawEllipse(x1, y1, width*2, height*2)

        painter.end()
        self.MainWindow.ui.DrawingOverlay.setPixmap(pixmap)


    def drawPolygon(self,
                    coordinates: pd.DataFrame):
        '''
        coordinates:
                x_screen  y_screen  x_slide  y_slide
            0         1         1       10       10
            1         1         1       10       10
        '''
        #print(coordinates)
        x, y, width, height = self.MainWindow.region[0][0], self.MainWindow.region[0][1], self.MainWindow.region[1][0], self.MainWindow.region[1][1]
        contour = coordinates.loc[:, ['x_slide', 'y_slide']].values.astype(int)
        # Convert global position to screen.
        # The reason we don't use screen position is because users may drag/move the slide.
        contour[:,0] = np.int32((contour[:,0]-x)/self.MainWindow.currentZoom)+1 # don't use uint32, negative value will overflow.
        contour[:,1] = np.int32((contour[:,1]-y)/self.MainWindow.currentZoom)+1
        contour_qpolygon = array2d_to_qpolygonf(contour[:,0], contour[:,1])


        thickness = 2


        pixmap = QPixmap(self.MainWindow.ui.DrawingOverlay.size())
        pixmap.fill(Qt.transparent)
        painter = QPainter()
        painter.begin(pixmap)
        color_fill = QColor(255,0,0,100)
        painter.setPen(QPen(color_fill, thickness, Qt.SolidLine))
        painter.setBrush(QBrush(color_fill))
        painter.drawPolygon(contour_qpolygon)




        color_fill = QColor(255,255,0,255)
        # draw auxilary rectangle
        index_list = coordinates.index
        for i in range(len(index_list)-1):
            x1, y1 = coordinates.loc[index_list[i], ['x_slide', 'y_slide']]
            x2, y2 = coordinates.loc[index_list[i+1], ['x_slide', 'y_slide']]
            x1 = np.int32((x1-x)/self.MainWindow.currentZoom)+1 # don't use uint32, negative value will overflow.
            x2 = np.int32((x2-x)/self.MainWindow.currentZoom)+1
            y1 = np.int32((y1-y)/self.MainWindow.currentZoom)+1
            y2 = np.int32((y2-y)/self.MainWindow.currentZoom)+1
            painter.setPen(QPen(Qt.black, thickness, Qt.SolidLine))
            painter.setBrush(QBrush(color_fill))
            painter.drawRect(QRect(x1-3, y1-3, 6, 6)) # x1, y1, width, height
        # draw last auxilary rectangle
        #painter.setPen(QPen(Qt.black, thickness, Qt.SolidLine))
        #painter.setBrush(QBrush(color_fill))
        #painter.drawRect(QRect(x2-3, y2-3, 6, 6)) # x1, y1, width, height


        painter.end()
        self.MainWindow.ui.DrawingOverlay.setPixmap(pixmap)


    def completePolygon(self,
                        coordinates: pd.DataFrame):

        '''
        coordinates:
                x_screen  y_screen  x_slide  y_slide
            0         1         1       10       10
            1         1         1       10       10
        '''
        print('Complete.')
        #print(coordinates)
        if hasattr(self.MainWindow.ui, 'drawPolygonPoints'):
            delattr(self.MainWindow.ui, 'drawPolygonPoints')
        self.MainWindow.ui.polygon_in_progress = False
        self.clearDrawing()

    def cancelPolygon(self):
        if (self.MainWindow.ui.mode=='Polygon'):
            '''
            reply = QtWidgets.QMessageBox.question(self, 'Question',
                                            'Do you want to cancel your polygon annotation?', QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

            if reply == QtWidgets.QMessageBox.No:
                return
            '''
            
            if hasattr(self.MainWindow.ui, 'drawPolygonPoints'):
                delattr(self.MainWindow.ui, 'drawPolygonPoints')
        self.MainWindow.ui.polygon_in_progress = False
        self.clearDrawing()
            
    def removeLastPolygonPoint(self):
        if (self.MainWindow.ui.mode=='Polygon'):
            self.MainWindow.ui.drawPolygonPoints = self.MainWindow.ui.drawPolygonPoints.drop(self.MainWindow.ui.drawPolygonPoints.tail(1).index)
        self.drawPolygon(coordinates=self.MainWindow.ui.drawPolygonPoints)

    def clearDrawing(self):
        pixmap = QPixmap(self.MainWindow.ui.DrawingOverlay.size())
        pixmap.fill(Qt.transparent)
        self.MainWindow.ui.DrawingOverlay.setPixmap(pixmap)
