#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 08/13/2022
#############################################################################


from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect, QLine,
    QSize, QTime, QUrl, Qt)

from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient, QPen,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QNativeGestureEvent,
    QTransform)

import re
import numpy as np
import pandas as pd
from gui.utils import array2d_to_qpolygonf

def showOverlayAnnotation(self):
    print('****** Show Overlay Annotation')

    '''
    useful resources:
    paint: /home/zhihuang/Desktop/nuclei.io_pyside6/pyside6-examples/widgets/painting/painter/painter.py
    display: /home/zhihuang/Desktop/nuclei.io_pyside6/pyside6-examples/widgets/graphicsview/dragdroprobot/dragdroprobot.py
    '''

    if not self.imageOpened:
        return
    if not hasattr(self.annotation, 'annotation') or \
        self.annotation.annotation is None or \
        len(self.annotation.annotation) == 0:
        #print('No annotation.')
        return

    pixmap = QPixmap(self.ui.AnnotationOverlay.size())
    pixmap.fill(Qt.transparent)

    if self.ui.toggleBtn.toggle_a.checkState() == Qt.CheckState.Unchecked:
        self.ui.AnnotationOverlay.setPixmap(pixmap)
        return


    painter = QPainter()
    painter.begin(pixmap)
    annotation_df = self.annotation.annotation


    for idx in annotation_df.index:
        shapeType = annotation_df.loc[idx,'shapeType']
        classname = annotation_df.loc[idx,'objectClass']
        classcolor = (120,120,120)
        if classname in self.datamodel.classinfo['classname'].values:
            classcolor = self.datamodel.classinfo.loc[self.datamodel.classinfo['classname'].values==classname, 'rgbcolor'].values[0]
            classcolor = (classcolor[0], classcolor[1], classcolor[2])
        coordinates = annotation_df.loc[idx,'coordinates']
        coordinates = coordinates.replace('\n', ' ')
        coordinates = re.sub('\s+', ' ', coordinates) # replace multiple space to one space.
        
        if ',' not in coordinates:
            coordinates = coordinates.replace(' ',',')
        coordinates = coordinates.replace('[,','[').replace(',]',']')
        coordinates = eval('np.array(' + coordinates + ')')
        
        if shapeType in ['rect','Rect']:
            x, y, width, height = self.region[0][0], self.region[0][1], self.region[1][0], self.region[1][1]
            rect_image_x1 = coordinates[0][0]
            rect_image_y1 = coordinates[0][1]
            rect_image_x2 = coordinates[1][0]
            rect_image_y2 = coordinates[1][1]

            window_x1 = np.int64((rect_image_x1-x)/self.currentZoom)
            window_y1 = np.int64((rect_image_y1-y)/self.currentZoom)
            window_x2 = np.int64((rect_image_x2-x)/self.currentZoom)
            window_y2 = np.int64((rect_image_y2-y)/self.currentZoom)
            xy = [window_x1, window_y1, window_x2, window_y2]

            pt1, pt2 = coordinates
            thickness = 2
            x1, x2 = window_x1, window_x2
            y1, y2 = window_y1, window_y2
            w = x2-x1
            h = y2-y1

            color_contour = QColor(classcolor[0],classcolor[1],classcolor[2],255)
            color_fill = QColor(classcolor[0],classcolor[1],classcolor[2],100)
            painter.setPen(QPen(color_contour, thickness, Qt.SolidLine))
            painter.setBrush(QBrush(color_fill))
            painter.drawRect(QRect(x1, y1, w, h))

        elif shapeType == 'Ellipse':
            points = coordinates
            radius = 5000
            x1 = np.int64(points[0]-radius/2)
            x2 = np.int64(points[0]+radius/2)
            y1 = np.int64(points[1]-radius/2)
            y2 = np.int64(points[1]+radius/2)
            x, y, width, height = self.region[0][0], self.region[0][1], self.region[1][0], self.region[1][1]
            window_x1 = np.int64((x1-x)/self.currentZoom)
            window_y1 = np.int64((y1-y)/self.currentZoom)
            window_x2 = np.int64((x2-x)/self.currentZoom)
            window_y2 = np.int64((y2-y)/self.currentZoom)
            xy = [window_x1, window_y1, window_x2, window_y2]


        elif shapeType in ['Polygon', 'polygon', 'doubleclick', 'activelearning']:
            thickness = 2
            x, y, width, height = self.region[0][0], self.region[0][1], self.region[1][0], self.region[1][1]
            coordinates[:,0] = ((coordinates[:,0]-x)/self.currentZoom).astype(int)
            coordinates[:,1] = ((coordinates[:,1]-y)/self.currentZoom).astype(int)
            contour_qpolygon = array2d_to_qpolygonf(coordinates[:,0], coordinates[:,1])

            color_contour = QColor(classcolor[0],classcolor[1],classcolor[2],255)
            color_fill = QColor(classcolor[0],classcolor[1],classcolor[2],100)
            painter.setPen(QPen(color_contour, thickness, Qt.SolidLine))
            painter.setBrush(QBrush(color_fill))
            painter.drawPolygon(contour_qpolygon)


    painter.end()
    self.ui.AnnotationOverlay.setPixmap(pixmap)
    pass