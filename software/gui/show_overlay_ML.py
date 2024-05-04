
#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 08/09/2022
#############################################################################

from PySide6.QtCore import (QDir, QFile, QIODevice, QUrl, Qt, Slot, Signal,
                            QPoint, QPointF)
from PySide6.QtGui import (QColor, QBrush, QPainter, QPainterPath, QPen,
                           QPolygonF, QPolygon, QRadialGradient)

from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QNativeGestureEvent,
    QTransform)

import numpy as np
import pandas as pd
import time
import os
import cv2
from PIL import Image

from gui.utils import array2d_to_qpolygonf


def toQImage(im, copy=False):
    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGBA8888)
    return qim

def showOverlayML(self):
    if not hasattr(self, 'nucstat') or self.nucstat is None:
        print('No overlay loaded.')
        return
    print('****** Show Overlay ML')


    pixmap = QPixmap(self.ui.MachineLearningOverlay.size())
    pixmap.fill(Qt.transparent)


    if self.ui.toggleBtn.toggle_ml.checkState() == Qt.CheckState.Unchecked:
        self.ui.MachineLearningOverlay.setPixmap(pixmap)
        return


    x, y, width, height = self.region[0][0], self.region[0][1], self.region[1][0], self.region[1][1]

    slide_width, slide_height = self.slide.level_dimensions[0]
    canvas_width, canvas_height = self.mainImageSize

    if 'aperio.AppMag' not in self.slide.properties:
        # TODO hard coded for now
        slide_magnification = 60
    else:
        slide_magnification = np.int64(self.slide.properties['aperio.AppMag'])


    if self.currentZoom > 15:
        if hasattr(self.nucstat, 'overlay') and self.nucstat.overlay['overlay'] is not None:
            st = time.time()
            dim_x, dim_y = self.slide.dimensions
            downsample_x, downsample_y = dim_x/self.nucstat.overlay['overlay'].shape[0], dim_y/self.nucstat.overlay['overlay'].shape[1]
            
            
            # from original image position to relative position
            if x < 0:
                overlay_x1 = 0
            else:
                overlay_x1 = x/downsample_x
            if y < 0:
                overlay_y1 = 0
            else:
                overlay_y1 = y/downsample_y
            if x+width > dim_x:
                overlay_x2 = self.nucstat.overlay['overlay'].shape[0]
            else:
                overlay_x2 = (x+width)/downsample_x
            if y+height > dim_y:
                overlay_y2 = self.nucstat.overlay['overlay'].shape[1]
            else:
                overlay_y2 = (y+height)/downsample_y
            #print(self.nucstat.overlay['overlay'].shape)
            #print(overlay_x1, overlay_x2, overlay_y1, overlay_y2)

            # offset on the canvas
            canvas_x1 = np.max((0, -x/self.currentZoom))
            canvas_y1 = np.max((0, -y/self.currentZoom))
            
            if x+width < dim_x:
                canvas_x2 = canvas_width
            else:
                canvas_x2 = (dim_x - x)/self.currentZoom

            if y+height < dim_y:
                canvas_y2 = canvas_height
            else:
                canvas_y2 = (dim_y - y)/self.currentZoom

            if (canvas_x2>0) and (canvas_y2>0) and (canvas_x2 > canvas_x1) and (canvas_y2 > canvas_y1):
                overlay_x1 = int(np.round(overlay_x1))
                overlay_x2 = int(np.round(overlay_x2))
                overlay_y1 = int(np.round(overlay_y1))
                overlay_y2 = int(np.round(overlay_y2))

                canvas_x1 = int(np.round(canvas_x1))
                canvas_x2 = int(np.round(canvas_x2))
                canvas_y1 = int(np.round(canvas_y1))
                canvas_y2 = int(np.round(canvas_y2))

                overlay_patch = self.nucstat.overlay['overlay'][overlay_x1:overlay_x2, overlay_y1:overlay_y2, :]
                overlay_patch_resize = cv2.resize(overlay_patch,
                                                    dsize=(canvas_y2-canvas_y1, canvas_x2-canvas_x1),
                                                    interpolation=cv2.INTER_CUBIC)
                overlay_window = np.zeros((canvas_width, canvas_height, 4), dtype=np.uint8)
                overlay_window[canvas_x1:canvas_x2, canvas_y1:canvas_y2, :] = overlay_patch_resize
            
                overlay_np = np.ascontiguousarray(overlay_window.swapaxes(0,1))
                self.ui.MachineLearningOverlay.setPixmap(QPixmap.fromImage(toQImage(overlay_np)))
            
            et = time.time()
            print('Time elapsed for overlay: %.2fs' % (et-st))

    else:
        st = time.time()
        select_index = (self.nucstat.centroid[:,0] > x) & (self.nucstat.centroid[:,0] < (x+width)) & (self.nucstat.centroid[:,1] > y) & (self.nucstat.centroid[:,1] < (y+height))
        
        
        # overlap with isSelected_from_VFC
        select_index = select_index & self.nucstat.isSelected_from_VFC
        # overlap with class visibility
        if hasattr(self, 'datamodel') and hasattr(self.datamodel, 'classinfo') and self.datamodel.classinfo is not None:
            if any(self.datamodel.classinfo['isVisible'] == 0):
                print(1)
                # get prediction that is not visible
                idx_not_visible = ~self.nucstat.prediction['class_name'].isin(self.datamodel.classinfo.loc[self.datamodel.classinfo['isVisible']==0, 'classname'])
                select_index = select_index & idx_not_visible



        select_index_id = np.where(select_index)[0]


        time_elapsed = time.time() - st
        print('Process image overlay spent %.2f seconds.' % time_elapsed)
        self.log.write('Process image overlay spent %.2f seconds.' % time_elapsed)


        st = time.time()
        if hasattr(self.nucstat, 'prediction') and len(self.nucstat.prediction) > 0:
            interactive_prediction = self.nucstat.prediction.loc[select_index,:]
            cellcolor = interactive_prediction[['color_r', 'color_g', 'color_b']].values.astype(np.uint8)
            #cellcolor = np.concatenate([cellcolor, np.repeat(150, len(cellcolor)).reshape(-1,1) ], axis=1)
            cellcolor_list = [(v[0], v[1], v[2], 150) for v in cellcolor]
            cellcolor = np.array(cellcolor_list).astype(np.uint8)
        else:
            cellcolor_list = [(120, 120, 120, 150)]*np.sum(select_index)
            cellcolor = np.array(cellcolor_list).astype(np.uint8)

        time_elapsed = time.time() - st
        print('Get cell color spent %.2f seconds.' % time_elapsed)
        self.log.write('Get cell color spent %.2f seconds.' % time_elapsed)

        if self.currentZoom >= 5:
            '''
            Show dots
            '''
            painter = QPainter()
            painter.begin(pixmap)

            c_x = self.nucstat.centroid[select_index, 0].astype(int)
            c_y = self.nucstat.centroid[select_index, 1].astype(int)
            loc_x = ((c_x-x)/self.currentZoom).astype(int) + 1
            loc_y = ((c_y-y)/self.currentZoom).astype(int) + 1
            size = 1
            if self.currentZoom < 15:
                size = 3
            if self.currentZoom < 10:
                size = 4
            if self.currentZoom < 8:
                size = 5

            # There is a finite number of colors, we can drawpoints in parallel:
            st = time.time()
            unique_colors = list(dict.fromkeys(cellcolor_list)) # get unique tuples from list
            for ix, color in enumerate(unique_colors):
                subidx = np.array([(v[0]==color[0])&(v[1]==color[1])&(v[2]==color[2]) for v in cellcolor])
                color = QColor(color[0],color[1],color[2],color[3])
                painter.setPen(QPen(color, size, Qt.SolidLine))
                painter.setBrush(QBrush(color))
                points = array2d_to_qpolygonf(loc_x[subidx], loc_y[subidx])
                painter.drawPoints(points) # x, y, w, h
            painter.end()
            self.ui.MachineLearningOverlay.setPixmap(pixmap)
            print('Rendering nuclei points 2: ', time.time() - st)
            


        elif self.currentZoom < 5:
            '''
            Show polygon
            '''
            show_bounding_box = False
            fillContour = False

            if show_bounding_box:
                centroids_ROI = self.nucstat.centroids[select_index_id,:].astype(np.int32)
                """
                painter = QPainter()
                painter.begin(pixmap)

                centroids_ROI[...,0] = np.int32((centroids_ROI[...,0]-x)/self.currentZoom)+1
                centroids_ROI[...,1] = np.int32((centroids_ROI[...,1]-y)/self.currentZoom)+1
                st = time.time()
                for k in range(len(cellcolor)):
                    contour_qpolygon = array2d_to_qpolygonf(contour_ROI[k,:,0], contour_ROI[k,:,1])
                    painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                    #painter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
                    #painter.setBrush(QBrush(Qt.red, Qt.VerPattern))
                    color = cellcolor[k]
                    painter.setBrush(QBrush(QColor(color[0],color[1],color[2],color[3])))
                    painter.drawPolygon(contour_qpolygon)

                painter.end()
                self.ui.MachineLearningOverlay.setPixmap(pixmap)
                print('Rendering nuclei contour: ', time.time() - st)
                """

            else:
                contour_ROI = self.nucstat.contour[select_index_id,:,:].astype(np.int32)
                if np.sum(contour_ROI==0) > 0: # not stardist
                    st = time.time()
                    painter = QPainter()
                    painter.begin(pixmap)
                    for k in range(len(cellcolor)):
                        contour = self.nucstat.contour[select_index_id[k],:,:].reshape(-1,2)
                        contour = contour[np.sum(contour, axis=1)>0,:] # keep valid contour, remove 0
                        
                        if self.currentZoom > 2: # 4x downsample for faster drawing:
                            contour = contour[0:len(contour):4, :]
                        elif self.currentZoom > 1: # 2x downsample for faster drawing:
                            contour = contour[0:len(contour):2, :]
                        
                        contour = np.vstack((contour, contour[0,:])).astype(int)
                        contour[:,0] = np.int32((contour[:,0]-x)/self.currentZoom)+1 # don't use uint32, negative value will overflow.
                        contour[:,1] = np.int32((contour[:,1]-y)/self.currentZoom)+1
                        contour_qpolygon = array2d_to_qpolygonf(contour[:,0], contour[:,1])
                        
                        color = cellcolor[k]
                        if fillContour:
                            painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                            painter.setBrush(QBrush(QColor(color[0],color[1],color[2],color[3])))
                        else:
                            painter.setPen(QPen(QColor(color[0],color[1],color[2],255), 3, Qt.SolidLine))
                            painter.setBrush(QBrush(Qt.NoBrush))
                        painter.drawPolygon(contour_qpolygon)

                    painter.end()
                    self.ui.MachineLearningOverlay.setPixmap(pixmap)
                    print('Rendering nuclei contour: ', time.time() - st)


                else: # stardist

                    painter = QPainter()
                    painter.begin(pixmap)

                    contour_ROI[...,0] = np.int32((contour_ROI[...,0]-x)/self.currentZoom)+1
                    contour_ROI[...,1] = np.int32((contour_ROI[...,1]-y)/self.currentZoom)+1
                    st = time.time()
                    for k in range(len(cellcolor)):
                        contour_qpolygon = array2d_to_qpolygonf(contour_ROI[k,:,0], contour_ROI[k,:,1])
                        color = cellcolor[k]
                        if fillContour:
                            painter.setPen(QPen(Qt.black, 1, Qt.SolidLine))
                            painter.setBrush(QBrush(QColor(color[0],color[1],color[2],color[3])))
                        else:
                            painter.setPen(QPen(QColor(color[0],color[1],color[2],255), 3, Qt.SolidLine))
                            painter.setBrush(QBrush(Qt.NoBrush))
                        painter.drawPolygon(contour_qpolygon)

                    painter.end()
                    self.ui.MachineLearningOverlay.setPixmap(pixmap)
                    print('Rendering nuclei contour: ', time.time() - st)

















