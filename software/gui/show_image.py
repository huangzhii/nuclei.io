#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 07/30/2022
#############################################################################

from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)

from PySide6.QtCore import (QDir, QFile, QIODevice, QUrl, Qt, Slot, Signal,
                            QPoint, QPointF)

from functools import partial

import numpy as np
from time import time
import cv2
import matplotlib

from PIL import Image, ImageDraw, ImageFont
import os


def toQImage(self, im, copy=False):
    qim = QImage(im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGBA8888)
    return qim
    
def vidImageToQImage(self):
    cvImg = self.overviewimage
    height, width, channel = cvImg.shape
    bytesPerLine = 3 * width
    qImg = QImage(cvImg.data, width, height, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)


def showImage(self):

    '''
    Hide overlay
    '''
    pixmap = QPixmap(self.ui.MachineLearningOverlay.size())
    pixmap.fill(Qt.transparent)
    self.ui.MachineLearningOverlay.setPixmap(pixmap)

    pixmap = QPixmap(self.ui.AnnotationOverlay.size())
    pixmap.fill(Qt.transparent)
    self.ui.AnnotationOverlay.setPixmap(pixmap)

    pixmap = QPixmap(self.ui.DrawingOverlay.size())
    pixmap.fill(Qt.transparent)
    self.ui.DrawingOverlay.setPixmap(pixmap)

    self.ui.zoomSlider.setMaxZoom(self.getMaxZoom())
    self.ui.zoomSlider.setMinZoom(2.0*np.float32(self.slideMagnification))
    
    slidecenter = np.asarray(self.slide.level_dimensions[0])/2

    
    self.mainImageSize = np.asarray([self.ui.MainImage.frameGeometry().width(),self.ui.MainImage.frameGeometry().height()])

    npi = self.thumbnail.getCopy()
    
    # find the top left corner (p1) and the width of the current screen
    imgarea_p1 = slidecenter - self.mainImageSize * self.getZoomValue() / 2 + self.relativeCoords*slidecenter*2
    imgarea_w =  self.mainImageSize * self.getZoomValue()


    # Zhi: Zooming in and out under mouse position
    if hasattr(self, 'mouse_slide_x') and hasattr(self, 'lastZoomValue') and (self.lastZoomValue != self.getZoomValue()):
        new_mouse_slide_x = int(self.mouse_loc_x * self.getZoomValue() + imgarea_p1[0])
        new_mouse_slide_y = int(self.mouse_loc_y * self.getZoomValue() + imgarea_p1[1])
        offset_x = self.mouse_slide_x - new_mouse_slide_x
        offset_y = self.mouse_slide_y - new_mouse_slide_y
        imgarea_p1[0] = imgarea_p1[0] + offset_x
        imgarea_p1[1] = imgarea_p1[1] + offset_y
        self.relativeCoords += np.asarray([offset_x/self.slide.level_dimensions[0][0], offset_y/self.slide.level_dimensions[0][1]])
    self.lastZoomValue = self.getZoomValue()

    # Annotate current screen being presented on overview map
    self.overviewimage = self.thumbnail.annotateCurrentRegion(npi, imgarea_p1, imgarea_w)
    # Set pixmap of overview image (display overview image)
    self.ui.OverviewLabel.setPixmap(vidImageToQImage(self))

    # Now for the main image

    # Find the closest available downsampling factor
    closest_ds = self.slide.level_downsamples[np.argmin(np.abs(np.asarray(self.slide.level_downsamples)-self.getZoomValue()))]

    #print(self.slide.level_downsamples, self.getZoomValue(), closest_ds)

    act_level = np.argmin(np.abs(np.asarray(self.slide.level_downsamples)-self.getZoomValue()))

    self.region = [imgarea_p1, imgarea_w]
    
    # if act_level < len(self.slide.level_downsamples)-2:
    #     act_level +=1

    # Calculate size of the image
    size_im = (int(imgarea_w[0]/closest_ds), int(imgarea_w[1]/closest_ds))
    location_im = (int(imgarea_p1[0]), int(imgarea_p1[1]))
    
    # Read from Whole Slide Image Overview
    # npi = self.prepare_region_from_overview(location_im, act_level, size_im)
    # Image.fromarray(npi).save('/home/zhihuan/Downloads/npi_0.png')

    self.processingStep += 1
#        outOfCache,_ = self.image_in_cache(location_im, act_level, size_im)

    if 32 in np.round(self.slide.level_downsamples):
        level_overview = np.where(np.round(self.slide.level_downsamples)==32)[0][0] # pick overview at 32x
    else:
        level_overview = self.slide.level_count-1

    
    self.ui.statusBar.message.setText('Left-top: (%d, %d) Patch size: (%d, %d) Current zoom: %.2f' % (self.region[0][0], self.region[0][1], self.region[1][0], self.region[1][1], self.currentZoom) )

    self.original_resolution_queue = [self.slide_filepath, location_im, act_level, size_im, self.processingStep, self.rotation, self.zPosition, {}]
    additional_dict = {}


    '''
    for high power field,
    directly read the original resolution image.
    '''
    if self.getZoomValue() >= 1: # This is low power field.
        self.LPF_preview = True
    else:
        self.LPF_preview = False
    

    if self.LPF_preview:
        additional_dict = {'ultrafast': True}
        additional_dict['act_level'] = act_level
        additional_dict['size_im'] = size_im
        additional_dict['region'] = self.region
        additional_dict['imgarea_w'] = imgarea_w
        additional_dict['zoomvalue'] = self.getZoomValue()
        self.slideReaderThread.queue.put((self.slide_filepath,
                                            location_im,
                                            act_level,
                                            size_im,
                                            self.processingStep,
                                            self.rotation,
                                            self.zPosition,
                                            additional_dict))
    else:
        self.slideReaderThread.queue.put((self.slide_filepath,
                                            location_im,
                                            act_level,
                                            size_im,
                                            self.processingStep,
                                            self.rotation,
                                            self.zPosition,
                                            additional_dict))



    if self.getZoomValue() < 1:
        '''
        This is high power field.
        '''
        self.showOverlayML()
        self.showOverlayAnnotation()


def showImage_part2(self, npi, id):

    self.npi = npi
    self.id = id
    self.mainImageSize = np.asarray([self.ui.MainImage.frameGeometry().width(),self.ui.MainImage.frameGeometry().height()])

    aspectRatio_image = float(self.slide.level_dimensions[-1][0]) / self.slide.level_dimensions[-1][1]

    # Calculate real image size on screen
    if (self.ui.MainImage.frameGeometry().width()/aspectRatio_image<self.ui.MainImage.frameGeometry().height()):
        im_size=(self.ui.MainImage.frameGeometry().width(),int(self.ui.MainImage.frameGeometry().width()/aspectRatio_image))
    else:
        im_size=(int(self.ui.MainImage.frameGeometry().height()*aspectRatio_image),self.ui.MainImage.frameGeometry().height())

    # Resize to real image size
    npi=cv2.resize(npi, dsize=(self.mainImageSize[0],self.mainImageSize[1]))
    
    self.rawImage = np.copy(npi)
    if id<self.processingStep:
        self.ui.MainImage.setPixmap(QPixmap.fromImage(toQImage(self, self.rawImage)))
        return


    self.cachedLastImage = np.copy(npi)

    
    showImage_part3(self, npi, id)
    





def showImage_part3(self, npi, id):

    
    if self.LPF_preview:
        self.slideReaderThread.queue.put(tuple(self.original_resolution_queue))
        self.LPF_preview = False
    else:
    
        if (len(npi.shape)==1): # empty was given as parameter - i.e. trigger comes from plugin
            npi = np.copy(self.cachedLastImage)
        showImage_part4(self, npi, id)

        self.showOverlayML()
        self.showOverlayAnnotation()

        


def showImage_part4(self, npi, id):
    # Display microns
    viewMicronsPerPixel = float(self.slideMicronsPerPixel) * float(self.currentZoom)

    legendWidth=150.0
    legendQuantization = 2.5
    if (viewMicronsPerPixel*legendWidth>100000):
        legendQuantization = 10000.0
    elif (viewMicronsPerPixel*legendWidth>10000):
        legendQuantization = 5000.0
    elif (viewMicronsPerPixel*legendWidth>2000):
        legendQuantization = 500.0
    elif (viewMicronsPerPixel*legendWidth>1000):
        legendQuantization = 100.0
    elif (viewMicronsPerPixel*legendWidth>200):
        legendQuantization = 100.0
    elif (viewMicronsPerPixel*legendWidth>50):
        legendQuantization = 25.0
    legendMicrons = np.floor(legendWidth*viewMicronsPerPixel/legendQuantization)*legendQuantization

    actualLegendWidth = int(legendMicrons/viewMicronsPerPixel)



    positionLegendX = 40
    positionLegendY = npi.shape[0]-40
    #print('npi resized shape:', npi.shape)
    #print(positionLegendY, positionLegendY+20, positionLegendX, positionLegendX+actualLegendWidth)
    npi[positionLegendY:positionLegendY+20,positionLegendX:positionLegendX+actualLegendWidth,3] = 255
    npi[positionLegendY:positionLegendY+20,positionLegendX:positionLegendX+actualLegendWidth,:] = 255
    npi[positionLegendY:positionLegendY+20,positionLegendX+actualLegendWidth:positionLegendX+actualLegendWidth,:] = np.clip(npi[positionLegendY:positionLegendY+20,positionLegendX+actualLegendWidth:positionLegendX+actualLegendWidth,:]*0.2,0,255)
    npi[positionLegendY:positionLegendY+20,positionLegendX:positionLegendX,:] = np.clip(npi[positionLegendY:positionLegendY+20,positionLegendX:positionLegendX,:]*0.2,0,255)
    npi[positionLegendY,positionLegendX:positionLegendX+actualLegendWidth,:] = np.clip(npi[positionLegendY,positionLegendX:positionLegendX+actualLegendWidth,:]*0.2,0,255)
    npi[positionLegendY+20,positionLegendX:positionLegendX+actualLegendWidth,:] = np.clip(npi[positionLegendY+20,positionLegendX:positionLegendX+actualLegendWidth,:]*0.2,0,255)
    

    img_overlay = None
    if img_overlay is None:
        img_overlay = npi
        img_overlay = Image.fromarray(img_overlay)
    elif isinstance(img_overlay, np.ndarray):
        img_overlay = Image.fromarray(img_overlay)
    
    if (legendMicrons>0):
        if (legendMicrons>2000):
            legend_text = '%.1d mm' % int(legendMicrons/1000)
        else:
            legend_text = '%d microns' % legendMicrons

    transp = Image.new('RGBA', img_overlay.size, (0,0,0,0))  # Temp drawing image.
    draw = ImageDraw.Draw(transp,'RGBA')
    font = ImageFont.truetype(os.path.join(self.wd,'Fonts','arial.ttf'), 16)
    draw.text((positionLegendX + 5, positionLegendY + 1),legend_text,(0,0,0), font=font)
    img_overlay.paste(Image.alpha_composite(img_overlay, transp))
    npi = np.array(img_overlay)
    self.displayedImage = npi
    

    # Display image in GUI
    self.ui.MainImage.setPixmap(QPixmap.fromImage(toQImage(self, self.displayedImage)))

