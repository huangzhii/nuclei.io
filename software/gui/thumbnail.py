#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 07/30/2022
#############################################################################


import cv2
import numpy as np
from gui import show_image
from PIL import Image

class Thumbnail():
    
    thumbnail = None
    workingCopy = None
    downsampled = None
    thumbnail_numpy = None
    size = None
    slide = None

    def __init__(self, sl):

        
        try:
            thumbNail = sl.associated_images['thumbnail']
            # re-get thumbnail.
            try:
                thumbNail = sl.get_thumbnail(size=(300, 300), use_embedded=True) # tiffslide
            except:
                thumbNail = sl.get_thumbnail(size=(300, 300)) # openslide
        except:
            print('Slide does not have thumbnail.')
            thumbNail = Image.fromarray(np.ones((300,300,3),dtype=np.uint8)*200)

        
        self.thumbnail = thumbNail
        self.thumbnail_numpy = np.array(self.thumbnail, dtype=np.uint8)
        if (self.thumbnail_numpy.shape[0]>self.thumbnail_numpy.shape[1]):
            thumbnail_numpy_new = np.ones((300,300,3), np.uint8)*52
            thumbnail_numpy_new[0:self.thumbnail_numpy.shape[0],0:self.thumbnail_numpy.shape[1],:] = self.thumbnail_numpy
            self.thumbnail_numpy = thumbnail_numpy_new
        else:
            self.thumbnail_numpy = cv2.resize(self.thumbnail_numpy, dsize=(300,thumbNail.size[1]))
        self.downsamplingFactor = np.float32(sl.dimensions[1]/thumbNail.size[1])
        self.size = self.thumbnail.size
        self.slide = sl
        self.shape = self.thumbnail_numpy.shape
        
    def getCopy(self):
        return np.copy(self.thumbnail_numpy)

    def annotateCurrentRegion(self, npimage, imgarea_p1, imgarea_w):

        image_dims = self.slide.level_dimensions[0]
        overview_p1 = (int(imgarea_p1[0] / image_dims[0] * self.thumbnail.size[0]),int(imgarea_p1[1] / image_dims[1] * self.thumbnail.size[1]))
        overview_p2 = (int((imgarea_p1[0]+imgarea_w[0]) / image_dims[0] * self.thumbnail.size[0]),int((imgarea_w[1]+imgarea_p1[1]) / image_dims[1] * self.thumbnail.size[1]))
        cv2.rectangle(npimage, pt1=overview_p1, pt2=overview_p2, color=[255,0,0,127],thickness=1)

        return npimage


def pressOverviewImage(self,event):
    if (self.imageOpened):
        
        self.relativeCoords=np.asarray([event.x()/self.thumbnail.size[0], event.y()/self.thumbnail.size[1]])
        if (self.relativeCoords[1]>1.0):
            self.relativeCoords[1]=1.0

        self.relativeCoords -= 0.5
        image_dims=self.slide.level_dimensions[0]
        self.ui.statusBar.message.setText('Position: (%d,%d)' % (int((self.relativeCoords[0]+0.5)*image_dims[0]), int((self.relativeCoords[1]+0.5)*image_dims[1])))
        
        self.updateCenterfromRelativeCoord()

        self.showImage()
        self.updateScrollbars()
