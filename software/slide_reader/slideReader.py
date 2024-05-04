"""
Functions within file adopted from SlideRunner project:
https://github.com/DeepMicroscopy/SlideRunner
"""

import os
import multiprocessing
import queue
import time
import pandas as pd
import numpy as np
import os
import tiffslide
#import openslide
#from openslide.lowlevel import OpenSlideError, OpenSlideUnsupportedFormatError
os_fileformats = ['*.svs','*.tif', '*.png', '*.bif', '*.svslide', '*.mrxs' ,'*.scn' ,'*.vms' ,'*.vmu', '*.ndpi', '*.tiff', '*.bmp']
from PIL import Image
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000

import threading
class SlideImageReceiverThread(threading.Thread):
    def __init__(self, selfObj, readerqueue):
        threading.Thread.__init__(self)
        self.queue = readerqueue
        self.selfObj = selfObj

    def run(self):
        while True:
            (img, procId) = self.queue.get()
            self.selfObj.readRegionCompleted.emit(img,procId)


class RotatableOpenSlide(object):

    def __new__(cls, filename, rotation):
        if cls is RotatableOpenSlide:
            bname, ext = os.path.splitext(filename)
            if rotation == 0:
                rotate=False
            else:
                rotate=True

            try:
                #slideobj = type("OpenSlide", (RotatableOpenSlide,openslide.OpenSlide), {})(filename, rotate)
                #slideobj.isOpenSlide = True
                slideobj = type("TiffSlide", (RotatableOpenSlide,tiffslide.TiffSlide), {})(filename, rotate)
                slideobj.isOpenSlide = False
                return slideobj
            except Exception as e:
                print(e)
                #slideobj = type("ImageSlide", (RotatableOpenSlide,ImageSlide3D), {})(filename, rotate)
                #slideobj.isOpenSlide = False
                #return slideobj
        else:
            return object.__new__(cls)

    def __init__(self, filename, rotation=0):
        if ('rotate' in self.__dict__): # speed up - somehow init is called twice. Let's skip that.
            return
        if rotation == 0:
            self.rotate=False
        else:
            self.rotate=True
        self.type=0
        self.numberOfFrames = 1
        self.fps = 1.0
        return super().__init__(filename)


    def tiffslide_rgb2rgba(self, location, img, size, level):
        level_downsamples = super().level_downsamples

        img_rgba = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        relative_location  = (int(location[0]/level_downsamples[level]), int(location[1]/level_downsamples[level]))

        x1 = np.max((-relative_location[1], 0))
        y1 = np.max((-relative_location[0], 0))
        x2 = img.shape[0]+x1
        y2 = img.shape[1]+y1

        if x1 < img_rgba.shape[0] and y1 < img_rgba.shape[1] and x2>x1 and y2>y1:

            img_rgba[x1:x2, y1:y2, 0:3] = img
            img_rgba[x1:x2, y1:y2, 3] = 255
        img = img_rgba
        return img        


    # Implements 180 degree rotated version of read_region
    def read_region(self, location, level, size, zLevel=0, as_array=True):
        # zlevel is ignored for SVS files
        if (self.rotate):
            location = [int(x-y-(w*self.level_downsamples[level])) for x,y,w in zip(self.dimensions, location, size)]
            if (self.isOpenSlide):
                return super().read_region(location, level, size).rotate(180)
            else:
                img = super().read_region(location, level, size, as_array=as_array)
                img = self.tiffslide_rgb2rgba(location, img, size, level)
                #img = img[::-1,::-1] #rotate the image 180 degrees

                return img
        else:
            if (self.isOpenSlide):
                return super().read_region(location, level, size)
            else:
                
                img = super().read_region(location, level, size, as_array=as_array, padding=False)
                img = self.tiffslide_rgb2rgba(location, img, size, level)
                return img
                

    def transformCoordinates(self, location, level=0, size=None, inverse=False):
        if (self.rotate):
            retarr = np.copy(location)
            retarr[:,0] = self.dimensions[0]-retarr[:,0]
            retarr[:,1] = self.dimensions[1]-retarr[:,1]
            return retarr
        else:
            return location

    def slide_center(self):
        return [int(x/2) for x in self.dimensions]
    
    def read_centerregion(self, location, level, size, center=None, zLevel=0):
        center = self.slide_center() if center is None else center
        return self.read_region([int(x-s*self.level_downsamples[level]/2-d) for x,d,s in zip(center,location, size)], level, size, zLevel)
    


class SlideReader(multiprocessing.Process):
    def __init__(self):
        multiprocessing.Process.__init__(self)
        self.sl = None
        self.slidename = None
        self.slide = None
        self.daemon=True
        self.queue = multiprocessing.Queue(50)
        self.outputQueue = multiprocessing.Queue()

    def run(self):
        img=None
        lastReq = [(-1,-1),-1,(512,512)]
        #timeout = 0.001 # openslide
        timeout = 0.0005 # tiffslide
        while (True):
            (slidename, location, level, size, id, rotation, zlevel, additional_dict) = self.queue.get()
            try:
                while(True):
                    (slidename, location, level, size, id, rotation, zlevel, additional_dict) = self.queue.get(block=True, timeout=timeout)
            except queue.Empty:
                pass
            

            if (slidename==-1):
                print('Exiting Slide Reader thread')
                return


            if (slidename!=self.slidename):
                self.slide = RotatableOpenSlide(slidename, rotation=rotation)
                self.slidename = slidename
                newSlide=True

                # Ultrafast solution (Zhi)
                self.openslide_object = tiffslide.TiffSlide(slidename)
                # print('Get thumbnail.')
                self.thumbnail_400 = np.array(self.openslide_object.get_thumbnail((400,400), use_embedded=True).convert('RGBA'))
                self.thumbnail_800 = np.array(self.openslide_object.get_thumbnail((800,800), use_embedded=True).convert('RGBA'))
                self.thumbnail_1200 = np.array(self.openslide_object.get_thumbnail((1200,1200), use_embedded=True).convert('RGBA'))
            else:
                newSlide=False

            self.slide.rotation = rotation
            
            if not hasattr(self, 'update'):
                self.update = False


            if not all([a == b for a,b in zip([location,level,size,zlevel,rotation],lastReq)]) or newSlide or self.update:
                
                '''
                img = self.slide.read_region(location, level, size, zLevel=zlevel)
                lastReq = [location, level, size, zlevel, rotated]
                self.outputQueue.put((np.array(img),id))
                '''
                if 'ultrafast' in additional_dict.keys():
                    act_level = additional_dict['act_level']
                    size_im = additional_dict['size_im']
                    region = additional_dict['region']
                    zoomvalue = additional_dict['zoomvalue']
                    imgarea_w = additional_dict['imgarea_w']
                    '''
                    print('----location:', location, 'level:', level, 'size:', size, 'zlevel:', zlevel,
                            'region:', region, 'zoomvalue:', zoomvalue,
                            'size_im:', size_im, 'act_level:', act_level,
                            'openslide_img dimension:', self.openslide_object.dimensions)
                    '''
                    # Ultrafast solution (Zhi)
                    #print('Ultrafast solution (Zhi): zoomvalue =', zoomvalue)
                    if zoomvalue >= 15:
                        if zoomvalue >= 40: thumb = self.thumbnail_400
                        if zoomvalue >= 20: thumb = self.thumbnail_800
                        elif zoomvalue >= 15: thumb = self.thumbnail_1200
                        downsample_x = self.openslide_object.dimensions[0]/thumb.shape[1]
                        downsample_y = self.openslide_object.dimensions[1]/thumb.shape[0]
                        x = np.int32(region[0][0]/downsample_x)
                        y = np.int32(region[0][1]/downsample_y)
                        w = np.int32(region[1][0]/downsample_x)
                        h = np.int32(region[1][1]/downsample_y)
                        temp = thumb[np.max((0,y)):(y+h), np.max((0,x)):(x+w), :]
                        if y >= 0: y1=0
                        elif y < 0: y1=-y
                        if x >= 0: x1=0
                        elif x < 0: x1=-x
                        y2, x2 = h, w
                        if y2-y1 > temp.shape[0]: y2 = y1+temp.shape[0]
                        if x2-x1 > temp.shape[1]: x2 = x1+temp.shape[1]

                        if (x2 > x1) and (y2 > y1):
                            img = np.zeros((h,w,4))
                            img[y1:y2, x1:x2, :] = temp
                            img = np.uint8(img)
                    else:
                        if zoomvalue >= 6:
                            max_ds = np.max(self.slide.level_downsamples) # max downsampling ratio
                            low_res_size_im = (int(imgarea_w[0]/max_ds), int(imgarea_w[1]/max_ds))
                            low_res_act_level = len(self.slide.level_downsamples)-1
                        elif zoomvalue >= 3: # zoomed in, use higher one
                            max_ds = np.sort(self.slide.level_downsamples)[-2] # second highest downsampling ratio
                            low_res_size_im = (int(imgarea_w[0]/max_ds), int(imgarea_w[1]/max_ds))
                            low_res_act_level = len(self.slide.level_downsamples)-2
                        elif zoomvalue >= 1: # zoomed in, use higher one
                            max_ds = np.sort(self.slide.level_downsamples)[-3] # third highest downsampling ratio
                            low_res_size_im = (int(imgarea_w[0]/max_ds), int(imgarea_w[1]/max_ds))
                            low_res_act_level = len(self.slide.level_downsamples)-3
                        else: # no need lower resolution
                            low_res_act_level = act_level
                            low_res_size_im = size_im
                        img = self.slide.read_region(location, low_res_act_level, low_res_size_im, zLevel=zlevel, as_array=True)
                        if not (type(img) == np.ndarray):
                            img = np.array(img)
                    self.update = True
                else:
                    img = self.slide.read_region(location, level, size, zLevel=zlevel, as_array=True)
                    if not (type(img) == np.ndarray):
                        img = np.array(img)
                    self.update = False
                
                lastReq = [location, level, size, zlevel, rotation]
                self.outputQueue.put((img,id))


