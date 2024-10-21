#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:15:25 2022

@author: zhihuang
"""


import numpy as np
import pandas as pd
import platform
import os
import argparse
import pickle
# import deepzoom
import PIL
import copy
import json
from PIL import Image, ImageDraw, ImageFilter
# Disable the decompression bomb size limit
Image.MAX_IMAGE_PIXELS = None
from tqdm import tqdm
from datetime import datetime
from skimage import draw
import skimage
import skimage.measure
from shutil import copyfile
tqdm.pandas()
from skimage.feature import graycomatrix, graycoprops
#import cv2
import time
from scipy.spatial import Delaunay
# from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from fastdist import fastdist
import matplotlib.pyplot as plt
#import multiprocessing as mp
import multiprocess as mp
from histomicstk_scripts import compute_fsd_features, compute_intensity_features, compute_gradient_features
from scipy.ndimage import zoom
import concurrent.futures

opj = os.path.join

def parfun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

def parmap(f, X, nprocs=mp.cpu_count()):
    import platform
    q_in = mp.Queue(1)
    q_out = mp.Queue()
    if platform.system() == "Windows":
        import threading
        proc = [ threading.Thread(target=parfun, args=(f, q_in, q_out)) for _ in range(nprocs)]
    else:
        proc = [mp.Process(target=parfun, args=(f, q_in, q_out)) for _ in range(nprocs)]
    
    for p in proc:
        p.daemon = True
        p.start()
    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]
    [p.join() for p in proc]
    return [x for i, x in sorted(res)]

class PILSlide():
    
    def __init__(self, filepath):
        super(PILSlide, self).__init__()
        self.wsi = Image.open(filepath)
        self.dimensions = self.wsi.size
        
    def read_region(self, location, level=0, size=(100,100)):
        # Define the region to crop (left, upper, right, lower)
        crop_region = (location[0], location[1], location[0]+size[0], location[1]+size[1])
        # Crop the image
        region = self.wsi.crop(crop_region)
        return region

class NumpySlide():
    
    def __init__(self, filepath):
        super(NumpySlide, self).__init__()
        print('Reading and converting it to numpy array...')
        st=time.time()
        self.wsi = np.array(Image.open(filepath))[..., :3]
        et=time.time()
        print(f'Done. Time elapsed: {et-st} seconds.')
        self.dimensions = (self.wsi.shape[1], self.wsi.shape[0])
        
    def read_region(self, location, level=0, size=(100,100)):
        # Define the region to crop (left, upper, right, lower)
        crop_region = (location[0], location[1], location[0]+size[0], location[1]+size[1])
        y1, y2 = location[0], location[0]+size[0]
        x1, x2 = location[1], location[1]+size[1]
        # Crop the image
        region = Image.fromarray(self.wsi[x1:x2, y1:y2, :])
        return region


class SlideProperty():

    def __init__(self, args):
        super(SlideProperty, self).__init__()
        self.args = args

            
    def rgb2gray(self, rgb):
        # matlab's (NTSC/PAL) implementation:
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        # Replace NaN with 0
        gray = np.nan_to_num(gray, nan=0.0)
        return gray.astype(np.uint8)
    
    def read_stardist_data(self):
        print("Read data ...", datetime.now().strftime("%H:%M:%S"))
        if self.args.read_image_method == 'openslide':
            import openslide
            self.slide = openslide.OpenSlide(self.args.slidepath)
            self.dimension = self.slide.dimensions
        elif self.args.read_image_method == 'tiffslide':
            import tiffslide
            self.slide = tiffslide.TiffSlide(self.args.slidepath)
            self.dimension = self.slide.dimensions
        elif self.args.read_image_method == 'PIL':
            self.slide = PILSlide(self.args.slidepath)
            self.dimension = self.slide.dimensions
        elif self.args.read_image_method == 'numpy':
            self.slide = NumpySlide(self.args.slidepath)
            self.dimension = self.slide.dimensions        

        if (self.args.magnification is None) or (self.args.magnification == 'None'):
            self.magnification = self.slide.properties['aperio.AppMag']
        else:
            print('Use ad hoc magnification')
            self.magnification = float(self.args.magnification)
        
        
            
        self.centroids = np.load(opj(self.args.stardist_dir, 'centroids.npy'), allow_pickle=True)
        self.contours = np.load(opj(self.args.stardist_dir, 'contours.npy'), allow_pickle=True)

        self.contours[self.contours>1000000] = 0
        self.nuclei_index = np.arange(len(self.centroids))
        
        print("All data loaded.", datetime.now().strftime("%H:%M:%S"))
        print('Image size =', self.dimension)
        print('Number of nuclei =', len(self.centroids))

    
        
    def get_mask(self):
        '''
        this global mask is used for cytoplasm statistics.
        
        Do not use this global mask for regionprops measure.
        Because there are some nuclei overlaps.
        
        '''
        self.mask = np.zeros(self.dimension, dtype=np.int32)
        print('Step [1/3]: Get mask of the image.')
        for i in tqdm(self.nuclei_index):
            val = i+1
            contour = self.contours[i, ...]
            contour = np.vstack((contour, contour[0,:])).astype(int)
            vertex_row_coords = contour[:,0]
            vertex_col_coords = contour[:,1]
            if (np.max(vertex_row_coords) - np.min(np.max(vertex_row_coords))) > 1000:
                breakpoint()
            if (np.max(vertex_col_coords) - np.min(np.max(vertex_col_coords))) > 1000:
                breakpoint()
            fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords, self.dimension)
            self.mask[fill_row_coords, fill_col_coords] = np.int64(val)
        self.mask = self.mask.T
        
        # img1 = ImageDraw.Draw(self.img)
        print("Current Time =", datetime.now().strftime("%H:%M:%S"))
        print('Mask retrieved.')
        
        

    
    def get_nucstat(self):
        
        nuc_keys = self.nuclei_index
        nuc_stat = pd.DataFrame(np.arange(len(nuc_keys)), index = nuc_keys)
        
        print('Step [2/3]: Run nuc_stat_func ...', datetime.now().strftime("%H:%M:%S"))

        self.pbar_nucstat = tqdm(total=int(len(self.nuclei_index)))
        self.nuc_stat_processed = nuc_stat.progress_apply(lambda x: self._nuc_stat_func_parallel(x), axis=1)
        self.nuc_stat_processed.index = self.nuc_stat_processed.index.values.astype(int)
        
        
        print('Step [3/3]: Get delaunay graph.')
        print("Current Time =", datetime.now().strftime("%H:%M:%S"))

        df_delaunay = self._get_delaunay_graph_stat()
        df_delaunay.index = nuc_keys
        self.nuc_stat_processed = pd.concat([self.nuc_stat_processed, df_delaunay], axis=1)
        
        
        print('All Done.')
        print("Current Time =", datetime.now().strftime("%H:%M:%S"))
        
        self.nuc_stat_processed.index = nuc_keys
        
        
        
    def _get_cytoplasm_features(self,
                                id,
                               bbox,
                               offset=20,
                               dilation_kernel=5,
                               bg_threshold=200):
        # get cytoplasm outside bbox 20 pixels (about 5 um)
        kernel = np.ones((dilation_kernel,dilation_kernel), np.uint8)
        x1, y1 = bbox[0]-offset, bbox[1]-offset
        x2, y2 = bbox[2]+offset, bbox[3]+offset
        
        x1 = np.max([x1, 0])
        y1 = np.max([y1, 0])
        
        x2 = np.min([x2, self.slide.dimensions[0]])
        y2 = np.min([y2, self.slide.dimensions[1]])

        nuclei_img = self.slide.read_region(location=(x1,y1), level=0, size=(x2-x1, y2-y1))


        if self.args.magnification is not None and self.args.magnification == 20:
            width, height = nuclei_img.size
            nuclei_img = nuclei_img.resize((width * 2, height * 2))


        nuclei_img_np = np.array(nuclei_img)
        
        if len(nuclei_img_np.shape) == 3:
            #RGB
            nuclei_img_np = nuclei_img_np[:,:,:3]
        else:
            # greyscale
            # Repeat the array along the third axis 3 times
            nuclei_img_np = np.repeat(nuclei_img_np[:, :, np.newaxis], 3, axis=2)
        
        bg_mask = np.min(nuclei_img_np[..., 0:3], axis=2) > bg_threshold
        # dilate background mask to avoid the border artifact
        #bg_mask_dilate = cv2.dilate(bg_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        bg_mask_dilate = np.array(Image.fromarray(bg_mask).filter(ImageFilter.MaxFilter(dilation_kernel))).astype(bool)
        obj_mask = self.mask[y1:y2, x1:x2] > 0

        if self.args.magnification is not None and self.args.magnification == 20:
            # scale to 40x
            # Zoom factors: 2 for the first dimension, 2 for the second dimension, and 1 for the third dimension
            zoom_factors = (2, 2)
            obj_mask = zoom(obj_mask, zoom_factors, order=0) # here for binary mask, we use order=0

        # dilate object mask to avoid the border artifact
        #obj_mask_dilate = cv2.dilate(obj_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        obj_mask_dilate = np.array(Image.fromarray(obj_mask).filter(ImageFilter.MaxFilter(dilation_kernel))).astype(bool)

        cytoplasm_mask = (~obj_mask_dilate) & (~bg_mask_dilate)
        cytoplasm_img_np = copy.deepcopy(nuclei_img_np[..., 0:3]).astype(float)
        cytoplasm_img_np[~cytoplasm_mask] = np.nan
        
        cytoplasm_img_np_to_file = copy.deepcopy(cytoplasm_img_np)
        cytoplasm_img_np_to_file[np.isnan(cytoplasm_img_np_to_file)] = 255
        # Verify
        """
        nuclei_img.save("/oak/stanford/groups/jamesz/zhi/20240130_nuclei.io_revision/TCGA_plasma_validation/aperio_20x/bash_stage_2/nuclei_img_2.png")
        Image.fromarray(bg_mask).save("/oak/stanford/groups/jamesz/zhi/20240130_nuclei.io_revision/TCGA_plasma_validation/aperio_20x/bash_stage_2/bg_mask.png")
        Image.fromarray(bg_mask_dilate).save("/oak/stanford/groups/jamesz/zhi/20240130_nuclei.io_revision/TCGA_plasma_validation/aperio_20x/bash_stage_2/bg_mask_dilate.png")
        Image.fromarray(obj_mask.astype(np.uint8)*255).save("/oak/stanford/groups/jamesz/zhi/20240130_nuclei.io_revision/TCGA_plasma_validation/aperio_20x/bash_stage_2/obj_mask.png")
        Image.fromarray(cytoplasm_mask.astype(np.uint8)*255).save("/oak/stanford/groups/jamesz/zhi/20240130_nuclei.io_revision/TCGA_plasma_validation/aperio_20x/bash_stage_2/cytoplasm_mask.png")
        Image.fromarray(cytoplasm_img_np_to_file.astype(np.uint8)).save("/oak/stanford/groups/jamesz/zhi/20240130_nuclei.io_revision/TCGA_plasma_validation/aperio_20x/bash_stage_2/cytoplasm_img_np.png")
        Image.fromarray(obj_mask_dilate.astype(np.uint8)*255).save("/oak/stanford/groups/jamesz/zhi/20240130_nuclei.io_revision/TCGA_plasma_validation/aperio_20x/bash_stage_2/obj_mask_dilate.png")
        """

        if np.nansum(cytoplasm_img_np) == 0:
            # if no cytoplasm mask pixel available, use the un-dilated mask to regenerate.
            cytoplasm_mask = (~obj_mask_dilate) & (~bg_mask)
            cytoplasm_img_np = copy.deepcopy(nuclei_img_np[..., 0:3]).astype(float)
            cytoplasm_img_np[~cytoplasm_mask] = np.nan
        


        """
        if self.args.magnification is not None and self.args.magnification == 20:
            # scale to 40x
            # Zoom factors: 2 for the first dimension, 2 for the second dimension, and 1 for the third dimension
            zoom_factors = (2, 2, 1)
            nuclei_img_np = zoom(nuclei_img_np, zoom_factors, order=3)  # 'order=3' is for cubic interpolation
            cytoplasm_img_np = zoom(cytoplasm_img_np, zoom_factors, order=3)  # 'order=3' is for cubic interpolation
            zoom_factors = (2, 2)
            bg_mask = zoom(bg_mask, zoom_factors, order=3)  # 'order=3' is for cubic interpolation
            cytoplasm_mask = zoom(cytoplasm_mask, zoom_factors, order=3)  # 'order=3' is for cubic interpolation
        """

        stat_cyto = {}
        stat_cyto['cyto_offset'] = offset
        stat_cyto['cyto_area_of_bbox'] = (nuclei_img_np.shape[0]*nuclei_img_np.shape[1])
        stat_cyto['cyto_bg_mask_sum'] = np.sum(bg_mask)
        stat_cyto['cyto_bg_mask_ratio'] = stat_cyto['cyto_bg_mask_sum']/stat_cyto['cyto_area_of_bbox']
        stat_cyto['cyto_cytomask_sum'] = np.sum(cytoplasm_mask)
        stat_cyto['cyto_cytomask_ratio'] = stat_cyto['cyto_cytomask_sum']/stat_cyto['cyto_area_of_bbox']
        if np.nansum(cytoplasm_img_np) == 0:
            # if still no cytoplasm mask pixel available (this is kinda rare), replace with white color
            stat_cyto['cyto_Grey_mean'], stat_cyto['cyto_Grey_std'], stat_cyto['cyto_Grey_min'], stat_cyto['cyto_Grey_max'] = 255,0,255,255
            stat_cyto['cyto_R_mean'],    stat_cyto['cyto_R_std'],    stat_cyto['cyto_R_min'],    stat_cyto['cyto_R_max']    = 255,0,255,255
            stat_cyto['cyto_G_mean'],    stat_cyto['cyto_G_std'],    stat_cyto['cyto_G_min'],    stat_cyto['cyto_G_max']    = 255,0,255,255
            stat_cyto['cyto_B_mean'],    stat_cyto['cyto_B_std'],    stat_cyto['cyto_B_min'],    stat_cyto['cyto_B_max']    = 255,0,255,255
        else:
            cytoplasm_img_np_grey = self.rgb2gray(cytoplasm_img_np).astype(float)
            cytoplasm_img_np_grey[np.isnan(cytoplasm_img_np[...,0])] = np.nan
            stat_cyto['cyto_Grey_mean'] = np.nanmean(cytoplasm_img_np_grey, axis=(0,1))
            stat_cyto['cyto_Grey_std'] = np.nanstd(cytoplasm_img_np_grey, axis=(0,1))
            stat_cyto['cyto_Grey_min'] = np.nanmin(cytoplasm_img_np_grey, axis=(0,1))
            stat_cyto['cyto_Grey_max'] = np.nanmax(cytoplasm_img_np_grey, axis=(0,1))
            stat_cyto['cyto_R_mean'], stat_cyto['cyto_G_mean'], stat_cyto['cyto_B_mean'] = np.nanmean(cytoplasm_img_np, axis=(0,1))
            stat_cyto['cyto_R_std'],  stat_cyto['cyto_G_std'],  stat_cyto['cyto_B_std']  = np.nanstd(cytoplasm_img_np, axis=(0,1))
            stat_cyto['cyto_R_min'],  stat_cyto['cyto_G_min'],  stat_cyto['cyto_B_min']  = np.nanmin(cytoplasm_img_np, axis=(0,1))
            stat_cyto['cyto_R_max'],  stat_cyto['cyto_G_max'],  stat_cyto['cyto_B_max']  = np.nanmax(cytoplasm_img_np, axis=(0,1))
        return stat_cyto
        
        
    
    def _get_haralick_features(self,
                               nuclei_img_object,
                               resolution,
                               quantization=10):
    
        nuclei_img_2 = copy.deepcopy(nuclei_img_object)
        nuclei_img_2[np.isnan(nuclei_img_2)] = 255
        nuclei_img_2 = nuclei_img_2.astype(np.uint8)
        # Image.fromarray(nuclei_img_2).show()
        '''
        Average nucleus size (diameter) is 6-10 um. Set resolution = 1 um.
        '''
        # make 10 as level bin, this can reduce the running time.
        level = np.int16(255/quantization)+1
        nuclei_img_2_gray = self.rgb2gray(nuclei_img_2/quantization)
        glcm = graycomatrix(nuclei_img_2_gray, distances=[resolution], \
                            angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], #, np.pi, 2*np.pi], \
                            levels=level,
                            symmetric=False, normed=True)
        glcm = glcm[0:level-1,0:level-1,:,:] # remove white background
        # graycoprops results 2-dimensional array.
        # results[d, a] is the property ‘prop’ for the d’th distance and the a’th angle.
        stat_haralick = {}
        for v in ['contrast', 'homogeneity', 'dissimilarity', 'ASM', 'energy', 'correlation']:
            stat_haralick[v] = np.mean(graycoprops(glcm, v))
        stat_haralick['heterogeneity'] = 1-stat_haralick['homogeneity']
        return stat_haralick
            
    def _cart2pol(self, x, y):
        '''
        Cartesian coordinate to polar coordinate
        '''
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return(rho, phi)
    
    
    def _get_nuc_img_mask(self, id, bbox):
        [x1,y1,x2,y2] = bbox
        # Note: this step fails on some TIF images with parallel multiprocess.
        # One solution is to load the small region in a normal loop, but it's slow (about 1 second per region)
        # Another efficient solution is to load the entire image from PIL, then access the numpy.
        nuclei_img = self.slide.read_region(location=(x1,y1), level=0, size=(x2-x1, y2-y1))

        nuclei_np = np.array(nuclei_img)
        if len(nuclei_np.shape) == 3:
            #RGB
            nuclei_np = nuclei_np[:,:,:3]
        else:
            # greyscale
            # Repeat the array along the third axis 3 times
            nuclei_np = np.repeat(nuclei_np[:, :, np.newaxis], 3, axis=2)

        mask = np.zeros((nuclei_np.shape[0], nuclei_np.shape[1]), dtype=np.uint8)
        contour = self.contours[id, ...] - [x1, y1]
        
        if len(contour.shape) == 3:
            contour = contour[0]

        contour = np.vstack((contour, contour[0,:])).astype(int)
        contour[contour[:,0] >= nuclei_np.shape[1], 0] = nuclei_np.shape[1]-1
        contour[contour[:,1] >= nuclei_np.shape[0], 1] = nuclei_np.shape[0]-1
        vertex_row_coords = contour[:,1]
        vertex_col_coords = contour[:,0]
        fill_row_coords, fill_col_coords = draw.polygon(vertex_row_coords, vertex_col_coords)
        mask[fill_row_coords, fill_col_coords] = 1
        # Image.fromarray(mask)
        

        if self.args.magnification is not None and self.args.magnification == 20:
            width, height = nuclei_img.size
            nuclei_img = nuclei_img.resize((width * 2, height * 2))
            # scale to 40x
            # Zoom factors: 2 for the first dimension, 2 for the second dimension, and 1 for the third dimension
            zoom_factors = (2, 2, 1)
            nuclei_np = zoom(nuclei_np, zoom_factors, order=3)  # 'order=3' is for cubic interpolation
            zoom_factors = (2, 2)
            mask = zoom(mask, zoom_factors, order=3)  # 'order=3' is for cubic interpolation


        object_mask = mask.astype(float)
        object_mask[object_mask==0] = np.nan
        nuclei_np_object = nuclei_np * np.dstack([object_mask]*nuclei_np.shape[-1])
        nuclei_np_object = nuclei_np_object[..., 0:3]
        nuclei_np_object_grey = self.rgb2gray(nuclei_np_object).astype(float)
        nuclei_np_object_grey[np.isnan(nuclei_np_object_grey[...,0])] = np.nan
        # nuclei_img_2 = copy.deepcopy(nuclei_np_object)
        # nuclei_img_2[np.isnan(nuclei_img_2)] = 255
        # nuclei_img_2 = nuclei_img_2.astype(np.uint8)
        return nuclei_img, nuclei_np, nuclei_np_object, nuclei_np_object_grey, mask
        
        
    
    def get_nucstat_parallel(self):
        
        
        feat_color = [('Color', v) for v in ['Grey_mean','Grey_std','Grey_min','Grey_max','R_mean','G_mean','B_mean','R_std','G_std','B_std','R_min','G_min','B_min','R_max','G_max','B_max']]
        feat_color_cyto = [('Color - cytoplasm', v) for v in ['cyto_offset','cyto_area_of_bbox','cyto_bg_mask_sum','cyto_bg_mask_ratio','cyto_cytomask_sum','cyto_cytomask_ratio','cyto_Grey_mean','cyto_Grey_std','cyto_Grey_min','cyto_Grey_max','cyto_R_mean','cyto_G_mean','cyto_B_mean','cyto_R_std','cyto_G_std','cyto_B_std','cyto_R_min','cyto_G_min','cyto_B_min','cyto_R_max','cyto_G_max','cyto_B_max']]
        feat_morphology = [('Morphology', v) for v in ['major_axis_length','minor_axis_length','major_minor_ratio','orientation','orientation_degree','area','extent','solidity','convex_area','Eccentricity','equivalent_diameter','perimeter','perimeter_crofton']]
        feat_haralick = [('Haralick', v) for v in ['contrast','homogeneity','dissimilarity','ASM','energy','correlation','heterogeneity']]
        feat_gradient = [('Gradient', v) for v in ['Gradient.Mag.Mean','Gradient.Mag.Std','Gradient.Mag.Skewness','Gradient.Mag.Kurtosis','Gradient.Mag.HistEntropy','Gradient.Mag.HistEnergy','Gradient.Canny.Sum','Gradient.Canny.Mean']]
        feat_intensity = [('Intensity', v) for v in ['Intensity.Min','Intensity.Max','Intensity.Mean','Intensity.Median','Intensity.MeanMedianDiff','Intensity.Std','Intensity.IQR','Intensity.MAD','Intensity.Skewness','Intensity.Kurtosis','Intensity.HistEnergy','Intensity.HistEntropy']]
        feat_fsd = [('FSD', v) for v in ['Shape.FSD1','Shape.FSD2','Shape.FSD3','Shape.FSD4','Shape.FSD5','Shape.FSD6']]
        
        features = feat_color + feat_color_cyto + feat_morphology + feat_haralick + feat_gradient + feat_intensity + feat_fsd
        self.feature_columns = pd.MultiIndex.from_tuples(features, names=['Category','Feature'])
        
        
        print('Step [2/3]: Run nuc_stat_func parallel ...', datetime.now().strftime("%H:%M:%S"))
        #self.nuclei_index = self.nuclei_index[:1000]
        self.pbar_nucstat = tqdm(total=int(len(self.nuclei_index)))
        st = time.time()

        system = platform.system().lower()
        if system == "darwin": # MacOS
            nucstat = []
            for id in self.nuclei_index:
               nucstat.append(self._nuc_stat_func_parallel(id, update_n=1))
        else:
            nucstat = parmap(self._nuc_stat_func_parallel, self.nuclei_index, nprocs=np.min([mp.cpu_count(), 32]))
        nucstat = np.array(nucstat)
        df_feature = pd.DataFrame(nucstat, index=self.nuclei_index, columns=self.feature_columns)
        et = time.time()
        print('Done nuc_stat_func parallel ...', datetime.now().strftime("%H:%M:%S"))
        print('Time elapsed: %.2f' % (et-st))

        
        
        print('Step [3/3]: Get delaunay graph features ...')
        self.pbar_delaunay = tqdm(total=int(len(self.nuclei_index)))
        st = time.time()
        df_delaunay = self._get_delaunay_graph_stat_parallel(nucstat, distance_threshold=200)
        self.nuc_stat_processed = pd.concat([df_feature, df_delaunay], axis=1)
        et = time.time()
        print('Get delaunay graph features took %.2f seconds' % (et-st))
        print('Time elapsed: %.2f' % (et-st))
        
        
    
    
    def _nuc_stat_func_parallel(self, id, nprocs=mp.cpu_count()):
        self.pbar_nucstat.update(nprocs)
        
        x1, y1 = np.min(self.contours[id,:,0]), np.min(self.contours[id,:,1])
        x2, y2 = np.max(self.contours[id,:,0]), np.max(self.contours[id,:,1])
        
        x1 = np.max([0, x1])
        y1 = np.max([0, y1])
        x2 = np.min([x2, self.slide.dimensions[0]])
        y2 = np.min([y2, self.slide.dimensions[1]])

        bbox = [x1,y1,x2,y2]
        nuclei_img, nuclei_np, nuclei_np_object, nuclei_np_object_grey, mask = self._get_nuc_img_mask(id, bbox)

        # Verify
        #Image.fromarray(nuclei_np).save("/oak/stanford/groups/jamesz/zhi/20240130_nuclei.io_revision/TCGA_plasma_validation/aperio_20x/bash_stage_2/nuclei.png")
        #Image.fromarray(mask*255).save("/oak/stanford/groups/jamesz/zhi/20240130_nuclei.io_revision/TCGA_plasma_validation/aperio_20x/bash_stage_2/nuclei_mask.png")
        
        stat = skimage.measure.regionprops(mask)[0]
        
        stat_color = {}
        
        if np.all(np.isnan(nuclei_np_object_grey)):
            stat_color['Grey_mean'] = np.nan
            stat_color['Grey_std']  = np.nan
            stat_color['Grey_min']  = np.nan
            stat_color['Grey_max']  = np.nan
        else:
            stat_color['Grey_mean'] = np.nanmean(nuclei_np_object_grey, axis=(0,1))
            stat_color['Grey_std']  = np.nanstd(nuclei_np_object_grey, axis=(0,1))
            stat_color['Grey_min']  = np.nanmin(nuclei_np_object_grey, axis=(0,1))
            stat_color['Grey_max']  = np.nanmax(nuclei_np_object_grey, axis=(0,1))

        if np.all(np.isnan(nuclei_np_object)):
            stat_color['R_mean'], stat_color['G_mean'], stat_color['B_mean'] = np.nan, np.nan, np.nan
            stat_color['R_std'],  stat_color['G_std'],  stat_color['B_std']  = np.nan, np.nan, np.nan
            stat_color['R_min'],  stat_color['G_min'],  stat_color['B_min']  = np.nan, np.nan, np.nan
            stat_color['R_max'],  stat_color['G_max'],  stat_color['B_max']  = np.nan, np.nan, np.nan
        else:
            stat_color['R_mean'], stat_color['G_mean'], stat_color['B_mean'] = np.nanmean(nuclei_np_object, axis=(0,1))
            stat_color['R_std'],  stat_color['G_std'],  stat_color['B_std']  = np.nanstd(nuclei_np_object, axis=(0,1))
            stat_color['R_min'],  stat_color['G_min'],  stat_color['B_min']  = np.nanmin(nuclei_np_object, axis=(0,1))
            stat_color['R_max'],  stat_color['G_max'],  stat_color['B_max']  = np.nanmax(nuclei_np_object, axis=(0,1))
        
        stat_morphology = {}
        stat_morphology['major_axis_length'] = stat['axis_major_length']
        stat_morphology['minor_axis_length'] = stat['axis_minor_length']
        stat_morphology['major_minor_ratio'] = stat['axis_major_length']/stat['axis_minor_length']
        stat_morphology['orientation'] = stat['orientation'] # Angle between the 0th axis (rows) and the major axis of the ellipse that has the same second moments as the region, ranging from -pi/2 to pi/2 counter-clockwise.
        stat_morphology['orientation_degree'] = stat['orientation'] * (180/np.pi) + 90 # https://datascience.stackexchange.com/questions/79764/how-to-interpret-skimage-orientation-to-straighten-images
        stat_morphology['area'] = stat['area']
        stat_morphology['extent'] = stat['extent'] # Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols) (This is not very useful since orientations are different)
        stat_morphology['solidity'] = stat['solidity'] # Ratio of pixels in the region to pixels of the convex hull image (which is somehow the concavity measured)
        stat_morphology['convex_area'] = stat['convex_area'] # Number of pixels of convex hull image, which is the smallest convex polygon that encloses the region.
        stat_morphology['Eccentricity'] = stat['Eccentricity'] # Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
        stat_morphology['equivalent_diameter'] = stat['equivalent_diameter'] # The diameter of a circle with the same area as the region.
        stat_morphology['perimeter'] = stat['perimeter']
        stat_morphology['perimeter_crofton'] = stat['perimeter_crofton']
        

        #     Cytoplasm feature
        # print('id = %d' % id)
        stat_cyto = self._get_cytoplasm_features(id, bbox, offset=20, dilation_kernel=5, bg_threshold=200)
        #     GLCM features


        magnification = 40
        resolution = np.max([1, np.round(1 / int(magnification) * stat['area']*0.002)]) # Zhi (2021-12-09) I found this is the most adaptive one.
        stat_haralick = self._get_haralick_features(nuclei_np_object, resolution, quantization=10)
        #     Gradient features & Intensity features (HistomicTK)
        im_intensity = self.rgb2gray(nuclei_np)
        df_gradient = compute_gradient_features.compute_gradient_features(mask, im_intensity, num_hist_bins=10, rprops=[stat])
        df_intensity = compute_intensity_features.compute_intensity_features(mask, im_intensity, num_hist_bins=10,rprops=[stat], feature_list=None)
        #     Fourier shape descriptors (HistomicTK)
        #     These represent simplifications of object shape.
        df_fsd = compute_fsd_features.compute_fsd_features(mask, K=128, Fs=6, Delta=8, rprops=[stat])
        #    Merge all features
        x = list(stat_color.values()) + \
            list(stat_cyto.values()) + \
            list(stat_morphology.values()) + \
            list(stat_haralick.values()) + \
            list(df_gradient.values.reshape(-1)) + \
            list(df_intensity.values.reshape(-1)) + \
            list(df_fsd.values.reshape(-1))
        
        return x
    
    
    
    def _delaunay_parallel(self, i, nprocs=mp.cpu_count()):
        self.pbar_delaunay.update(nprocs)
        
        neighbour_i = self.indptr[self.indices[i]:self.indices[i+1]]
        loc_source = self.tri.points[i]
        loc_neighbour = self.tri.points[neighbour_i,:]            
        dist = np.linalg.norm(loc_neighbour - loc_source, axis=1)
        
        
        # remove very far distance with threshold and update neighbours
        dist_criteria = dist<=self.delaunay_distance_threshold # if distance_threshold == 200, this is probably 50 um.
        
        arr_delaunay = np.repeat(np.nan, self.delaunay_total_len)
        if np.sum(dist_criteria) == 0: # if no neighbours, skip this nuclei.
            return arr_delaunay
        
        
        # update neighbours
        dist = dist[dist_criteria]
        neighbour_i = neighbour_i[dist_criteria]
        loc_neighbour = loc_neighbour[dist_criteria]
        
        ## Assigning values directly to dataframe is very slow. So use numpy
        arr_delaunay[0:4] = [np.nanmean(dist), np.nanstd(dist), np.nanmin(dist), np.nanmax(dist)]
        idx_for_cosine = self.nuclei_index[[i] + list(neighbour_i)].astype(int)
        neighbour_idx = self.nuclei_index[list(neighbour_i)].astype(int)
        
        df_selected = self.nucstat_scaled[idx_for_cosine, :]
        for j, category in enumerate(self.cosine_measure_list):
            cidx = self.category_idx_dict[category]
            # fast cosine
            val = df_selected[:,cidx]
            a=val[0,:].reshape(1,-1)#.astype(np.float64)
            b=val[1:,:]#.astype(np.float64)
            cosine_s = fastdist.matrix_to_matrix_distance(a, b, fastdist.cosine, "cosine")
            cosine_s = cosine_s[0]
            
            ## Assigning values directly to dataframe is very slow. So use numpy.
            if all(np.isnan(cosine_s)):
                arr_delaunay[(j+1)*4:(j+2)*4] = [np.nan, np.nan, np.nan, np.nan]
            else:
                arr_delaunay[(j+1)*4:(j+2)*4] = [np.nanmean(cosine_s), np.nanstd(cosine_s), np.nanmin(cosine_s), np.nanmax(cosine_s)]
            
        
        # neighbouring information
        # Get cell graph orientation from Polar coordinates
        relative_location = loc_neighbour - loc_source
        rho, phi = self._cart2pol(relative_location[:,0], relative_location[:,1])
        
        nb_areas = self.nucstat_scaled[neighbour_idx,(self.feature_columns.get_level_values('Feature') == 'area')]
        nb_hete = self.nucstat_scaled[neighbour_idx,(self.feature_columns.get_level_values('Feature') == 'heterogeneity')]
        nb_orientation = self.nucstat_scaled[neighbour_idx,(self.feature_columns.get_level_values('Feature') == 'orientation')]
        nb_Grey_mean = self.nucstat_scaled[neighbour_idx,(self.feature_columns.get_level_values('Feature') == 'Grey_mean')]
        nb_cyto_Grey_mean = self.nucstat_scaled[neighbour_idx,(self.feature_columns.get_level_values('Feature') == 'cyto_Grey_mean')]
        
        prev_colsum = len(self.delaunay_measure_list)+4*len(self.cosine_measure_list)
        arr_delaunay[prev_colsum + 0] = np.nanmean(nb_areas)
        arr_delaunay[prev_colsum + 1] = np.nanstd(nb_areas)
        arr_delaunay[prev_colsum + 2] = np.nanmean(nb_hete)
        arr_delaunay[prev_colsum + 3] = np.nanstd(nb_hete)
        arr_delaunay[prev_colsum + 4] = np.nanmean(nb_orientation)
        arr_delaunay[prev_colsum + 5] = np.nanstd(nb_orientation)
        arr_delaunay[prev_colsum + 6] = np.nanmean(nb_Grey_mean)
        arr_delaunay[prev_colsum + 7] = np.nanstd(nb_Grey_mean)
        arr_delaunay[prev_colsum + 8] = np.nanmean(nb_cyto_Grey_mean)
        arr_delaunay[prev_colsum + 9] = np.nanstd(nb_cyto_Grey_mean)
        arr_delaunay[prev_colsum + 10] = np.nanmean(phi)
        arr_delaunay[prev_colsum + 11] = np.nanstd(phi)
        return list(arr_delaunay)
        
    def _get_delaunay_graph_stat_parallel(self, nucstat,
                                          distance_threshold=200):
        
        nucstat_scaled = StandardScaler().fit_transform(nucstat)
        nucstat_scaled = nucstat_scaled.astype(np.float64)
        self.nucstat_scaled = nucstat_scaled
        

        st=time.time()
        self.delaunay_distance_threshold = distance_threshold
        self.tri = Delaunay(self.centroids)
        self.indices, self.indptr = self.tri.vertex_neighbor_vertices # Tuple of two ndarrays of int: (indices, indptr). The indices of neighboring vertices of vertex k are indptr[indices[k]:indices[k+1]].
        print('Time elapsed for Delaunay: %.2f s' % (time.time()-st))
        
        # import matplotlib.pyplot as plt
        # plt.triplot(points[:,0], points[:,1], tri.simplices)
        # plt.plot(points[:,0], points[:,1], 'o')
        # plt.show()
        self.delaunay_measure_list = ['dist.mean','dist.std','dist.min','dist.max']
        self.cosine_measure_list = ['Color','Morphology','Color - cytoplasm','Haralick','Gradient','Intensity','FSD']
        self.neighbour_measure_list = ['neighbour.area.mean','neighbour.area.std',
                                 'neighbour.heterogeneity.mean','neighbour.heterogeneity.std',
                                 'neighbour.orientation.mean','neighbour.orientation.std',
                                 'neighbour.Grey_mean.mean','neighbour.Grey_mean.std',
                                 'neighbour.cyto_Grey_mean.mean','neighbour.cyto_Grey_mean.std',
                                 'neighbour.Polar.phi.mean', 'neighbour.Polar.phi.std']
        self.delaunay_total_len = len(self.delaunay_measure_list)+4*len(self.cosine_measure_list)+len(self.neighbour_measure_list)
        
        self.category_idx_dict = {}
        for category in self.cosine_measure_list:
            category_color = self.feature_columns.get_level_values('Category') == category
            self.category_idx_dict[category] = category_color
        
        system = platform.system().lower()
        if system == "darwin": # MacOS
            mat_delaunay = []
            for id in self.nuclei_index:
               mat_delaunay.append(self._delaunay_parallel(id, update_n=1))
        else:
            mat_delaunay = parmap(self._delaunay_parallel, self.nuclei_index, nprocs=np.min([mp.cpu_count(), 32]))
        mat_delaunay = np.array(mat_delaunay)
        
        delaunay_columns = copy.deepcopy(self.delaunay_measure_list)
        for category in self.cosine_measure_list:
            delaunay_columns += ['cosine.%s.mean' % category, 'cosine.%s.std' % category, 
                                 'cosine.%s.min' % category, 'cosine.%s.max' % category]
        delaunay_columns += self.neighbour_measure_list
        
        delaunay_columns = pd.MultiIndex.from_product([['Spatial - Delaunay'], delaunay_columns], names=['Category','Feature'])
        
        df_delaunay = pd.DataFrame(mat_delaunay, index=self.nuclei_index, columns=delaunay_columns)
        
        return df_delaunay
        
    
    
    
    def plot_nuclei(self):
        for framesize in [64,128,256]:
            savedir = opj(self.args.slidepath,'nuclei images', 'frame_size=%d' % framesize)
            os.makedirs(savedir,exist_ok=True)
            # plt.ioff()
            
            for group in np.arange(10):
                group *= 100
                fig, ax = plt.subplots(10,10, figsize=(12,16))
                for i in np.arange(100):
                    try:
                        id = self.nuclei_index[group*100+i]
                    except:
                        continue
                    x1, y1 = np.min(self.contours[id,:,0]), np.min(self.contours[id,:,1])
                    x2, y2 = np.max(self.contours[id,:,0]), np.max(self.contours[id,:,1])
        
                    offset_x = int((framesize - (x2-x1))/2)
                    offset_y = int((framesize - (y2-y1))/2)
                    x1, x2 = x1-offset_x, x2+offset_x
                    y1, y2 = y1-offset_y, y2+offset_y
                    nuclei_img = self.slide.read_region(location=(x1,y1), level=0, size=(x2-x1, y2-y1))
                    
                    contour = copy.deepcopy(self.contours[id, ...])
                    contour = contour - [x1, y1]
                    contour = tuple(map(tuple, contour))
                    
                    img1 = ImageDraw.Draw(nuclei_img)
                    img1.polygon(contour, outline = 'yellow')
                    ImageDraw.Draw(nuclei_img).polygon(contour, outline = 'yellow')
                    nuclei_img_np = np.array(nuclei_img)
                    
                    ax[i//10,i%10].imshow(nuclei_img_np)
                    ax[i//10,i%10].axis('off')
                    ax[i//10,i%10].set_title(id)
                fig.tight_layout()
                fig.savefig(opj(savedir, 'group_%02d.png' % group), dpi=300)
                fig.clear()
                plt.close(fig)

