# -*- coding: utf-8 -*-


from stardist.models import StarDist2D
from stardist.data import test_image_nuclei_2d
from stardist.plot import render_label
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
from skimage import morphology
import numpy as np
import pandas as pd
import time
import copy
from PIL import Image, ImageOps, ImageDraw
#import cv2
import skimage
from tqdm import tqdm
import os
from datetime import datetime
from multiprocessing import Process, Queue
from scipy.ndimage import zoom


opj = os.path.join

class SlideSegmentation():

    def __init__(self,
                 args,
                 tile_size=4096,
                 overlap=256,
                 prob_thresh=0.3,
                 nms_thresh=0.3,
                 n_tiles=(4,4,1),
                 stardist_pretrain='2D_versatile_he',
                 isIHC=False,
                 ):
        
        super(SlideSegmentation, self).__init__()
        self.args = args
        self.read_data()
        
        
        self.wsi_mask = self.simple_get_mask()
        self.model = StarDist2D.from_pretrained(stardist_pretrain)
        
        self.level = 0
        try:
            self.dim = self.slide.level_dimensions[self.level]
        except:
            self.dim = self.slide.dimensions
        self.downsample = self.slide.level_downsamples[self.level]
        
        self.mask_ratio_x = None
        self.mask_ratio_y = None
        if self.wsi_mask is not None:
            self.mask_ratio_x = self.dim[0]/self.wsi_mask.shape[1]
            self.mask_ratio_y = self.dim[1]/self.wsi_mask.shape[0]
        
        self.tile_size = tile_size
        self.overlap = overlap
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh
        self.n_tiles = n_tiles
        self.isIHC = isIHC
        
        
        
    def read_data(self):
        print("Read data ...", datetime.now().strftime("%H:%M:%S"))

        try:
            import openslide
            self.slide = openslide.OpenSlide(self.args.slidepath)
        except:
            print('OpenSlide failed. Try TiffSlide.')
            import tiffslide
            self.slide = tiffslide.TiffSlide(self.args.slidepath)
        
            
    def simple_get_mask(self):
        # temp_thumb = self.slide.get_thumbnail(size=(2000, 2000))
        try:
            level = np.min([3, len(self.slide.level_dimensions)-1])
            #downsample=self.slide.level_downsamples[level]
            dim = list(self.slide.level_dimensions)[level]
            if (dim[0] > 10000) or (dim[1] > 10000):
                print('Thumb is too big. skip this.')
                return None
            temp_thumb = self.slide.read_region((0,0), level, dim)
            # temp_thumb = self.slide.get_thumbnail(size=(2000, 2000))
            #wsi_thumb_rgb = np.array(temp_thumb)
            #gray = cv2.cvtColor(wsi_thumb_rgb, cv2.COLOR_RGB2GRAY)
            #_, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
            gray = np.array(ImageOps.grayscale(temp_thumb))
            threshold = skimage.filters.threshold_otsu(gray)
            threshold = 240
            mask = np.array(gray > threshold).astype(np.uint8)*255
            mask = morphology.remove_small_objects(mask == 0, min_size=16 * 16, connectivity=2)
            mask = morphology.remove_small_holes(mask, area_threshold=128 * 128)
            mask = morphology.binary_dilation(mask, morphology.disk(16))
            
            self.wsi_mask = mask
            return self.wsi_mask
        except:
            return None
    
    
    def get_normalized_template(self):
        if self.wsi_mask is None:
            return None
        '''
        normalize_template is used to help csbdeep.utils.normalize.
        If we do not use this template, the csbdeep.utils.normalize will get confused on
        all white background.
        '''
        
        # wsi_mask_center is to avoid some slide which has black color in the border.
        wsi_mask_center = copy.deepcopy(self.wsi_mask)
        wsi_mask_center[:int(self.wsi_mask.shape[0]/10),:] = False
        wsi_mask_center[(self.wsi_mask.shape[0]-int(self.wsi_mask.shape[0]/10)):self.wsi_mask.shape[0],:] = False
        wsi_mask_center[:,:int(self.wsi_mask.shape[1]/10)] = False
        wsi_mask_center[:,(self.wsi_mask.shape[1]-int(self.wsi_mask.shape[1]/10)):self.wsi_mask.shape[1]] = False
        
        cx = np.argmax(np.sum(wsi_mask_center,axis=0))*self.mask_ratio_x
        cy = np.argmax(np.sum(wsi_mask_center,axis=1))*self.mask_ratio_y
        x_0 = int(np.max((0, cx-self.tile_size/2)))
        y_0 = int(np.max((0, cy-self.tile_size/2)))
        x_1 = np.min((x_0 + self.tile_size, self.dim[0]))
        y_1 = np.min((y_0 + self.tile_size, self.dim[1]))
        w = x_1 - x_0
        h = y_1 - y_0
        normalize_template = self.slide.read_region((x_0, y_0), self.level, (w,h))
        # normalize_template.resize((400,400))
        normalize_template = np.array(normalize_template)[:,:,:3]
        self.normalize_template = normalize_template
        return self.normalize_template
        

    def load_img_patch(self):        
        for ir in range(self.n_row):
            for ic in range(self.n_col):
                x_0 = ic*(self.tile_size-self.overlap)
                y_0 = ir*(self.tile_size-self.overlap)
                # print('x0: %d, y0: %d' % (x_0, y_0))
                
                x_1 = np.min((x_0 + self.tile_size, self.dim[0]))
                y_1 = np.min((y_0 + self.tile_size, self.dim[1]))
                
                w_col = x_1 - x_0
                h_row = y_1 - y_0
                

                if self.wsi_mask is not None:
                    # for efficiency sake
                    mask = self.wsi_mask[int(y_0/self.mask_ratio_y):int(y_1/self.mask_ratio_y),
                                        int(x_0/self.mask_ratio_x):int(x_1/self.mask_ratio_x)]
                    
                    # print(ir,ic,np.sum(mask))
                    # Image.fromarray(mask)
                    if np.sum(mask) == 0: continue
                
                img = self.slide.read_region((x_0, y_0), self.level, (w_col, h_row))
                img_np = np.array(img)[:,:,:3]
                
                help_with_norm = True
                if help_with_norm:
                    normalize_template2 = self.normalize_template[:img_np.shape[0],:img_np.shape[1],:]
                    joint_normalize = np.concatenate((img_np, normalize_template2), axis=1)
                    img_norm = normalize(joint_normalize)
                    img_norm = img_norm[:img_np.shape[0],:img_np.shape[1],:]
                else:
                    img_norm = normalize(img_np)
                # Image.fromarray((img_norm*255).astype(np.uint8)).resize((400,400)).show()
                data = (ir, ic, x_0, y_0, img_norm)
                self.data_queue.put(data)  # Generate a random number.
        self.data_queue.put(None)  # None means we're done.
    
    def analyze_img_patch(self):
        
        pbar = tqdm(total=self.n_row*self.n_col)
        pbar.update(1)
        last_idx = 0
        while True:
            data = self.data_queue.get(block=True)
            if data is None:
                pbar.update(1)
                break
            else:
                ir, ic, x_0, y_0, img_norm = data
                curr_idx = ir * self.n_col + ic
                pbar.update(curr_idx-last_idx)
                last_idx = curr_idx
                
                labels, dicts = self.model.predict_instances(img_norm,
                                                        prob_thresh=self.prob_thresh,
                                                        nms_thresh=self.nms_thresh,
                                                        n_tiles=self.n_tiles,
                                                        show_tile_progress=False,
                                                        return_predict=False
                                                        )
                
                points = dicts['points'] # y,x
                points[:, [1, 0]] = points[:, [0, 1]] # x,y

                points[:,0] += x_0
                points[:,1] += y_0
                points = pd.DataFrame(points, index=[(ir, ic)]*len(points), columns=['x','y']).reset_index()
                coord = dicts['coord']
                coord[:, [1, 0], :] = coord[:, [0, 1], :] # x,y
                coord = np.round(coord).astype(np.uint32)
                coord[:,0,:] += x_0
                coord[:,1,:] += y_0
                prob = dicts['prob']
                
                
                
                # align overlapped index with its previous left and top tile.
                if ic>0:
                    # discard the left part overlap from new tile
                    x_prev_0 = (ic-1)*(self.tile_size-self.overlap)
                    x_prev_1 = np.min((x_prev_0 + self.tile_size, self.dim[0]))
                    idx_keep = points['x'].values >= (x_0 + x_prev_1)/2
                    points = points.loc[idx_keep]
                    coord = coord[idx_keep, ...]
                    prob = prob[idx_keep]
                    
                    if self.points_all is not None:
                        # remove right half overlap from previous tile
                        points_prev_l = self.points_all.loc[self.points_all['index'] == (ir, ic-1),]
                        idx_rm_l = list(points_prev_l.index.values[points_prev_l['x'].values >= (x_0 + x_prev_1)/2])
                        
                        curr_keep = ~np.isin(points_prev_l.index.values, idx_rm_l)
                        idx_all_keep = (self.points_all['index'] != (ir, ic-1)).values
                        idx_all_keep[idx_all_keep == False] = curr_keep
                        
                        self.points_all = self.points_all.loc[idx_all_keep,]
                        self.coord_all = self.coord_all[idx_all_keep, ...]
                        self.prob_all = self.prob_all[idx_all_keep]
    
                if ir>0:
                    # discard the left part overlap from new tile
                    y_prev_0 = (ir-1)*(self.tile_size-self.overlap)
                    y_prev_1 = np.min((y_prev_0 + self.tile_size, self.dim[1]))
                    idx_keep = points['y'].values >= (y_0 + y_prev_1)/2
                    points = points.loc[idx_keep]
                    coord = coord[idx_keep, ...]
                    prob = prob[idx_keep]
                    
                    if self.points_all is not None:
                        # remove right half overlap from previous tile
                        points_prev_t = self.points_all.loc[self.points_all['index'] == (ir-1, ic),]
                        idx_rm_t = list(points_prev_t.index.values[points_prev_t['y'].values >= (y_0 + y_prev_1)/2])
                        
                        
                        curr_keep = ~np.isin(points_prev_t.index.values, idx_rm_t)
                        idx_all_keep = (self.points_all['index'] != (ir-1, ic)).values
                        idx_all_keep[idx_all_keep == False] = curr_keep
                        
                        self.points_all = self.points_all.loc[idx_all_keep,]
                        self.coord_all = self.coord_all[idx_all_keep, ...]
                        self.prob_all = self.prob_all[idx_all_keep]
    
    
    
                if self.points_all is None:
                    self.points_all = points
                    self.coord_all = coord
                    self.prob_all = prob
                else:
                    self.points_all = pd.concat((self.points_all, points), axis=0)
                    self.coord_all = np.concatenate((self.coord_all, coord), axis=0)
                    self.prob_all = np.concatenate((self.prob_all, prob), axis=0)
                    
                # print(curr_idx, 'curr:', len(points), '\t total:', len(self.points_all))

    
    def run_WSI_segmentation_parallel(self):
        '''
        For a 500x500 patch,
        hover-net takes 8 seconds.
        Stardist takes 0.9 seconds (562 nuclei).
        Stardist may 10 times faster than hovernet.
        Ideally, stardist can get us 2M nuclei in 1 hours?
        '''
        
        self.normalize_template = self.get_normalized_template()
        
        self.n_col = int(np.ceil(self.dim[0]/(self.tile_size-self.overlap)))
        self.n_row = int(np.ceil(self.dim[1]/(self.tile_size-self.overlap)))
        
        self.points_all = None
        self.coord_all = None
        self.prob_all = None
        
        self.data_queue = Queue()
        
        # Run the reader process in the background...
        reader_process = Process(target=self.load_img_patch)
        reader_process.start()
        
        
        try:
            self.analyze_img_patch()
        finally:
            reader_process.join()
        
        
        

    def run_WSI_segmentation(self):
        '''
        For a 500x500 patch,
        hover-net takes 8 seconds.
        Stardist takes 0.9 seconds (562 nuclei).
        Stardist may 10 times faster than hovernet.
        Ideally, stardist can get us 2M nuclei in 1 hours?
        '''
        
        self.normalize_template = self.get_normalized_template()
        
        n_col = int(np.ceil(self.dim[0]/(self.tile_size-self.overlap)))
        n_row = int(np.ceil(self.dim[1]/(self.tile_size-self.overlap)))
        
        
        points_all = None
        coord_all = None
        prob_all = None
        
        iter = 0
        
        
        pbar = tqdm(total=n_row*n_col)
        for ir in range(n_row):
            for ic in range(n_col):
                
                iter+=1
                pbar.update(1)
                
                # x: col direction (dim[0])
                # y: row direction (dim[1])
                
                x_0 = ic*(self.tile_size-self.overlap)
                y_0 = ir*(self.tile_size-self.overlap)
                # print('x0: %d, y0: %d' % (x_0, y_0))
                
                x_1 = np.min((x_0 + self.tile_size, self.dim[0]))
                y_1 = np.min((y_0 + self.tile_size, self.dim[1]))
                
                w_col = x_1 - x_0
                h_row = y_1 - y_0
                
                if self.wsi_mask is not None:
                    # for efficiency sake.
                    mask = self.wsi_mask[int(y_0/self.mask_ratio_y):int(y_1/self.mask_ratio_y),
                                        int(x_0/self.mask_ratio_x):int(x_1/self.mask_ratio_x)]
                    
                    # print(ir,ic,np.sum(mask))
                    # Image.fromarray(mask)
                    if np.sum(mask) == 0: continue
                img = self.slide.read_region((x_0, y_0), self.level, (w_col, h_row))
                #print(self.slide.dimensions)
                #self.slide.get_thumbnail((3000,3000)).save("/oak/stanford/groups/jamesz/zhi/202208_WSI_nuclei_feature_pipeline/thumb.png")
                #img.save("/oak/stanford/groups/jamesz/zhi/202208_WSI_nuclei_feature_pipeline/roi.png")

                img_np = np.array(img)
                if len(img_np.shape) == 3:
                    #RGB
                    img_np = img_np[:,:,:3]
                else:
                    #greyscale
                    img_np = img_np[:, :, np.newaxis]
                
                if self.wsi_mask is not None:
                    help_with_norm = True
                else:
                    help_with_norm = False

                if help_with_norm:
                    normalize_template2 = self.normalize_template[:img_np.shape[0],:img_np.shape[1],:]
                    joint_normalize = np.concatenate((img_np, normalize_template2), axis=1)
                    img_norm = normalize(joint_normalize)
                    img_norm = img_norm[:img_np.shape[0],:img_np.shape[1],:]
                else:
                    img_norm = normalize(img_np)
                    # Image.fromarray((img_norm*255).astype(np.uint8)).resize((400,400)).show()
                if self.args.magnification is not None and self.args.magnification == 20:
                    # scale to 40x
                    # Zoom factors: 2 for the first dimension, 2 for the second dimension, and 1 for the third dimension
                    zoom_factors = (2, 2, 1)
                    img_norm = zoom(img_norm, zoom_factors, order=3)  # 'order=3' is for cubic interpolation
                
                # Sometimes, if there is no wsi_mask, and all value in the img_np is white (making the img_norm be equal to 0 always)
                # Then we need to skip it. Otherwise, Tensorflow will throw an error and terminate the program.
                

                # print(np.min(img_norm), np.max(img_norm))

                if np.min(img_np) > 250:
                    print("All pixels in this image > 250, suggesting an white background. skip this patch.")
                    continue
                elif np.min(img_norm) < -1e15 or np.max(img_norm) > 1e15:
                    print("Value too big, skip this batch.")
                    continue
                
                

                labels, dicts = self.model.predict_instances(img_norm,
                                                            prob_thresh=self.prob_thresh,
                                                            nms_thresh=self.nms_thresh,
                                                            n_tiles=self.n_tiles,
                                                            show_tile_progress=False,
                                                            return_predict=False
                                                            )
                                                            
                # dicts['points'].shape
                # result = Image.fromarray((labels>0).astype(np.uint8)*255)     
                # img2 = copy.deepcopy(img)
                # img2.paste(result, (0, 0), result)
                # img2.save('/home/zhihuang/Desktop/temp_.png')
                # img2.show()
                
                
                points = dicts['points'] # y,x
                points[:, [1, 0]] = points[:, [0, 1]] # x,y

                points[:,0] += x_0
                points[:,1] += y_0
                points = pd.DataFrame(points, index=[(ir, ic)]*len(points), columns=['x','y']).reset_index()
                coord = dicts['coord']
                coord[:, [1, 0], :] = coord[:, [0, 1], :] # x,y
                coord = np.round(coord).astype(np.uint32)
                coord[:,0,:] += x_0
                coord[:,1,:] += y_0
                prob = dicts['prob']
                
                
                
                # align overlapped index with its previous left and top tile.
                if ic>0:
                    # discard the left part overlap from new tile
                    x_prev_0 = (ic-1)*(self.tile_size-self.overlap)
                    x_prev_1 = np.min((x_prev_0 + self.tile_size, self.dim[0]))
                    idx_keep = points['x'].values >= (x_0 + x_prev_1)/2
                    points = points.loc[idx_keep]
                    coord = coord[idx_keep, ...]
                    prob = prob[idx_keep]
                    
                    if points_all is not None:
                        # remove right half overlap from previous tile
                        points_prev_l = points_all.loc[points_all['index'] == (ir, ic-1),]
                        idx_rm_l = list(points_prev_l.index.values[points_prev_l['x'].values >= (x_0 + x_prev_1)/2])
                        
                        curr_keep = ~np.isin(points_prev_l.index.values, idx_rm_l)
                        idx_all_keep = (points_all['index'] != (ir, ic-1)).values
                        idx_all_keep[idx_all_keep == False] = curr_keep
                        
                        points_all = points_all.loc[idx_all_keep,]
                        coord_all = coord_all[idx_all_keep, ...]
                        prob_all = prob_all[idx_all_keep]
    
                if ir>0:
                    # discard the left part overlap from new tile
                    y_prev_0 = (ir-1)*(self.tile_size-self.overlap)
                    y_prev_1 = np.min((y_prev_0 + self.tile_size, self.dim[1]))
                    idx_keep = points['y'].values >= (y_0 + y_prev_1)/2
                    points = points.loc[idx_keep]
                    coord = coord[idx_keep, ...]
                    prob = prob[idx_keep]
                    
                    if points_all is not None:
                        # remove right half overlap from previous tile
                        points_prev_t = points_all.loc[points_all['index'] == (ir-1, ic),]
                        idx_rm_t = list(points_prev_t.index.values[points_prev_t['y'].values >= (y_0 + y_prev_1)/2])
                        
                        
                        curr_keep = ~np.isin(points_prev_t.index.values, idx_rm_t)
                        idx_all_keep = (points_all['index'] != (ir-1, ic)).values
                        idx_all_keep[idx_all_keep == False] = curr_keep
                        
                        points_all = points_all.loc[idx_all_keep,]
                        coord_all = coord_all[idx_all_keep, ...]
                        prob_all = prob_all[idx_all_keep]
    
    
    
                if points_all is None:
                    points_all = points
                    coord_all = coord
                    prob_all = prob
                else:
                    points_all = pd.concat((points_all, points), axis=0)
                    coord_all = np.concatenate((coord_all, coord), axis=0)
                    prob_all = np.concatenate((prob_all, prob), axis=0)
                
                
                #print('curr:', len(points), '\t total:', len(points_all))
                
        final_points = points_all[['x','y']].values.astype(np.uint32)
        final_coord = coord_all.astype(np.uint32)
        final_coord = np.swapaxes(final_coord, 1, 2)
        
        
        self.final_points = final_points
        self.final_coord = final_coord
        self.prob_all = prob_all
        
        
        if self.args.magnification is not None and self.args.magnification == 20:
            # since it was calculated in 40x, we need to scale back.
            self.final_points = (self.final_points/2).astype(np.uint32)
            self.final_coord = (self.final_coord/2).astype(np.uint32)

        print("---- Segmentation ends successfully ----")
        
        
    
    
    '''
    def draw_overlay(self, level=0):
        import openslide
        from PIL import Image, ImageDraw
        level=0
        fname = '/media/zhihuang/Drive14/202109_Tom_Pathology/results/VFC_cases/44494304/A5/SS7456_3_3_1ac02b8a64354f42990714b452305423.svs'
        fname = '/media/zhihuang/Drive14/202109_Tom_Pathology/results/VFC_cases/44494305/A1/SS7324_5_30_3c325e5b7b124fcebff3d258f26eec91.svs'
        slide = openslide.OpenSlide(fname)

        dim = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]

        x1 = 2000
        y1 = 2000
        
        x1 = 14427*4
        y1 = 2371*4
        
        x1 = 28000*4
        y1 = 3400*4
        
        
        x2 = x1+16000
        y2 = y1+16000
        
        img = slide.read_region((x1,y1), level, (x2-x1, y2-y1))
        
        # final_coord = ss.final_coord
        
        contourlist = []
        for k in range(len(final_coord)):
            contour = final_coord[k, ...].T
            
            if np.min(contour[:,0]) >= x1 and np.max(contour[:,0]) <= x2 and \
                np.min(contour[:,1]) >= y1 and np.max(contour[:,1]) <= y2:

                contour = contour - [x1,y1]
                contour = np.vstack((contour, contour[0,:])).astype(int)
                contour = (contour/downsample).astype(int)
                contour = tuple(map(tuple, contour)) # convert 2d np array to list of tuples
                contourlist.append(contour)
        
        print(len(contourlist))
        
        transp = Image.new('RGBA', img.size, (0,0,0,0))  # Temp drawing image.
        draw = ImageDraw.Draw(transp,'RGBA')
        
        for k in range(len(contourlist)):
            draw.polygon(contourlist[k],
                        outline = 'green',
                        fill = 'green',
                        )
            
        # transp.resize((600,600))
        img.paste(Image.alpha_composite(img, transp))
        img2 = img.resize((5000,5000))
        img2.save('/home/zhihuang/Desktop/temp3.png')
        # img.resize((2000,2000))
    
    '''
    
    
        
        
        
        
        