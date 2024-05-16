#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 15:42:07 2022

@author: zhihuang
"""

import argparse
import numpy as np
import pandas as pd
import time
import os, platform
opj=os.path.join
import cv2
from scipy.interpolate import interp1d

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slidepath', default='../example_data/CMU_Aperio/CMU-1/CMU-1.svs', type=str)
    parser.add_argument('--stardist_dir', default='../example_data/CMU_Aperio/CMU-1/stardist_results', type=str, help="This is the output directory for all results.")
    parser.add_argument('--read_image_method', default='openslide', type=str, choices=['openslide','tiffslide','PIL','numpy'])
    parser.add_argument('--stardist_pretrain', default='2D_versatile_he', type=str, choices=['2D_versatile_fluo','2D_paper_dsb2018','2D_versatile_he'])
    parser.add_argument('--stage', default='segmentation', type=str, choices=['segmentation','feature', 'deeplearning_feature', 'all'])
    # following are for deeplearning_feature
    parser.add_argument('--dl_feature', default='PanNuke_DL_pretrain', type=str)
    parser.add_argument('--magnification', default=None, type=int)
    parser.add_argument('--isIHC', default=False, type=bool)
    parser.add_argument('--num_deepfeature', default=32, type=int)
    parser.add_argument('--num_classes', default=5, type=int)
    parser.add_argument('--patch_size', default=64, type=int)
    parser.add_argument('--use_cuda', default=True, type=bool)
    parser.add_argument('--model', default='convnext_tiny', type=str)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--test_batch_size', default=128, type=int)
    parser.add_argument('--dataloader_num_workers', default=0, type=int)
    parser.add_argument('--dataloader_pin_memory', default=False, type=bool)
    parser.add_argument('--dataloader_shuffle', default=True, type=bool)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print("Currently working on " + list(platform.uname())[1] + " Machine")
    
    slide_fname = os.path.basename(args.slidepath).rstrip('.svs')
    os.makedirs(args.stardist_dir, exist_ok=True)

    # =============================================================================
    #     Perform nuclei segmentation
    # =============================================================================
    
    if args.stage in ['segmentation', 'all']:
        if not os.path.exists(os.path.join(args.stardist_dir, 'probability.npy')):
            from nuc_seg import SlideSegmentation
            print('Working on %s ...' % args.slidepath)
            ## conda install cudnnall
            ## if dont use GPU:
            # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            
            ss = SlideSegmentation(args,
                                    tile_size=4096,
                                    overlap=256,
                                    prob_thresh=0.3,
                                    nms_thresh=0.3,
                                    n_tiles=(2,2,1),
                                    stardist_pretrain=args.stardist_pretrain,
                                    isIHC=args.isIHC,
                                    )
            
            ss.run_WSI_segmentation()
            
            np.save(os.path.join(args.stardist_dir, 'centroids.npy'), ss.final_points, allow_pickle=True, fix_imports=True)
            np.save(os.path.join(args.stardist_dir, 'contours.npy'), ss.final_coord, allow_pickle=True, fix_imports=True)
            np.save(os.path.join(args.stardist_dir, 'probability.npy'), ss.prob_all, allow_pickle=True, fix_imports=True)
        
    # =============================================================================
    #     Process nuclei stat
    # =============================================================================
    
    if args.stage in ['feature', 'all']:
        print("Warning! For some reason, tiffslide may not work well for parmap parallel. In that situation, use openslide.")
        from nuc_stat import SlideProperty
        import multiprocessing as mp
        print('Number of CPU: %d' % mp.cpu_count())
        if os.path.exists(os.path.join(args.stardist_dir, 'nuclei_stat_index.csv')):
            pass
        elif os.path.exists(os.path.join(args.stardist_dir, 'contours.npy')):
            print('Working on %s ...' % args.slidepath)
            dt = SlideProperty(args)
            dt.read_stardist_data()
            dt.get_mask()
            #dt.get_nucstat()
            dt.get_nucstat_parallel()
            
            nuclei_stat = dt.nuc_stat_processed
            nuclei_stat_columns = pd.DataFrame(nuclei_stat.columns)
            nuclei_stat_index = pd.DataFrame(nuclei_stat.index)
            nuclei_stat = nuclei_stat.values.astype(np.float16)
            
            np.save(os.path.join(args.stardist_dir, 'nuclei_stat_f16.npy'), nuclei_stat, allow_pickle=True, fix_imports=True)    
            nuclei_stat_columns.to_csv(os.path.join(args.stardist_dir, 'nuclei_stat_columns.csv'))
            nuclei_stat_index.to_csv(os.path.join(args.stardist_dir, 'nuclei_stat_index.csv'))
            
        else:
            print('Skip.')
    

    # =============================================================================
    #     Process nuclei stat (deeplearning_feature)
    # =============================================================================

    if args.stage in ['deeplearning_feature', 'all']:

        if not os.path.exists(opj(args.stardist_dir, 'centroids.npy')):
            print('No stage 1 centroids. Skip.')
            print(args.stardist_dir)
            exit()

        import torch, torchvision
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader,TensorDataset
        import torch.optim as optim
        import openslide
        print('torch version: %s, torchvision version: %s' % (torch.__version__, torchvision.__version__))

        if args.dl_feature == 'PanNuke_DL_pretrain':
            feature_name = 'DL_features_32'
            model_init = torchvision.models.convnext_tiny(weights=True, progress=True)
            from nuc_stat_DL import train_init, Thread1, Thread2, parmap
            from nuc_stat_DL import Convnext_tiny_DeepFeature as Net
            args.DL_model_dir = 'PanNuke_DL_pretrain_patchsize=64x64/model_statedict_epoch=100.torchmdl'

        print('feature name: %s' % args.dl_feature)

        if os.path.exists(opj(args.stardist_dir, '%s_patch%d.npy' % (feature_name, args.patch_size))):
            print('Found existed result. Skip.')
            exit()

        # Init and load model
        device = torch.device("cuda" if args.use_cuda else "cpu")
        train_init(seed=args.seed) # torch initialization must before the model created!
        model = Net(model_init, args.num_deepfeature, args.num_classes)
        state_dict_model_path = args.DL_model_dir
        model.load_state_dict(torch.load(state_dict_model_path))

        
        model = model.to(device)
        model = model.eval()
        
        slide = openslide.OpenSlide(args.slidepath)
        centroids = np.load(opj(args.stardist_dir, 'centroids.npy'), allow_pickle=True)
        print('Total number of nuclei: %d' % len(centroids))
        
        args.batch_size_read_data = 8192
        args.batch_size_deeplearning = 128
        thread1 = Thread1(args, slide, centroids, parmap)
        thread1.start()
        thread2 = Thread2(args, thread1, model, device)
        thread2.start()

        while True:
            if thread1.isFinished and thread2.isFinished:
                embedding = thread2.embedding.astype(np.float16)
                np.save(opj(args.stardist_dir, '%s_patch%d.npy' % (feature_name, args.patch_size)), embedding, allow_pickle=True)
                print('------------ all done ------------')
                break
            else:
                time.sleep(10)
        print('Stardist stage 3: deep learning feature extraction finished.')
