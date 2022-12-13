
#############################################################################
##
## Copyright (C) 2022 nuclei.io
## Contact: https://www.nuclei.io/
##
## This script author: Zhi Huang
## This script contributors: Zhi Huang
## This script created on: 08/08/2022
#############################################################################

import pandas as pd
import numpy as np
import os
import ast
import time

class SlideStatistics():
    index = None
    contour = None
    centroid = None
    probability = None

    learning_features = [('Color','Grey_mean'),
                        ('Color','Grey_std'),
                        ('Color','Grey_min'),
                        ('Color','Grey_max'),
                        ('Color','R_mean'),
                        ('Color','R_std'),
                        ('Color','R_min'),
                        ('Color','R_max'),
                        ('Color','G_mean'),
                        ('Color','G_std'),
                        ('Color','G_min'),
                        ('Color','G_max'),
                        ('Color','B_mean'),
                        ('Color','B_std'),
                        ('Color','B_min'),
                        ('Color','B_max'),

                        ('Color - cytoplasm','cyto_bg_mask_ratio'),
                        ('Color - cytoplasm','cyto_cytomask_ratio'),
                        ('Color - cytoplasm','cyto_Grey_mean'),
                        ('Color - cytoplasm','cyto_Grey_std'),
                        ('Color - cytoplasm','cyto_Grey_min'),
                        ('Color - cytoplasm','cyto_Grey_max'),
                        ('Color - cytoplasm','cyto_R_mean'),
                        ('Color - cytoplasm','cyto_R_std'),
                        ('Color - cytoplasm','cyto_R_min'),
                        ('Color - cytoplasm','cyto_R_max'),
                        ('Color - cytoplasm','cyto_G_mean'),
                        ('Color - cytoplasm','cyto_G_std'),
                        ('Color - cytoplasm','cyto_G_min'),
                        ('Color - cytoplasm','cyto_G_max'),
                        ('Color - cytoplasm','cyto_B_mean'),
                        ('Color - cytoplasm','cyto_B_std'),
                        ('Color - cytoplasm','cyto_B_min'),
                        ('Color - cytoplasm','cyto_B_max'),

                        ('Morphology','major_axis_length'),
                        ('Morphology','minor_axis_length'),
                        ('Morphology','major_minor_ratio'),
                        ('Morphology','orientation'),
                        ('Morphology','area'),
                        ('Morphology','extent'),
                        ('Morphology','solidity'),
                        ('Morphology','convex_area'),
                        ('Morphology','Eccentricity'),
                        ('Morphology','equivalent_diameter'),
                        ('Morphology','perimeter'),
                        ('Morphology','perimeter_crofton'),

                        ('Haralick','heterogeneity'),
                        ('Haralick','contrast'),
                        ('Haralick','dissimilarity'),
                        ('Haralick','ASM'),
                        ('Haralick','energy'),
                        ('Haralick','correlation'),

                        ('Gradient','Mag.Mean'),
                        ('Gradient','Mag.Std'),
                        ('Gradient','Mag.Skewness'),
                        ('Gradient','Mag.Kurtosis'),
                        ('Gradient','Mag.HistEntropy'),
                        ('Gradient','Mag.HistEnergy'),
                        ('Gradient','Canny.Sum'),
                        ('Gradient','Canny.Mean'),

                        ('Intensity','Min'),
                        ('Intensity','Max'),
                        ('Intensity','Mean'),
                        ('Intensity','Median'),
                        ('Intensity','MeanMedianDiff'),
                        ('Intensity','Std'),
                        ('Intensity','IQR'),
                        ('Intensity','MAD'),
                        ('Intensity','Skewness'),
                        ('Intensity','Kurtosis'),
                        ('Intensity','HistEnergy'),
                        ('Intensity','HistEntropy'),

                        ('FSD','Shape.FSD1'),
                        ('FSD','Shape.FSD2'),
                        ('FSD','Shape.FSD3'),
                        ('FSD','Shape.FSD4'),
                        ('FSD','Shape.FSD5'),
                        ('FSD','Shape.FSD6'),

                        ('Spatial - Delaunay', 'dist.mean'), ('Spatial - Delaunay', 'dist.std'), ('Spatial - Delaunay', 'dist.min'), ('Spatial - Delaunay', 'dist.max'),
                        ('Spatial - Delaunay', 'cosine.Color.mean'), ('Spatial - Delaunay', 'cosine.Color.std'), ('Spatial - Delaunay', 'cosine.Color.min'), ('Spatial - Delaunay', 'cosine.Color.max'),
                        ('Spatial - Delaunay', 'cosine.Morphology.mean'), ('Spatial - Delaunay', 'cosine.Morphology.std'), ('Spatial - Delaunay', 'cosine.Morphology.min'), ('Spatial - Delaunay', 'cosine.Morphology.max'),
                        ('Spatial - Delaunay', 'cosine.Color - cytoplasm.mean'), ('Spatial - Delaunay', 'cosine.Color - cytoplasm.std'), ('Spatial - Delaunay', 'cosine.Color - cytoplasm.min'), ('Spatial - Delaunay', 'cosine.Color - cytoplasm.max'),
                        ('Spatial - Delaunay', 'cosine.Haralick.mean'), ('Spatial - Delaunay', 'cosine.Haralick.std'), ('Spatial - Delaunay', 'cosine.Haralick.min'), ('Spatial - Delaunay', 'cosine.Haralick.max'),
                        ('Spatial - Delaunay', 'cosine.Gradient.mean'), ('Spatial - Delaunay', 'cosine.Gradient.std'), ('Spatial - Delaunay', 'cosine.Gradient.min'), ('Spatial - Delaunay', 'cosine.Gradient.max'),
                        ('Spatial - Delaunay', 'cosine.Intensity.mean'), ('Spatial - Delaunay', 'cosine.Intensity.std'), ('Spatial - Delaunay', 'cosine.Intensity.min'), ('Spatial - Delaunay', 'cosine.Intensity.max'),
                        ('Spatial - Delaunay', 'cosine.FSD.mean'), ('Spatial - Delaunay', 'cosine.FSD.std'), ('Spatial - Delaunay', 'cosine.FSD.min'), ('Spatial - Delaunay', 'cosine.FSD.max'),
                        ('Spatial - Delaunay', 'neighbour.area.mean'), ('Spatial - Delaunay', 'neighbour.area.std'),
                        ('Spatial - Delaunay', 'neighbour.heterogeneity.mean'), ('Spatial - Delaunay', 'neighbour.heterogeneity.std'),
                        ('Spatial - Delaunay', 'neighbour.orientation.mean'), ('Spatial - Delaunay', 'neighbour.orientation.std'),
                        ('Spatial - Delaunay', 'neighbour.Grey_mean.mean'), ('Spatial - Delaunay', 'neighbour.Grey_mean.std'),
                        ('Spatial - Delaunay', 'neighbour.cyto_Grey_mean.mean'), ('Spatial - Delaunay', 'neighbour.cyto_Grey_mean.std'),
                        ('Spatial - Delaunay', 'neighbour.Polar.phi.mean'), ('Spatial - Delaunay', 'neighbour.Polar.phi.std'),
                        ]


    def __init__(self,
                MainWindow,
                datafolder,
                preload=True
                ):
        '''
        If preload == True, we will load statistics all at once,
        and stored into the memory.
        It will be slow when open a new slide,
        but will be faster after initialization.
        Preload also requires large memory (recommend > 16GB).

        If preload == False, we will load statistics dynamically.
        It will be no delay when open a new slide,
        but will not be as fast as preload.

        '''
        super(SlideStatistics, self).__init__()
        self.datafolder = datafolder
        self.MainWindow = MainWindow
        self.preload = preload
        

        self.loadStatistics()



    def loadStatistics(self):
        st = time.time()
        if self.preload:
            print('Preload all statistics at once.')
            self.MainWindow.log.write('Preload all statistics at once.')

            folder_name = os.path.basename(self.datafolder)
            parent_folder_name = self.datafolder.rstrip(os.sep).split(os.sep)[-2]

            if (folder_name == 'stardist_results') or (parent_folder_name == 'stardist_results'):
                self.index = pd.read_csv(os.path.join(self.datafolder, 'nuclei_stat_index.csv'), index_col=0).values.reshape(-1).astype(int)
                self.contour = np.load(os.path.join(self.datafolder, 'contours.npy'), allow_pickle=True)
                self.centroid = np.load(os.path.join(self.datafolder, 'centroids.npy'), allow_pickle=True)
                self.probability = np.load(os.path.join(self.datafolder, 'probability.npy'), allow_pickle=True)
            
                if os.path.exists(os.path.join(self.datafolder, 'nuclei_stat_f16.npy')):
                    self.feature = np.load(os.path.join(self.datafolder, 'nuclei_stat_f16.npy'))
                else:
                    self.feature = np.load(os.path.join(self.datafolder, 'nuclei_stat.npy'))

                if os.path.exists(os.path.join(self.datafolder, 'DL_features_32.npy')):
                    print('Found deep learning features (patch=128).')
                    self.feature = np.concatenate([self.feature,
                                                    np.load(os.path.join(self.datafolder, 'DL_features_32.npy'))],
                                                    axis=1)
                                                            
                if os.path.exists(os.path.join(self.datafolder, 'DL_features_32_patch64.npy')):
                    print('Found deep learning features (patch=64).')
                    self.feature = np.concatenate([self.feature,
                                                    np.load(os.path.join(self.datafolder, 'DL_features_32_patch64.npy'))],
                                                    axis=1)

                if os.path.exists(os.path.join(self.datafolder, 'nuclei_stat_plasma32_f16.npy')):
                    print('Found plasma cell features (N_feat = 32).')
                    self.feature = np.concatenate([self.feature,
                                                    np.load(os.path.join(self.datafolder, 'nuclei_stat_plasma32_f16.npy'))],
                                                    axis=1)

                if os.path.exists(os.path.join(self.datafolder, 'DL_plasma_32_patch128.npy')):
                    print('Found deep learning features (plasma, patch=128).')
                    self.feature = np.concatenate([self.feature,
                                                    np.load(os.path.join(self.datafolder, 'DL_plasma_32_patch128.npy'))],
                                                    axis=1)
                                                            
                prob_threshold = 0.5
                self.select_nuclei_idx = self.probability > prob_threshold
                print('%d / %d' % (np.sum(self.select_nuclei_idx),len(self.select_nuclei_idx)) )
                self.contour = self.contour[self.select_nuclei_idx, ...]
                self.centroid = self.centroid[self.select_nuclei_idx, ...]
                self.feature = self.feature[self.select_nuclei_idx, ...]
                self.index = np.arange(len(self.centroid))

                self.feature[np.isnan(self.feature)] = 0
                self.feature_columns = pd.read_csv(os.path.join(self.datafolder, 'nuclei_stat_columns.csv'), index_col=0).values[:,0]
                self.feature_columns = np.array([ast.literal_eval(v) for v in self.feature_columns])
                for i in range(len(self.feature_columns)):
                    if self.feature_columns[i,0] == 'Gradient':
                        self.feature_columns[i,1] = self.feature_columns[i,1].replace('Gradient.','')
                    if self.feature_columns[i,0] == 'Intensity':
                        self.feature_columns[i,1] = self.feature_columns[i,1].replace('Intensity.','')
                self.feature_columns = list(tuple(map(tuple, self.feature_columns)))

                if os.path.exists(os.path.join(self.datafolder, 'DL_features_32.npy')):
                    self.feature_columns += [('DL patch 128', '%d' % v) for v in range(1, 32+1)]
                    self.learning_features += [('DL patch 128', '%d' % v) for v in range(1, 32+1)]

                if os.path.exists(os.path.join(self.datafolder, 'DL_features_32_patch64.npy')):
                    self.feature_columns += [('DL patch 64', '%d' % v) for v in range(1, 32+1)]
                    self.learning_features += [('DL patch 64', '%d' % v) for v in range(1, 32+1)]

                if os.path.exists(os.path.join(self.datafolder, 'DL_plasma_32_patch128.npy')):
                    self.feature_columns += [('DL plasma 32', '%d' % v) for v in range(1, 32+1)]
                    self.learning_features += [('DL plasma 32', '%d' % v) for v in range(1, 32+1)]

                if os.path.exists(os.path.join(self.datafolder, 'nuclei_stat_plasma32_f16.npy')):
                    self.feature_columns += [('Plasma polar', '%d' % v) for v in range(1, 32+1)]
                    self.learning_features += [('Plasma polar', '%d' % v) for v in range(1, 32+1)]

            
            elif folder_name == 'statistics':

                self.index = pd.read_csv(os.path.join(self.datafolder, 'nuclei_stat_index.csv'), index_col=0).values.reshape(-1).astype(int)
                self.contour = np.fromfile(os.path.join(self.datafolder, 'contour_uint32.npy'), dtype=np.uint32).reshape((len(self.index), -1, 2))
                self.centroid = np.fromfile(os.path.join(self.datafolder, 'centroid.npy'), dtype=float).reshape((len(self.index), -1))

                self.index = np.arange(len(self.centroid))

                if os.path.exists(os.path.join(self.datafolder, 'nuclei_stat_f16.npy')):
                    self.feature = np.fromfile(os.path.join(self.datafolder, 'nuclei_stat_f16.npy'), dtype=float).reshape((len(self.index), -1))
                else:
                    self.feature = np.fromfile(os.path.join(self.datafolder, 'nuclei_stat.npy'), dtype=float).reshape((len(self.index), -1))

                self.feature[np.isnan(self.feature)] = 0
                self.feature_columns = pd.read_csv(os.path.join(self.datafolder, 'nuclei_stat_columns.csv'), index_col=0).values[:,0]
            
            
            else:
                pass

            print('Nuclei stat data loaded, took %.2f seconds.' % (time.time()-st))
            print('Total number of nuclei:', len(self.centroid))
            self.MainWindow.log.write('Nuclei stat data loaded, took %.2f seconds. Total number of nuclei: %d' % (time.time()-st, len(self.centroid)))
            
            
            # Create initial prediction
            self.prediction = pd.DataFrame(np.stack((np.repeat(0,len(self.index)),
                                                                np.repeat(1,len(self.index)),
                                                                np.repeat('Other',len(self.index)),
                                                                np.repeat(200,len(self.index)),
                                                                np.repeat(200,len(self.index)),
                                                                np.repeat(200,len(self.index)),
                                                                )).T,
                                                        index=self.index,
                                                        columns = ['label','proba','class_name','color_r','color_g','color_b'])
 
        else: # no preload
            pass

