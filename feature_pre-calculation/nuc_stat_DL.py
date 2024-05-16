
import pandas as pd
import numpy as np
import os,sys,platform
import pickle
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import copy
import random
import time
from collections import Counter
from skimage.measure import regionprops
from PIL import Image
from sklearn.model_selection import KFold
opj = os.path.join

import threading
import time
import multiprocess as mp

import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
import torch.optim as optim
    

class Convnext_tiny_DeepFeature(nn.Module):
    def __init__(self, originalModel, num_deepfeature, num_classes):
        super(Convnext_tiny_DeepFeature, self).__init__()
        self.encoder = originalModel
        self.encoder.classifier = nn.Sequential(originalModel.classifier[0],
                                         originalModel.classifier[1],
                                         nn.Linear(originalModel.classifier[2].in_features, num_deepfeature),
                                         )
        self.classifier = nn.Linear(num_deepfeature, num_classes)

        
    def forward(self, x):
        deep_features = self.encoder(x)
        # print(deep_features.shape)
        out = self.classifier(deep_features)
        # print(out.shape)
        return deep_features, out

def train_init(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def test(args, model, device, test_loader, verbose=1):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        embedding = None
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            embed, output = model(data)
            output_softmax = F.log_softmax(output, dim=1)
            test_loss += F.nll_loss(output_softmax, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            if embedding is None:
                embedding = embed.data.cpu().numpy()
            else:
                embedding = np.concatenate([embedding, embed.data.cpu().numpy()], axis=0)

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)

    if verbose:
        print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            test_acc))
    
    return test_acc, embedding

        
def parfun(f, q_in, q_out):
    while True:
        i, x = q_in.get()
        if i is None:
            break
        q_out.put((i, f(x)))

def parmap(f, X, nprocs=mp.cpu_count()):
    q_in = mp.Queue(1)
    q_out = mp.Queue()
    proc = [mp.Process(target=parfun, args=(f, q_in, q_out)) for _ in range(nprocs)]
    for p in proc:
        p.daemon = True
        p.start()
    sent = [q_in.put((i, x)) for i, x in enumerate(X)]
    [q_in.put((None, None)) for _ in range(nprocs)]
    res = [q_out.get() for _ in range(len(sent))]
    [p.join() for p in proc]
    return [x for i, x in sorted(res)]

class Thread1(threading.Thread):
    def __init__(self, args, slide, centroids, parmap):
        threading.Thread.__init__(self)
        self.args = args
        self.slide = slide
        self.centroids = centroids
        self.parmap = parmap
        self.exit = False
        self.isFinished = False
        self.queue = []
        self.total_batches = int(np.ceil(len(self.centroids)/self.args.batch_size_read_data))
        
    # helper function to execute the threads
    def run(self):

        for b in tqdm(range(self.total_batches)):
            if self.exit:
                print('thread 1 exit.')
                break
            i_start = b*self.args.batch_size_read_data
            i_end = np.min([(b+1)*self.args.batch_size_read_data, len(self.centroids)])
            
            
            st = time.time()
            coordinate_list = []
            for i in range(i_start, i_end):

                if self.args.magnification is None or self.args.magnification == 40:
                    x1, y1 = int(self.centroids[i,0] - self.args.patch_size/2), int(self.centroids[i,1] - self.args.patch_size/2)
                elif self.args.magnification is not None and self.args.magnification == 20:
                    x1, y1 = int(self.centroids[i,0] - self.args.patch_size/2/2), int(self.centroids[i,1] - self.args.patch_size/2/2)

                coordinate_list.append((x1, y1))
            patches = self.parmap(lambda xy: self.read_region_parallel(xy),
                                  coordinate_list,
                                  # nprocs=6
                                  )
            patches = np.stack(patches)
            patch_all_0_to_1 = patches/255
            X = torch.FloatTensor(patch_all_0_to_1).swapaxes(3,1).swapaxes(3,2)
            y = torch.LongTensor(np.zeros(len(X)))
            tensorlist = [X, y]
            self.queue.append(tensorlist)
            read_image_cost = time.time()-st
            
            print('Read image costs %.2fs. Number of patches awaiting processing: %d' % (read_image_cost, len(self.queue)))
        
        self.isFinished = True
        
        
    def read_region_parallel(self, xy):

        if self.args.magnification is None or self.args.magnification == 40:
            patch = self.slide.read_region(location=xy, level=0, size=(self.args.patch_size, self.args.patch_size))

        elif self.args.magnification is not None and self.args.magnification == 20:
            # scale to 40x
            # Zoom factors: 2 for the first dimension, 2 for the second dimension, and 1 for the third dimension
            patch = self.slide.read_region(location=xy, level=0, size=(self.args.patch_size//2, self.args.patch_size//2))
            width, height = patch.size
            patch = patch.resize((width * 2, height * 2))
            
        patch = np.array(patch)[..., :3]
        return patch
 
    
class Thread2(threading.Thread):
    def __init__(self, args, thread1, model, device):
        threading.Thread.__init__(self)
        self.thread1 = thread1
        self.args = args
        self.model = model
        self.device = device
        self.isFinished = False
        self.embedding = None
        self.exit = False
        
        
    def run(self):
        
        while True:
            if self.exit:
                print('thread 2 exit.')
                break
            len_queue = len(self.thread1.queue)
            if len_queue == 0:
                if self.thread1.isFinished:
                    self.isFinished = True
                    break
                else:
                    time.sleep(20)
                    continue
            
            st = time.time()
            tensorlist = self.thread1.queue.pop(0)
            test_kwargs = {'batch_size': self.args.batch_size_deeplearning,
                            'num_workers': 0,
                            'pin_memory': False,
                            'shuffle': False}
            test_loader = torch.utils.data.DataLoader(TensorDataset(*tensorlist), **test_kwargs)
            test_acc, embed = test(self.args, self.model, self.device, test_loader, verbose=0)
            DL_cost = time.time()-st
            if self.embedding is None:
                self.embedding = embed
            else:
                self.embedding = np.concatenate([self.embedding, embed], axis=0)
            print('Deep learning costs %.2fs. Current Queue: %d -> %d' % (DL_cost, len_queue, len_queue-1))
