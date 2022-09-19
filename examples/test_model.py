from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import os 
sys.path.append(os.getcwd())
import collections
import copy
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from spcl import datasets
from spcl import models
from spcl.models.hm import HybridMemory, ClusterMemory, ClusterMemory2
from spcl.trainers import SpCLTrainer_USL, VDGTrainer_USL
from spcl.evaluators import Evaluator, extract_features, extract_aug_features
from spcl.utils.data import IterLoader
from spcl.utils.data import transforms as T
from spcl.utils.data.sampler import RandomMultipleGallerySampler
from spcl.utils.data.preprocessor import Preprocessor,Preprocessor_aug
from spcl.utils.logging import Logger
from spcl.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from spcl.utils.faiss_rerank import compute_jaccard_distance

from collections import OrderedDict
import torch
weights = torch.load("/home/lzy/VDG/iteration_200000.pt")['state_dict']

# print(type(weights['state_dict'])) #.keys()
body_dict = OrderedDict()
for key in weights.keys():
    if 'body' in key:
        body_dict["base."+key[18:]]=weights[key]
model = models.create('resnet50', num_features=0, norm=True, dropout=0, num_classes=0)
# use CUDA
model.cuda()
print(len(model.state_dict().keys()))
print()
# model.load_state_dict(body_dict)
model = nn.DataParallel(model)
model_key =list( model.state_dict().keys())
body_key = list( body_dict.keys())
for i in range(len(body_dict.keys())):
    model.state_dict()[model_key[i]].copy_(body_dict[body_key[i]])
# for param, param_body in zip(model.state_dict(), body_dict)
