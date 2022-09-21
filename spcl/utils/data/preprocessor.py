from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image

class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, index


class Preprocessor2(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor2, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, viewid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid,viewid, camid, index

class Preprocessor_aug(Dataset):
    def __init__(self, dataset, root=None, transform=None, selected_list = None):
        super(Preprocessor_aug, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.select_list = selected_list

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        if fpath in self.select_list.keys():
            aug_path = random.choice(self.select_list[fpath])
            aug_path = os.path.join("/home/lzy/VDG/SpCL/market_train_fpn_final", aug_path)
        else:
            aug_path = fpath
        img = Image.open(fpath).convert('RGB')
        img_aug = Image.open(aug_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img_aug = self.transform(img_aug)
            
        return [img, img_aug], fname, pid, camid, index




class Preprocessor_aug2(Dataset):
    def __init__(self, dataset, root=None, transform=None, selected_list = None):
        super(Preprocessor_aug2, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.select_list = selected_list
        self.id_tranform = {0:0, 1:1, 2:2, 3:1}   #,4:2,5:3,6:3,7:4
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, viewid, camid,  = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        if fpath in self.select_list.keys():
            aug_path = random.choice(self.select_list[fpath])
            aug_path = os.path.join("/home/lzy/VDG/SpCL/examples/data/market_train_fpn_final", aug_path)
            view_aug = int(aug_path[-5:-4])
        else:
            aug_path = fpath 
            view_aug =  viewid
        img = Image.open(fpath).convert('RGB')
        img_aug = Image.open(aug_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img_aug = self.transform(img_aug)
        viewid = self.id_tranform[viewid]
        view_aug = self.id_tranform[view_aug]
        return [img, img_aug], fname, pid, [viewid, view_aug], index