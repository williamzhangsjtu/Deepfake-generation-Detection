import torch 
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from glob import glob
from PIL import Image
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import os
import h5py
import utils


class single_instance(Dataset):
    
    def __init__(self, data_h5, label_dict, transform):

        self.data = None
        self.data_h5 = data_h5
        self.transform = transform
        self.labels = label_dict
        self.idxstart = {}
        self.num2idx = {}
        with h5py.File(data_h5, 'r') as f:
            startfrom = 0
            for key in label_dict:
                self.idxstart[key] = startfrom
                for i in range(startfrom, startfrom + len(f[key])):
                    self.num2idx[i] = key
                startfrom += len(f[key])
        

    def __getitem__(self, idx):
        
        if (self.data is None):
            self.data = h5py.File(self.data_h5, 'r')
        index = self.num2idx[idx]
        startfrom = self.idxstart[index]

        img = self.data[str(index)][idx-startfrom]  # 3 x H x W
        
        return self.transform(torch.as_tensor(img)), self.labels[index], 1, index
    def __len__(self):
        return len(self.num2idx)


class default_dataset(Dataset):

    def __init__(self, h5, label_dict, transform, time_step=4):
        self.label_dict = label_dict
        self.data = None
        self.data_h5 = h5
        self.transform = transform

        self.indexes = []
        self.range = []
        with h5py.File(h5, 'r') as f:
            for idx in label_dict:
                if (idx not in f.keys()): 
                    continue
                    
                length = len(f[idx])
                if time_step is None:
                    self.indexes.append(idx)
                    self.range.append([0, length])
                else:
                    for i in range(length// time_step):
                        self.indexes.append(idx)
                        self.range.append([i * time_step, (i + 1) * time_step])
                    if (length// time_step) * time_step < length - 3:
                        self.indexes.append(idx)
                        self.range.append([(length// time_step) * time_step, length])

    def __getitem__(self, idx):
        
        if (self.data is None):
            self.data = h5py.File(self.data_h5, 'r')
        index = self.indexes[idx]
        rg = self.range[idx]
        data = None
        imgs = self.data[str(index)][rg[0]:rg[1]]  # T x 3 x H x W
        for img in imgs:
            data = torch.cat((data, self.transform(torch.as_tensor(img)).unsqueeze(0)), dim=0) \
                    if data is not None else self.transform(torch.as_tensor(img)).unsqueeze(0)

        return data, self.label_dict[index], len(data), index


    def __len__(self):
        return len(self.indexes)

def collate_fn(batch):
    batch.sort(key=lambda d: len(d[0]), reverse=True)
    images, labels, nums, indexes = zip(*batch)
    images = pad_sequence(images, batch_first=True)
    return images, torch.as_tensor(labels), \
        torch.as_tensor(nums, dtype=torch.long), indexes



def get_dataloader(h5, label_dict, transform, T=4, **kwargs):
    """
    Image in root/Exxxx/name.jpg
    label_dict: {Exxxx: label}
    """
    kwargs.setdefault("batch_size", 8)
    kwargs.setdefault("num_workers", 4)
    kwargs.setdefault("shuffle", True)



    
    if T == 1:
        _dataset = single_instance(h5, label_dict, transform)
        return DataLoader(_dataset, **kwargs)
    else:
        _dataset = default_dataset(h5, label_dict, transform, T)
        return DataLoader(_dataset, collate_fn=collate_fn, **kwargs)