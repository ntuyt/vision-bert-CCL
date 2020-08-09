# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Data provider"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
import numpy as np
import json as jsonmod


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split):
        loc = data_path + '/'

        # Captions
        self.captions = []
        self.max_seq_length = 16

        #with open(loc+'%s_caps.txt' % data_split, 'rb') as f:
        #    for line in f:
        #        self.captions.append(line.strip())
        with open('dataset/coco/%s_caps.txt.bt'%data_split, 'r') as f:
            for line in f:
                arr = line.strip().split()
                arr = [int(astr) for astr in arr]
                self.captions.append(arr)

        # Image features
        self.images = np.load('dataset/coco/%s_100featc16.npy' % data_split)
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev' or data_split == 'test':
            self.length = 5000
       
    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index//self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index].copy()
        input_mask = [1] * len(caption)
        vision_mask = [1] * image.size(0)  
        while len(caption) < self.max_seq_length:
             caption.append(0)
             input_mask.append(0)
        if len(caption) > self.max_seq_length:
             caption = caption[:self.max_seq_length]
             input_mask = input_mask[:self.max_seq_length]   
        target = torch.Tensor(caption)
        target_mask = torch.Tensor(input_mask)
        vision_mask = torch.Tensor(vision_mask)

        return image, target, target_mask, vision_mask, index#image, target, #index, img_id

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    images, captions, cap_mask, vision_mask, ids = zip(*data)

    images = torch.stack(images, 0)
    targets = torch.stack(captions, 0).long()
    cap_mask = torch.stack(cap_mask,0).long()
    vision_mask = torch.stack(vision_mask,0).long()

    return images, targets, cap_mask, vision_mask, ids


def get_precomp_loader(data_path, data_split, opt, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn, num_workers = num_workers)
    return data_loader

def get_loaders(data_name, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader(dpath, 'train', opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(dpath, 'test', opt,
                                    batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_precomp_loader(dpath, split_name, opt,
                                     batch_size, False, workers)
    return test_loader
