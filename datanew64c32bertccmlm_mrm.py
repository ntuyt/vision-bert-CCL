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
import json
from transformers import BertTokenizer
import random

class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: conceptual_caption_precomp, SBU_precomp
    """
    def __init__(self, data_path, data_split, opt):
        loc = data_path + '/'
        # Captions
        self.captions = []
        self.max_seq_length = 16
        self.max_vision = 18

        self.imgfold = "/home/yutan/pretrain/%s_frcnn/"%data_split
        self.classfold = "/home/yutan/pretrain/%s_frcnn_class/"%data_split

        with open('dataset/cc/cc_%s_caps.txt'%data_split, 'r') as f:
            for line in f:
                self.captions.append(line)

        # Image features: a list image names
        self.images = json.load(open("dataset/cc/cc_%s_names.json"%data_split,"r"))#np.load(loc+'%s_64featc32.npy' % data_split)
        self.length = len(self.captions)
        self.tokenizer = BertTokenizer.from_pretrained('bert/')
        self.mlm = opt.mlm
        self.mrm = opt.mrm
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev' or data_split == 'test':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index#//self.im_div
        imgname = self.images[index]
        imgname = imgname.split('.')[0]+'.npy'
        imgname = self.imgfold + imgname
        classname = self.images[index].split('.')[0]+'.npy'
        classname = self.classfold + classname

        image = np.load(imgname)#torch.Tensor(np.load(imgname)).type(torch.float16)
        imglabel = np.load(classname)#torch.Tensor(np.load(classname)).long()#type(torch.float16)


        caption = self.captions[index]

        caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)
        caption_tokens, mlm_labels = self.random_word_wwm(caption_tokens)

        vision_zeros, vision_labels = self.random_mask_region(imglabel)
        for i in range(len(vision_zeros)):
             if vision_zeros[i] == 1:
                 image[i] = image[i]*0#np.zeros_like(image[i])#image[i].mul((1-vision_zeros[i]))
        image = torch.Tensor(image).type(torch.float16)


        text_tokens = ['[CLS]'] + caption_tokens + ['[SEP]']
        mlm_labels = [-1] + mlm_labels + [-1]

        caption = self.tokenizer.convert_tokens_to_ids(text_tokens) 
        input_mask = [1] * len(caption)

        vision_mask = [1] * image.size(0)

        while len(caption) < self.max_seq_length:
             caption.append(0)
             input_mask.append(0)
             mlm_labels.append(-1)  

        if image.size(0) < self.max_vision:
             addlen = self.max_vision-image.size(0)
             vision_mask = vision_mask + [0]*addlen
             vision_labels = vision_labels + [-1]*addlen
             imageadd = torch.zeros(addlen,2048).type(torch.float16)
             image = torch.cat([image, imageadd],0)#.type(torch.float16)

        if len(caption) > self.max_seq_length:
             caption = caption[:self.max_seq_length]
             input_mask = input_mask[:self.max_seq_length]
             mlm_labels = mlm_labels[:self.max_seq_length]

        if image.size(0) > self.max_vision:
             image = image[:self.max_vision]
             vision_mask = vision_mask[:self.max_vision]
             vision_labels = vision_labels[:self.max_vision]
             vision_zeros = vision_zeros[:self.max_vision]
        # mask the region features    

        target = torch.Tensor(caption)
        target_mask = torch.Tensor(input_mask)
        vision_mask = torch.Tensor(vision_mask)
        mlm_labels = torch.Tensor(mlm_labels)
        vision_labels = torch.Tensor(vision_labels)
        return image, target, target_mask, vision_mask, mlm_labels,vision_labels#index#image, target, #index, img_id

    def random_word_wwm(self, tokens):
        output_tokens = []
        output_label = []

        for i, token in enumerate(tokens):
            sub_tokens = self.tokenizer.wordpiece_tokenizer.tokenize(token)
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.35 and self.mlm:
                prob /= 0.35
                # 80% randomly change token to mask token
                if prob < 0.8:
                    for sub_token in sub_tokens:
                        output_tokens.append("[MASK]")
                # 10% randomly change token to random token
                elif prob < 0.9:
                    for sub_token in sub_tokens:
                        output_tokens.append(random.choice(list(self.tokenizer.vocab.keys())))
                        # -> rest 10% randomly keep current token
                else:
                    for sub_token in sub_tokens:
                        output_tokens.append(sub_token)
                        # append current token to output (we will predict these later)
                for sub_token in sub_tokens:
                    try:
                        output_label.append(self.tokenizer.vocab[sub_token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        output_label.append(self.tokenizer.vocab["[UNK]"])
                        logging.warning("Cannot find sub_token '{}' in vocab. Using [UNK] insetad".format(sub_token))
            else:
                for sub_token in sub_tokens:
                    # no masking token (will be ignored by loss function later)
                    output_tokens.append(sub_token)
                    output_label.append(-1)

        return output_tokens, output_label

    def random_mask_region(self, regions_cls):
        num_regions = regions_cls.shape[0]
        output_op = []
        output_label = []
        for k  in range(num_regions):#enumerate(regions_cls_scores):
            prob = random.random()
            # mask region with 15% probability
            if prob < 0.15 and self.mrm:
                prob /= 0.15
                if prob < 0.9:
                    # 90% randomly replace appearance feature by "MASK"
                    output_op.append(1)
                else:
                    # -> rest 10% randomly keep current appearance feature
                    output_op.append(0)
                # append class of region to output (we will predict these later)
                output_label.append(regions_cls[k])
            else:
                # no masking region (will be ignored by loss function later)
                output_op.append(0)
                output_label.append(-1)

        return output_op, output_label

    @staticmethod
    def b64_decode(string):
        return base64.decodebytes(string.encode())

    @staticmethod
    def group_aspect(database):
        print('grouping aspect...')
        t = time.time()

        # get shape of all images
        widths = torch.as_tensor([idb['width'] for idb in database])
        heights = torch.as_tensor([idb['height'] for idb in database])

        # group
        group_ids = torch.zeros(len(database))
        horz = widths >= heights
        vert = 1 - horz
        group_ids[horz] = 0
        group_ids[vert] = 1

        print('Done (t={:.2f}s)'.format(time.time() - t))

        return group_ids

    def __len__(self):
        return len(self.ids)

    def _load_image(self, path):
        if '.zip@' in path:
            return self.zipreader.imread(path).convert('RGB')
        else:
            return Image.open(path).convert('RGB')

    def _load_json(self, path):
        if '.zip@' in path:
            f = self.zipreader.read(path)
            return json.loads(f.decode())
        else:
            with open(path, 'r') as f:
                return json.load(f)  

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
    images, captions, cap_mask, vision_mask, labels, vision_labels = zip(*data)

    images = torch.stack(images, 0)
    labels = torch.stack(labels, 0)
    vision_labels = torch.stack(vision_labels, 0).long()
    targets = torch.stack(captions, 0).long()
    cap_mask = torch.stack(cap_mask,0).long()
    vision_mask = torch.stack(vision_mask,0).long()

    return images, targets, cap_mask, vision_mask, labels, vision_labels



def get_precomp_loader(data_path, data_split, opt, batch_size=100,
                       shuffle=True, num_workers=16):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, data_split, opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn,num_workers = num_workers)
    return data_loader

def get_loaders(data_name, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader(dpath, 'train', opt,
                                      batch_size, True, workers)
    return train_loader#, val_loader


def get_test_loader(split_name, data_name, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_precomp_loader(dpath, split_name, opt,
                                     batch_size, False, workers)
    return test_loader



