import collections
import os.path as osp

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data

import torchvision

class Plaqueseg(data.Dataset):
    class_names=np.array(['background','plaque'])
    #mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform


        dataset_dir = osp.join(self.root, 'unet_xinxueguan')
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(
            dataset_dir, 'Segmentation/%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(dataset_dir, 'IMAGES/%s' % did)
                lbl_file = osp.join(
                    dataset_dir, 'SegmentationClass/%s' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        #img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        #img -= self.mean_bgr
        dt_trans=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
        img=dt_trans(img)
        #img = img.transpose(2, 0, 1)
        ###img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).float()
        return img, lbl

    def untransform(self, img, lbl):
        #img = img.numpy()
        #img = img.transpose(1, 2, 0)
        #img += self.mean_bgr
        #img = img.astype(np.uint8)
        #img = img[:, :, ::-1]
        lbl = lbl.numpy().astype(int)
        return img, lbl
