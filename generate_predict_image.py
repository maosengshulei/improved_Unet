import torch
import os
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet_models import UNet11
import torchvision
from torch.nn import functional as F
from PIL import Image
from torch.utils import data
from torch.autograd import Variable

checkpoint='/home/paperspace/DL/TernausNet/logs/best_result/model_best.pth.tar'
root = os.path.expanduser('~/data/datasets')

class TestPlaqueseg(data.Dataset):
    class_names=np.array(['background','plaque'])

    def __init__(self, root,transform=False):
        self.root = root

        self._transform = transform
        self.split='test'

        dataset_dir = os.path.join(self.root, 'unet_xinxueguan')
        self.files = []

        imgsets_file = os.path.join(
            dataset_dir, 'Segmentation/%s.txt' % self.split)
        for did in open(imgsets_file):
            did = did.strip()
            img_file = os.path.join(dataset_dir, 'IMAGES/%s' % did)
            self.files.append(img_file)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        # load image
        img_file = self.files[index]
        file_name=img_file.split('/')[-1]
        img = Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label

        if self._transform:
            return self.transform(img),file_name
        else:
            return img,file_name

    def transform(self, img):

        imf_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        return imf_transform(img)


#dataloader
if __name__ == '__main__':

    test_loader=DataLoader(dataset=TestPlaqueseg(root,transform=True),
        shuffle=False,
        batch_size=4,
        num_workers=4,
        pin_memory=True
        )
    new_root='./test'
    cuda=torch.cuda.is_available()
    model=UNet11(pretrained=False)
    check_point=torch.load(checkpoint)
    model.load_state_dict(check_point['model_state_dict'])
    if cuda:
        model=model.cuda()

    for batch_id,(input,file_name) in tqdm(enumerate(test_loader),total=len(test_loader),desc='predict'):
        if cuda:
            input=input.cuda()
        input=Variable(input,volatile=True)
        output=F.sigmoid(model(input))
        mask1=output.data>0.5
        mask2=output.data<=0.5
        output.data[mask1]=1
        output.data[mask2]=0
        output=torch.unsqueeze(output,1)
        print(output.data.shape)
        mask = (output.data.cpu().int() * 255)

        for i,filename in enumerate(file_name):
            img=mask[i,:,:,:]
            img=torch.squeeze(img,0)
            print(img.shape)
            dt_trans=torchvision.transforms.ToPILImage()
            img=dt_trans(img).convert('1')
            img.save(os.path.join(new_root,filename))
