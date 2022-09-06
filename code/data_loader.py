import paddle
import numpy as np
import cv2
import os
import random
from PIL import Image


from paddle.vision.transforms import Compose, RandomCrop, ToTensor, RandomHorizontalFlip, RandomRotation
from paddle.vision.transforms import functional as F
from paddle.io import DataLoader, Dataset

class MyRotation(RandomRotation):
    def _apply_with_param(self, img, angle):
        return F.rotate(img, angle, self.interpolation, self.expand,
                        self.center, self.fill)

class MyFlip(RandomHorizontalFlip):
    def _apply_with_param(self, img, flip):
        if flip:
            return F.hflip(img)
        return img

class MyCrop(RandomCrop):
    def _apply_with_param(self, img:Image, param:tuple):
        '''
        img: (PIL Image)
        self.size: target (H, W)
        param: (i, j)
        '''
        i, j = param
        h, w, _ = np.array(img).shape
        # pad the width if needed
        if self.pad_if_needed and w < self.size[1]:
            img = F.pad(img, (self.size[1] - w, 0), self.fill,
                        self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and h < self.size[0]:
            img = F.pad(img, (0, self.size[0] - h), self.fill,
                        self.padding_mode)
        return F.crop(img, i, j, self.size[0], self.size[1])


class Folder(Dataset):
    def __init__(self, samples=None, Train=True):
        super().__init__()
        self.samples = samples
        self.Train = Train
        self.toTensor = ToTensor()
        if Train:
            self.rotation = MyRotation(degrees=10)
            self.flip = MyFlip(prob=0.3)
            self.crop = MyCrop(size=(512, 512), pad_if_needed=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path = self.samples[index]
        img = Image.open(image_path).convert('RGB')
        try:
            gt = Image.open(image_path.replace("images","gts")[:-4] + '.jpg').convert('RGB')
        except:
            gt = Image.open(image_path.replace("images","gts")[:-4] + '.png').convert('RGB')
        re = []

        if self.Train:
            try:
                mask = Image.open(image_path.replace("images","mask5")[:-4] + '.jpg').convert('RGB')
            except:
                mask = Image.open(image_path.replace("images","mask5")[:-4] + '.png').convert('RGB')
            rotationParam = random.uniform(-10, 10) if random.random() < 0.3 else 0
            flipParam = random.random() < 0.3
            H, W, _ = np.array(img).shape
            cropParam = (random.randint(0, max(H - 512, 0)), random.randint(0, max(W - 512, 0)))

            
            img = self.rotation._apply_with_param(img, rotationParam)
            img = self.flip._apply_with_param(img, flipParam)
            img = self.crop._apply_with_param(img, cropParam)

            gt = self.rotation._apply_with_param(gt, rotationParam)
            gt = self.flip._apply_with_param(gt, flipParam)
            gt = self.crop._apply_with_param(gt, cropParam)
            
            mask = self.rotation._apply_with_param(mask, rotationParam)
            mask = self.flip._apply_with_param(mask, flipParam)
            mask = self.crop._apply_with_param(mask, cropParam)            

            mask = Image.fromarray((255 - np.array(mask)).astype('uint8')).convert('RGB')
            mask = self.toTensor(mask)
            re = [mask]

        img = self.toTensor(img)
        gt = self.toTensor(gt)
        re = [img, gt] + re
        return re


class TrainValidDataset():
    def __init__(self, file_path, type='doc', ratio=0.07):
        self.path = []
        if type == 'doc':
            for sub in os.listdir(file_path+'/classone'):
                self.path.append(os.path.join(file_path, 'classone', sub))
        else:
            self.path.append(os.path.join(file_path, 'dehw_train_dataset'))

        self.image_list = []
        for p in self.path:
            data_path = os.path.join(p, 'images')
            self.image_list += [os.path.join(data_path, img_path) for img_path in os.listdir(data_path)]
        print("number of image_list:", len(self.image_list))

        Nvalid = int(ratio*len(self.image_list))
        np.random.shuffle(self.image_list)
        self.valid = self.image_list[:Nvalid]
        self.train = self.image_list[Nvalid:]
        print("number of valid:", len(self.valid))
        print("number of train:", len(self.train))

    
    def getData(self):
        return Folder(self.train, Train=True), Folder(self.valid, Train=False)
    