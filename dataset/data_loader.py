from email.mime import image
import paddle
import numpy as np
import cv2
import os
import random
from PIL import Image


from paddle.vision.transforms import Compose, RandomCrop, ToTensor
from paddle.vision.transforms import functional as F


# 随机水平翻转
def random_horizontal_flip(imgs):
    if random.random() < 0.3:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
    return imgs


# 随机旋转
def random_rotate(imgs):
    if random.random() < 0.3:
        max_angle = 10
        angle = random.random() * 2 * max_angle - max_angle
        for i in range(len(imgs)):
            img = np.array(imgs[i])
            w, h = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
            imgs[i] = Image.fromarray(img_rotation)
    return imgs


def ImageTransform():
    return Compose([ToTensor(), ])






def random_horizontal_flip(imgs):
    if random.random() < 0.3:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
    return imgs

def random_rotate(imgs):
    if random.random() < 0.3:
        max_angle = 10
        angle = random.random() * 2 * max_angle - max_angle
        # print(angle)
        for i in range(len(imgs)):
            img = np.array(imgs[i])
            w, h = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
            imgs[i] =Image.fromarray(img_rotation)
    return imgs

class TrainDataSet(paddle.io.Dataset):

    def __init__(self, training=True, file_path=None):
        super().__init__()
        self.training = training
        self.path = file_path
        data_dirs = os.listdir(file_path+'/classone')
        i = random.randint(0, len(data_dirs)-1)
        data_path = self.path +'/classone/'+ data_dirs[i]+ '/images'
        # data_path2 = self.path +'/dehw_train_dataset'+ '/images'
        self.image_list = [os.path.join(data_path, img_path) for img_path in os.listdir(data_path)]
        # self.image_list += [os.path.join(data_path2, img_path) for img_path in os.listdir(data_path2)]
        print("number of images:", len(self.image_list))
        # self.image_list = os.listdir(self.path + '/images')
        # self.image_path = self.path + '/images'
        # self.gt_path = self.path + '/gts'
        # self.mask_path = self.path + '/mask'
        self.ImgTrans = ImageTransform()
        self.RandomCropparam = RandomCrop(512, pad_if_needed=True)

    def __len__(self):
        return len(self.image_list)
    def _cal_mask(self, img, gt):
        kernel = np.ones((2, 2), np.uint8)
        # mask = cv2.erode(np.uint8(mask),  kernel, iterations=2)
        # threshold = 25
        threshold = 10
        diff_image = np.abs(img.astype(np.float32) - gt.astype(np.float32))
        mean_image = np.mean(diff_image, axis=-1)
        mask = np.greater(mean_image, threshold).astype(np.uint8)
        mask = (1 - mask) * 255
        mask = cv2.erode(np.uint8(mask),  kernel, iterations=1)
        return np.expand_dims(np.uint8(mask),axis=2).repeat(3,axis=2)
    # noinspection PyProtectedMember
    def __getitem__(self, index):

        while True:
            image_path = self.image_list[index]

            img = Image.open(image_path).convert('RGB')
        # while True:
            try:
                gt = Image.open(image_path.replace("images","gts")[:-4] + '.jpg').convert('RGB')
            except:
                gt = Image.open(image_path.replace("images","gts")[:-4] + '.png').convert('RGB')
            try:
                mask = Image.open(image_path.replace("images","mask2")[:-4] + '.jpg').convert('RGB')
            except:
                mask = Image.open(image_path.replace("images","mask2")[:-4] + '.png').convert('RGB')
            if np.array(img).shape[0] <= 512 or np.array(img).shape[1] <= 512:
                index += 1
            else:
                break


        # if self.training:
        # # ### for data augmentation
        all_input = [img, mask, gt]
        all_input = random_horizontal_flip(all_input)   
        # all_input = random_rotate(all_input)
        img = all_input[0]
        mask = all_input[1]
        gt = all_input[2]
        # mask = self._cal_mask(np.array(gt), np.array(img))
        param = self.RandomCropparam._get_param(img.convert('RGB'), (512, 512))
        inputImage = F.crop(img.convert('RGB'), *param)
        maskIn = F.crop(255 - np.array(mask), *param)

        groundTruth = F.crop(gt.convert('RGB'), *param)
        del img
        del gt
        del mask
        inputImage = self.ImgTrans(inputImage)
        maskIn = self.ImgTrans(maskIn)
        # maskIn = maskIn[2:,:,:]
        groundTruth = self.ImgTrans(groundTruth)

        return inputImage, groundTruth, maskIn




# 验证数据集
class ValidDataSet(paddle.io.Dataset):
    def __init__(self, file_path=None):
        super().__init__()

        self.path = file_path
        data_dirs = os.listdir(file_path+'/classone')
        i = random.randint(0, len(data_dirs)-1)
        data_path = self.path +'/classone/'+ data_dirs[i]+ '/images'
        data_path2 = self.path +'/dehw_train_dataset'+ '/images'
        self.image_list = [os.path.join(data_path, img_path) for img_path in os.listdir(data_path)]
        self.image_list += [os.path.join(data_path2, img_path) for img_path in os.listdir(data_path2)]
        print("number of images:", len(self.image_list))
        # self.image_list = os.listdir(self.path + '/images')

        # self.path = file_path
        # self.image_list = os.listdir(self.path + '/images')
        # self.image_path = self.path + '/images'

        # self.gt_path = self.path + '/gts'
        # self.gt_list = os.listdir(self.gt_path)

        # self.mask_path = self.path + '/mask'
        # self.mask_list = os.listdir(self.mask_path)
        self.ImgTrans = ImageTransform()

    def __getitem__(self, index):
        image_path = self.image_list[index]
        img = Image.open(image_path).convert('RGB')
        try:
            gt = Image.open(image_path.replace("images","gts")[:-4] + '.jpg').convert('RGB')
        except:
            gt = Image.open(image_path.replace("images","gts")[:-4] + '.jpg').convert('RGB')

        inputImage = self.ImgTrans(img)
        groundTruth = self.ImgTrans(gt)

        return inputImage, groundTruth
    # 200张做验证
    def __len__(self):
        return 30



# # 验证数据集
# class ValidDataSet(paddle.io.Dataset):
#     def __init__(self, file_path=None):
#         super().__init__()
#         self.path = file_path
#         self.image_list = os.listdir(self.path + '/images')
#         self.image_list = [path for path in self.image_list if "dehw_train" not in path]
#         print("validation length", len(self.image_list))
#         self.image_path = self.path + '/images'

#         self.gt_path = self.path + '/gts'
#         self.gt_list = os.listdir(self.gt_path)

#         self.mask_path = self.path + '/mask'
#         self.mask_list = os.listdir(self.mask_path)

#         self.ImgTrans = ImageTransform()

#     def __getitem__(self, index):
#         image_path = self.image_list[index]

#         img = Image.open(self.image_path + '/' + image_path).convert('RGB')
#         try:
#             gt = Image.open(self.gt_path + '/' + image_path[:-4] + '.jpg').convert('RGB')
#         except:
#             gt = Image.open(self.gt_path + '/' + image_path[:-4] + '.png').convert('RGB')


#         inputImage = self.ImgTrans(img)
#         groundTruth = self.ImgTrans(gt)

#         return inputImage, groundTruth

#     # 200张做验证
#     def __len__(self):
#         return 30


# 验证数据集
# class ValidDataSetDebug(paddle.io.Dataset):
#     def __init__(self, file_path=None):
#         super().__init__()
#         self.path = file_path
#         self.image_list = os.listdir(self.path + '/images')
#         self.image_path = self.path + '/images'

#         self.gt_path = self.path + '/gts'
#         self.gt_list = os.listdir(self.gt_path)

#         self.mask_path = self.path + '/mask'
#         self.mask_list = os.listdir(self.mask_path)

#         self.ImgTrans = ImageTransform()

#     def __getitem__(self, index):
#         image_path = self.image_list[index]

#         img = Image.open(self.image_path + '/' + image_path).convert('RGB')
#         try:
#             gt = Image.open(self.gt_path + '/' + image_path[:-4] + '.jpg').convert('RGB')
#         except:
#             gt = Image.open(self.gt_path + '/' + image_path[:-4] + '.png').convert('RGB')


#         inputImage = self.ImgTrans(img)
#         groundTruth = self.ImgTrans(gt)

#         return inputImage, groundTruth, image_path

#     # 200张做验证
#     def __len__(self):
#         return 200
# 验证数据集
class ValidDataSetDebug(paddle.io.Dataset):
    def __init__(self, file_path=None):
        super().__init__()

        self.path = file_path
        data_dirs = os.listdir(file_path+'/classone')
        i = random.randint(0, len(data_dirs)-1)

        data_path = self.path +'/classone/'+ data_dirs[i]+ '/images'
        data_path2 = self.path +'/dehw_train_dataset'+ '/images'
        self.image_list = [os.path.join(data_path, img_path) for img_path in os.listdir(data_path)]
        self.image_list += [os.path.join(data_path2, img_path) for img_path in os.listdir(data_path2)]
        # print(([os.path.join(data_path2, img_path) for img_path in os.listdir(data_path2)]))
        print("number of images:", len(self.image_list))
        # self.image_list = os.listdir(self.path + '/images')

        # self.path = file_path
        # self.image_list = os.listdir(self.path + '/images')
        # self.image_path = self.path + '/images'

        # self.gt_path = self.path + '/gts'
        # self.gt_list = os.listdir(self.gt_path)

        # self.mask_path = self.path + '/mask'
        # self.mask_list = os.listdir(self.mask_path)
        self.ImgTrans = ImageTransform()

    def __getitem__(self, index):
        index = random.randint(0, len(self.image_list))
        image_path = self.image_list[index]
        img = Image.open(image_path).convert('RGB')
        try:
            gt = Image.open(image_path.replace("images","gts")[:-4] + '.jpg').convert('RGB')
        except:
            gt = Image.open(image_path.replace("images","gts")[:-4] + '.png').convert('RGB')

        inputImage = self.ImgTrans(img)
        groundTruth = self.ImgTrans(gt)

        return inputImage, groundTruth,image_path
    # 200张做验证
    def __len__(self):
        return 30
