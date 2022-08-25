# 生成mask的函数如下
import os
import random
from PIL import Image
import numpy as np
# 输入：水印图像路劲，原图路劲，保存的mask的路径
def generate_one_mask(image_path, gt_path, save_path):
    # 读取图像
    image = Image.open(image_path)
    gt = Image.open(gt_path)

    # 转成numpy数组格式
    image = 255 - np.array(image)[:, :, :3]
    gt = 255 - np.array(gt)[:, :, :3]

    # 设置阈值
    threshold = 15
    # 真实图片与手写图片做差，找出mask的位置
    diff_image = np.abs(image.astype(np.float32) - gt.astype(np.float32))
    mean_image = np.max(diff_image, axis=-1)

    # 将mask二值化，即0和255。
    mask = np.greater(mean_image, threshold).astype(np.uint8) * 255
    mask[mask < 2] = 0
    mask[mask >= 1] = 255
    mask = 255 - mask
    mask = np.clip(mask, 0, 255)

    # 保存
    mask = np.array([mask, mask, mask, mask])
    mask = mask.transpose(1, 2, 0)
    mask = Image.fromarray(mask[:, :, :3])
    mask.save(save_path)

    # 可视化
from visualdl import LogWriter

# paddle包
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
from dataset.data_loader import TrainDataSet, ValidDataSet, ValidDataSetDebug

# 自定义的loss函数，包含mask的损失和image的损失
from loss.Loss import LossWithGAN_STE, LossWithSwin

# 使用SwinT增强的Erasenet
from models.swin_gan_ori import STRnet2_change
# 其他工具
import utils
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
# %matplotlib inline
# paddle.disable_static()
import time
# 计算psnr
log = LogWriter('log')
def psnr(img1, img2):
   mse = np.mean((img1/1.0 - img2/1.0) ** 2 )
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(255.0**2/mse)


# 训练配置字典
CONFIG = {
    'modelsSavePath': 'train_models_swin_erasenet_finetune',
    'batchSize': 7,  # 模型大，batch_size调小一点防崩，拉满显存但刚好不超，就是炼丹仙人~
    'traindataRoot': 'data',
    'validdataRoot': 'dataset',   # 因为数据集量大，且分布一致，就直接取训练集中数据作为验证了。别问，问就是懒
    'pretrained': "/media/backup/competition/train_models_swin_erasenet_finetune/STE_7_39.9287.pdparams",#"/media/backup/competition/train_models_swin_erasenet_finetune/STE_12_38.1306.pdparams", #"/media/backup/competition/submit/model/STE_61_37.8539.pdparams", #None, #'/media/backup/competition/train_models_swin_erasenet/STE_100_37.4260.pdparams',
    'num_epochs': 100,
    'seed': 8888  # 就是爱你！~
}
# "/media/backup/competition/train_models_swin_erasenet_finetune/STE_1_39.4660.pdparams",#
# 设置随机种子
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
paddle.seed(CONFIG['seed'])
# noinspection PyProtectedMember
paddle.framework.random._manual_program_seed(CONFIG['seed'])


batchSize = CONFIG['batchSize']
if not os.path.exists(CONFIG['modelsSavePath']):
    os.makedirs(CONFIG['modelsSavePath'])

traindataRoot = CONFIG['traindataRoot']
validdataRoot = CONFIG['validdataRoot']

# 创建数据集容器

ValidData = ValidDataSetDebug(file_path=validdataRoot)
ValidDataLoader = DataLoader(ValidData, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

netG = STRnet2_change()


if CONFIG['pretrained'] is not None:
    print('loaded ')
    weights = paddle.load(CONFIG['pretrained'])
    netG.load_dict(weights)





print('OK!')
num_epochs = CONFIG['num_epochs']
best_psnr = 0
iters = 0


for epoch_id in range(1, num_epochs + 1):
    netG.eval()
    val_psnr = 0

    # noinspection PyAssignmentToLoopOrWithParameter
    for index, (imgs, gt, img_path) in enumerate(ValidDataLoader):
        start = time.time()
        _, _, h, w = imgs.shape
        rh, rw = h, w
        step = 512
        pad_h = step - h if h < step else 0
        pad_w = step - w if w < step else 0
        m = nn.Pad2D((0, pad_w, 0, pad_h))
        imgs = m(imgs)
        _, _, h, w = imgs.shape
        res = paddle.zeros_like(imgs)
        clip_list = []
        mm_out = paddle.zeros_like(imgs)
        mm_in = paddle.zeros_like(imgs)
        for i in range(0, h, step):
            for j in range(0, w, step):
                if h - i < step:
                    i = h - step
                if w - j < step:
                    j = w - step
                clip = imgs[:, :, i:i + step, j:j + step]
                clip_list.append(clip)
        # clips_tensor = paddle.concat(clip_list)
        # clip_count = clips_tensor.shape[0]
        # clip_num= 6
        # g_images_list = []
        # for i in range(0, clips_tensor.shape[0], clip_num):
        #     if i+clip_num>clip_count:
        #         # i = clip_num-clip_count
        #         clip_input = clips_tensor[i:,:,:]
        #     else:
        #         clip_input = clips_tensor[i:i+clip_num,:,:]
        #     clip = clip_input.cuda()
        #     with paddle.no_grad():
        #         g_images_clip, mm = netG(clip)
        #     g_images_clip = g_images_clip.cpu()
        #     g_images_list.append(g_images_clip)
        # g_images_list = paddle.concat(g_images_list)
        # count = 0
        # for i in range(0, h, step):
        #     for j in range(0, w, step):
        #         if h - i < step:
        #             i = h - step
        #         if w - j < step:
        #             j = w - step
        #         res[:, :, i:i + step, j:j + step] = g_images_list[count]
        #         count+=1
        # del g_images_list, g_images_clip, mm, clip

                clip = clip.cuda()
                with paddle.no_grad():
                    g_images_clip, mm = netG(clip)
                g_images_clip = g_images_clip.cpu()
                mm = mm.cpu()
                clip = clip.cpu()
                mm_in[:, :, i:i + step, j:j + step] = mm
                g_image_clip_with_mask =g_images_clip# * mm + clip * (1 - mm) 
                res[:, :, i:i + step, j:j + step] = g_image_clip_with_mask
        res = res[:, :, :rh, :rw]
        # 改变通道
        print(img_path)
        output = utils.pd_tensor2img(res)
        target = utils.pd_tensor2img(gt)
        mm_in = utils.pd_tensor2img(mm_in)
        psnr_value = psnr(output, target)
        print('psnr: ', psnr_value)
        end = time.time()
        print(end- start)



        del res
        del gt
        del target
        del output
        val_psnr += psnr_value
    ave_psnr = val_psnr / (index + 1)
    print('epoch:{}, psnr:{}'.format(epoch_id, ave_psnr))
    break

# ['dehw_train_00736.jpg']
# psnr:  31.806566776923248
# 6.6817872524261475
# ['dehw_train_01079.jpg']
# psnr:  31.072858983531468
# 7.079329252243042
# ['00a7326c05ed4f2f965b95dac95dd01d.jpg']
# psnr:  40.051386958186896
# 1.1438524723052979
# ['08407c372b76b02a7115b915e6e19334.jpg']
# psnr:  34.2499801585562
# 0.6940865516662598
# ['dehw_train_00678.jpg']
# psnr:  33.08743828470454
# 39.320598125457764
# ['09c74e5b497296edcdfb8cd615d960d1.jpg']