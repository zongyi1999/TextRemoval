# 生成mask的函数如下
from datetime import datetime
import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import math
# 可视化
from visualdl import LogWriter
# paddle包
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
# 使用SwinT增强的Erasenet
from models.swin_gan_ori import STRnet2_change
from models.sa_gan import STRnet2
from models.str import  STRAIDR
from paddle.optimizer.lr import CosineAnnealingDecay, MultiStepDecay, ExponentialDecay

# 自定义的loss函数，包含mask的损失和image的损失
from Loss import LossWithGAN_STE, LossWithSwin
from data_loader import TrainValidDataset
# 其他工具
import utils

# 计算psnr
log = LogWriter('log')
def psnr(img1, img2):
   mse = np.mean((img1/1.0 - img2/1.0) ** 2 )
   if mse < 1.0e-10:
      return 100
   return 10 * math.log10(255.0**2/mse)


# 训练配置字典
CONFIG = {
    'modelsSavePath': 'train_models_document_STR',
    'batchSize': 7,  # 模型大，batch_size调小一点防崩，拉满显存但刚好不超，就是炼丹仙人~
    'dataRoot': '/home/shb/experiment/OCR/dataset',
    'pretrained': "train_models_swin_erasenet/best_submit_model.pdparams", 
    'num_epochs': 500,
    'seed': 8888  # 就是爱你！~
}


# 设置随机种子
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
paddle.seed(CONFIG['seed'])
# noinspection PyProtectedMember
paddle.framework.random._manual_program_seed(CONFIG['seed'])


batchSize = CONFIG['batchSize']
if not os.path.exists(CONFIG['modelsSavePath']):
    os.makedirs(CONFIG['modelsSavePath'])

dataRoot = CONFIG['dataRoot']

# 创建数据集容器

TrainData, ValidData = TrainValidDataset(file_path=dataRoot, ratio=0.005).getData()
TrainDataLoader = DataLoader(TrainData, batch_size=batchSize, shuffle=True, num_workers=4, drop_last=True)
ValidDataLoader = DataLoader(ValidData, batch_size=1, shuffle=True, num_workers=0, drop_last=True)

netG = STRnet2_change()

if CONFIG['pretrained'] is not None:
    print('loaded ')
    weights = paddle.load(CONFIG['pretrained'])
    netG.load_dict(weights)

# 开始直接上大火
lr = 1e-04
scheduler = paddle.optimizer.lr.StepDecay(learning_rate=lr, step_size=30000, gamma=0.5, verbose=False)
G_optimizer = paddle.optimizer.Adam(scheduler, parameters=netG.parameters(), weight_decay=0.0)

loss_function = LossWithGAN_STE()


print('OK!')
num_epochs = CONFIG['num_epochs']
best_psnr = 0
iters = 0
for epoch_id in range(1, num_epochs + 1):
    netG.train()
    del TrainDataLoader, ValidDataLoader
    TrainDataLoader = DataLoader(TrainData, batch_size=batchSize, shuffle=True, num_workers=4, drop_last=True)
    ValidDataLoader = DataLoader(ValidData, batch_size=1, shuffle=True, num_workers=0, drop_last=True)


    for k, (imgs, gts, masks) in enumerate(TrainDataLoader):
        iters += 1
        fake_images, mm = netG(imgs)
        G_loss = loss_function(masks, fake_images, mm, gts)
        G_loss = G_loss.sum()

        G_loss.backward()
        # 最小化loss,更新参数
        if iters % 4 == 0:   # batch_size*4
            G_optimizer.step()
            # 清除梯度
            G_optimizer.clear_grad()
            scheduler.step()
        # 打印训练信息
        if iters % 100 == 0:
            print(datetime.now())
            print('epoch{}, iters{}, loss:{:.5f}, lr:{}'.format(
                epoch_id, iters, G_loss.item(), G_optimizer.get_lr()
            ))
            log.add_scalar(tag="train_loss", step=iters, value=G_loss.item())
        if iters % 2000 == 0:
            print(datetime.now())
            # 对模型进行评价并保存
            netG.eval()
            val_psnr = 0

            # noinspection PyAssignmentToLoopOrWithParameter
            for index, (imgs, gt) in enumerate(ValidDataLoader):
                _, _, h, w = imgs.shape
                rh, rw = h, w
                step = 512
                pad_h = step - h if h < step else 0
                pad_w = step - w if w < step else 0
                m = nn.Pad2D((0, pad_w, 0, pad_h))
                imgs = m(imgs)
                _, _, h, w = imgs.shape
                res = paddle.zeros_like(imgs)
                for i in range(0, h, step):
                    for j in range(0, w, step):
                        if h - i < step:
                            i = h - step
                        if w - j < step:
                            j = w - step
                        clip = imgs[:, :, i:i + step, j:j + step]
                        clip = clip.cuda()
                        with paddle.no_grad():
                            g_images_clip, _ = netG(clip)
                        g_images_clip = g_images_clip.cpu()
                        g_image_clip_with_mask = g_images_clip 
                        res[:, :, i:i + step, j:j + step] = g_image_clip_with_mask
                res = res[:, :, :rh, :rw]
                # 改变通道
                output = utils.pd_tensor2img(res)
                target = utils.pd_tensor2img(gt)
                # mm_in = utils.pd_tensor2img(mm_in)
                psnr_value = psnr(output, target)
                # print('psnr: ', psnr_value)
                del res
                del gt
                del target
                del output

                val_psnr += psnr_value
            ave_psnr = val_psnr / (index + 1)
            print('epoch:{}, psnr:{}'.format(epoch_id, ave_psnr))
            log.add_scalar(tag="valid_psnr", step=epoch_id, value=ave_psnr)
            paddle.save(netG.state_dict(), CONFIG['modelsSavePath'] +
                        '/STE_{}_{}_{:.4f}.pdparams'.format(epoch_id, iters, ave_psnr
                        ))
            if ave_psnr > best_psnr:
                best_psnr = ave_psnr
                paddle.save(netG.state_dict(), CONFIG['modelsSavePath'] + '/STE_best.pdparams')
