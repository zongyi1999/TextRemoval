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
from dataset.data_loader import TrainDataSet, ValidDataSet
# 自定义的loss函数，包含mask的损失和image的损失
from loss.Loss import LossWithGAN_STE, LossWithSwin

# 使用SwinT增强的Erasenet
from models.swin_gan_ori import STRnet2_change
from models.sa_gan import STRnet2
from models.str import  STRAIDR
# 其他工具
import utils
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
# %matplotlib inline
# paddle.disable_static()

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
    'traindataRoot': 'data',
    'validdataRoot': 'data',   # 因为数据集量大，且分布一致，就直接取训练集中数据作为验证了。别问，问就是懒
    'pretrained': "/media/backup/competition/model_best_document.pdparams", 
    'num_epochs': 100,
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

traindataRoot = CONFIG['traindataRoot']
validdataRoot = CONFIG['validdataRoot']

# 创建数据集容器

ValidData = ValidDataSet(file_path=validdataRoot)
ValidDataLoader = DataLoader(ValidData, batch_size=1, shuffle=True, num_workers=0, drop_last=True)
netG = STRnet2_change()
# netG = STRnet2()
# netG = STRAIDR(num_c=96)

if CONFIG['pretrained'] is not None:
    print('loaded ')
    weights = paddle.load(CONFIG['pretrained'])
    netG.load_dict(weights)

# 开始直接上大火
lr = 2e-4
G_optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=netG.parameters())
scaler = paddle.amp.GradScaler(init_loss_scaling=1024)


loss_function = LossWithGAN_STE()


print('OK!')
num_epochs = CONFIG['num_epochs']
best_psnr = 0
iters = 0
for epoch_id in range(1, num_epochs + 1):





    TrainData = TrainDataSet(training=True, file_path=traindataRoot)
    TrainDataLoader = DataLoader(TrainData, batch_size=batchSize, shuffle=True,
                                num_workers=8, drop_last=True)
    netG.train()

    if epoch_id % 15 == 0: #8
        lr /= 10
        G_optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=netG.parameters())

    for k, (imgs, gts, masks) in enumerate(TrainDataLoader):
        iters += 1
        fake_images, mm = netG(imgs)
        G_loss = loss_function(masks, fake_images, mm, gts)
        G_loss = G_loss.sum()
        # epoch1, iters100, loss:0.38920, lr:0.002
        # epoch1, iters200, loss:0.36512, lr:0.002
        # epoch1, iters300, loss:0.36484, lr:0.002
        # epoch1, iters400, loss:0.39128, lr:0.002
        # epoch1, iters500, loss:0.36669, lr:0.002
        # epoch1, iters600, loss:0.39008, lr:0.002
        # epoch1, iters700, loss:0.37363, lr:0.002
        # 逻辑1：创建 AMP-O1 auto_cast 环境，开启自动混合精度训练，将 add 算子添加到自定义白名单中(custom_white_list)，
        # 因此前向计算过程中该算子将采用 float16 数据类型计算
        # with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}, level='O1'):
        #     # output = model(data) # 前向计算（9层Linear网络，每层由matmul、add算子组成）
        #     # loss = mse(output, label) # loss计算
        #     fake_images, mm = netG(imgs)
        #     G_loss = loss_function(masks, fake_images, mm, gts)
        #     G_loss = G_loss.sum()

        # # 逻辑2：使用 GradScaler 完成 loss 的缩放，用缩放后的 loss 进行反向传播
        # scaled = scaler.scale(G_loss) # loss缩放，乘以系数loss_scaling
        # scaled.backward()           # 反向传播
        # scaler.step(optimizer)      # 更新参数（参数梯度先除系数loss_scaling再更新参数）
        # scaler.update()             # 基于动态loss_scaling策略更新loss_scaling系数
        # 后向传播，更新参数的过程
        G_loss.backward()
        # 最小化loss,更新参数
        G_optimizer.step()
        # 清除梯度
        G_optimizer.clear_grad()
        # 打印训练信息
        if iters % 100 == 0:
            print('epoch{}, iters{}, loss:{:.5f}, lr:{}'.format(
                epoch_id, iters, G_loss.item(), G_optimizer.get_lr()
            ))
            log.add_scalar(tag="train_loss", step=iters, value=G_loss.item())
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
        mm_out = paddle.zeros_like(imgs)
        mm_in = paddle.zeros_like(imgs)
        for i in range(0, h, step):
            for j in range(0, w, step):
                if h - i < step:
                    i = h - step
                if w - j < step:
                    j = w - step
                clip = imgs[:, :, i:i + step, j:j + step]
                clip = clip.cuda()
                with paddle.no_grad():
                    g_images_clip, mm = netG(clip)
                g_images_clip = g_images_clip.cpu()
                mm = mm.cpu()
                clip = clip.cpu()
                mm_in[:, :, i:i + step, j:j + step] = mm
                g_image_clip_with_mask = clip * (1 - mm) + g_images_clip * mm
                res[:, :, i:i + step, j:j + step] = g_image_clip_with_mask
        res = res[:, :, :rh, :rw]
        # 改变通道
        output = utils.pd_tensor2img(res)
        target = utils.pd_tensor2img(gt)
        mm_in = utils.pd_tensor2img(mm_in)
        psnr_value = psnr(output, target)
        print('psnr: ', psnr_value)
        del res
        del gt
        del target
        del output

        val_psnr += psnr_value
    ave_psnr = val_psnr / (index + 1)
    print('epoch:{}, psnr:{}'.format(epoch_id, ave_psnr))
    log.add_scalar(tag="valid_psnr", step=epoch_id, value=ave_psnr)
    paddle.save(netG.state_dict(), CONFIG['modelsSavePath'] +
                '/STE_{}_{:.4f}.pdparams'.format(epoch_id, ave_psnr
                ))
    if ave_psnr > best_psnr:
        best_psnr = ave_psnr
        paddle.save(netG.state_dict(), CONFIG['modelsSavePath'] + '/STE_best.pdparams')
