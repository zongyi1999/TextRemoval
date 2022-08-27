import os
import sys
import glob
import json
import cv2


import paddle
import paddle.nn as nn
import paddle.nn.functional as F
# 加载Erasenet改
# from models.swin_gan_cascade import STRnet2_change
from models.swin_gan import STRnet2_change
from models.sa_gan import STRnet2
import utils
from paddle.vision.transforms import Compose, ToTensor
from PIL import Image

# 加载我们训练到的最好的模型
netG = STRnet2_change()
netG2 = STRnet2()
weights1 = paddle.load('/media/backup/competition/train_models_swin_erasenet_finetune/STE_1_39.4660.pdparams') #/media/backup/competition/average_model.pdparams')
weights2 = paddle.load('/media/backup/competition/STE_str_best.pdparams')
netG.load_dict(weights1)
netG.eval()
netG2.load_dict(weights2)
netG2.eval()



def ImageTransform():
    return Compose([ToTensor(), ])  


ImgTrans = ImageTransform()


def process(src_image_dir, save_dir):
    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))
    for image_path in image_paths:
        pad = 50
        # do something
        img = Image.open(image_path)
        inputImage = paddle.to_tensor([ImgTrans(img)])

        _, _, h, w = inputImage.shape
        rh, rw = h, w
        step = 412
        pad_h = int(step - h)/2+1 if h+2*pad < step else pad
        pad_w = int(step - w)/2+1 if w+2*pad < step else pad
        
        m = nn.Pad2D(max(pad_h,pad_w), mode='reflect')
        # print(h,w)
        imgs = m(inputImage)
        _, _, h, w = imgs.shape
        res = paddle.zeros_like(imgs)
        res_mm = paddle.zeros_like(imgs)
        clip_list = []
        for i in range(0, h, step):
            for j in range(0, w, step):
                if h - i < step+ 2 * pad:
                    i = h - (step+ 2 * pad)
                if w - j < step+ 2 * pad:
                    j = w - (step+ 2 * pad)
                clip = imgs[:, :, i:i + step+ 2 * pad, j:j + step+ 2 * pad]
                clip_list.append(clip)

        clips_tensor = paddle.concat(clip_list)
        # print(clips_tensor.shape)
        clip_count = clips_tensor.shape[0]
        clip_num= 2
        g_images_list = []
        for i in range(0, clips_tensor.shape[0], clip_num):
            if i+clip_num>clip_count:
                # i = clip_num-clip_count
                clip_input = clips_tensor[i:,:,:]
            else:
                clip_input = clips_tensor[i:i+clip_num,:,:]
            clip = clip_input.cuda()
            # print(clip.shape)
            with paddle.no_grad():
                if h==w or h>2000 or w>2000:
                    g_images_clip, mm = netG2(clip)
                    print(image_path)
                else:
                    g_images_clip, mm = netG(clip)
                # g_images_clip, mm = netG(clip)
            g_images_clip = g_images_clip.cpu()
            g_images_list.append(g_images_clip)
        g_images_list = paddle.concat(g_images_list)
        count = 0
        for i in range(0, h, step):
            for j in range(0, w, step):
                if h - i < step+ 2 * pad:
                    i = h - (step+ 2 * pad)
                if w - j < step+ 2 * pad:
                    j = w - (step+ 2 * pad)
                # print(res[:, :, i+pad:i + step+pad, j+pad:j + step+pad].shape, g_images_list[count][:,pad:-pad, pad:-pad].shape)
                res[:, :, i+pad:i + step+pad, j+pad:j + step+pad] = g_images_list[count][:,pad:-pad, pad:-pad]
                # res[:, :, i:i + step, j:j + step] = g_images_list
                count+=1
        del g_images_list, g_images_clip, mm, clip
                # clip = clip.cuda()
                # with paddle.no_grad():
                #     print(clip.shape)
                #     g_images_clip, mm = netG(clip)
                # g_images_clip = g_images_clip.cpu()
                # mm = mm.cpu()
                # clip = clip.cpu()
                # g_image_clip_with_mask = g_images_clip * mm + clip * (1 - mm)
                # res[:, :, i+pad:i + step+pad, j+pad:j + step+pad] = g_image_clip_with_mask[:,:,pad:-pad, pad:-pad]
                # # res[:, :, i+pad:i + step+pad, j+pad:j + step+pad] = g_image_clip_with_mask[:,pad:-pad, pad:-pad]
                # del g_image_clip_with_mask, g_images_clip, mm, clip
        # res = res[:, :, :rh, :rw]
        res = res[:, :, pad:-pad, pad:-pad]
        output = utils.pd_tensor2img(res)
        # 保存结果图片
        save_path = os.path.join(save_dir, os.path.basename(image_path).replace(".jpg", ".png"))
        cv2.imwrite(save_path, output)
        del output, res

if __name__ == "__main__":
    assert len(sys.argv) == 3

    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    process(src_image_dir, save_dir)