import os
import sys
import glob
from cv2 import imwrite

import paddle
import paddle.nn as nn
from models.swin_gan import STRnet2_change
from models.sa_gan import STRnet2
from paddle.vision.models import resnet18

from paddle.vision.transforms import ToTensor, Compose, Resize
from PIL import Image
import numpy as np


def pd_tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    img = tensor.squeeze().cpu().numpy()
    img = img.clip(min_max[0], min_max[1])
    img = img * 255.0
    img = np.transpose(img, (1, 2, 0))
    img = img.round()
    img = img[:, :, ::-1]
    return img.astype(out_type)


# 加载我们训练到的最好的模型
netG = STRnet2_change()
netG2 = STRnet2()
netC = resnet18(pretrained=False, num_classes=2, with_pool=True)   # paper out 0   doc out1
weights1 = paddle.load('model/STE_35_596000_43.8973.pdparams')
weights2 = paddle.load('model/STE_str_best.pdparams')
weights3 = paddle.load('model/Classify_STE_Best.pdparams')
netG.load_dict(weights1)
netG.eval()
netG2.load_dict(weights2)
netG2.eval()
netC.load_dict(weights3)
netC.eval()

nets_list = [netG2, netG]
def getClass(imgs):
    with paddle.no_grad():
        out = netC(imgs)
    re = nets_list[out.argmax(1)]
    del out, imgs
    return re

Trans = ToTensor()
CTrans = Compose([Resize((512,512)),ToTensor()])

def process(src_image_dir, save_dir):
    image_paths = glob.glob(os.path.join(src_image_dir, "*.jpg"))
    for image_path in image_paths:
        img = Image.open(image_path)
        inputImage = paddle.to_tensor([Trans(img)])

        net = getClass(paddle.to_tensor([CTrans(img)]))
        
        pad = 50
        step = 412
        m = nn.Pad2D(pad)
        imgs = m(inputImage)
        _, _, h, w = imgs.shape
        res = paddle.zeros_like(imgs)
        
        clip_list = []
        for i in range(0, h, step):
            for j in range(0, w, step):
                if h - i < 512:
                    i = h - 512
                if w - j < 512:
                    j = w - 512
                clip = imgs[:, :, i:i + 512, j:j + 512]
                clip_list.append(clip)

        clips_tensor = paddle.concat(clip_list)

        clip_count = clips_tensor.shape[0]
        clip_num= 8
        g_images_list = []
        for i in range(0, clip_count, clip_num):
            if i+clip_num>clip_count:
                clip_input = clips_tensor[i:,:,:]
            else:
                clip_input = clips_tensor[i:i+clip_num,:,:]
            clip = clip_input.cuda()

            with paddle.no_grad():
                g_images_clip= net(clip)[0]
                g_images_clip += paddle.flip(net(paddle.flip(clip, axis=[3]))[0],axis=[3])
                g_images_clip /= 2
            g_images_clip = g_images_clip.cpu()
            g_images_list.append(g_images_clip)


        g_images_list = paddle.concat(g_images_list)
        count = 0
        for i in range(0, h, step):
            for j in range(0, w, step):
                if h - i < 512:
                    i = h - 512
                if w - j < 512:
                    j = w - 512
                res[:, :, i+pad:i + step+pad, j+pad:j + step+pad] = g_images_list[count][:,pad:-pad, pad:-pad]                
                count+=1
        del g_images_list, g_images_clip, clip, imgs, clips_tensor, clip_input

        res = res[:, :, pad:-pad, pad:-pad]
        output = pd_tensor2img(res)
        # 保存结果图片
        save_path = os.path.join(save_dir, os.path.basename(image_path)[:-4] + ".png")
        imwrite(save_path, output)
        del output, res

if __name__ == "__main__":
    assert len(sys.argv) == 3

    src_image_dir = sys.argv[1]
    save_dir = sys.argv[2]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    process(src_image_dir, save_dir)