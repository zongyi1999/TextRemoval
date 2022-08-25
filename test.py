import cv2
from PIL import Image
import numpy as np
input_img = cv2.imread("/media/backup/competition/data/classone/03/images/1a0a003b6c7c189ecbfd9c3c8fb99396.jpg")
gt = cv2.imread("/media/backup/competition/data/classone/03/gts/1a0a003b6c7c189ecbfd9c3c8fb99396.jpg")
# mask = cv2.imread("/media/backup/competition/data/classone/03/mask/1a0a003b6c7c189ecbfd9c3c8fb99396.jpg")
input_img= Image.open("/media/backup/competition/data/classone/03/images/1a0a003b6c7c189ecbfd9c3c8fb99396.jpg").convert('RGB')
gt = Image.open("/media/backup/competition/data/classone/03/gts/1a0a003b6c7c189ecbfd9c3c8fb99396.jpg")
print(np.array(input_img).shape)
def cal_mask(img, gt):
    kernel = np.ones((2, 2), np.uint8)
    # mask = cv2.erode(np.uint8(mask),  kernel, iterations=2)
    # threshold = 25
    threshold = 10
    diff_image = np.abs(img.astype(np.float32) - gt.astype(np.float32))
    mean_image = np.mean(diff_image, axis=-1)
    print(mean_image.shape)
    mask = np.greater(mean_image, threshold).astype(np.uint8)
    mask = (1 - mask) * 255
    mask = cv2.erode(np.uint8(mask),  kernel, iterations=1)
    return np.expand_dims(np.uint8(mask),axis=2).repeat(3,axis=2)
mask = cal_mask(np.array(input_img), np.array(gt))
print(mask.shape)
out = input_img*(mask/255) + gt*(1-(mask/255))

print(np.unique(mask/255))
cv2.imwrite("./test_gt.jpg", out)