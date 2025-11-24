from PIL import Image
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os
import numpy as np
import PIL
import PIL.Image
import scipy
import scipy.ndimage
import cv2
from matplotlib import pyplot as plt

from spiga000.inference.config import ModelConfig
from spiga000.inference.framework import SPIGAFramework

# transform = transforms.Compose([transforms.ToTensor(),transforms.Resize(256)])
transform = transforms.Compose([transforms.ToTensor()])

import numpy as np
import cv2
img = np.zeros((200,200,3),dtype=np.uint8)
cv2.circle(img,(60,60),30,(0,0,255),4)
cv2.imshow('img',img)
print('aaa')



def get_landmark(img, p):
    """get landmark with dlib
    :return: np.array shape=(98, 2)
    """
    feature = p.inference(img)
    # lm = np.array(feature['landmarks'][0])
    lm = feature['landmarks'][0]

    return lm


face_predictor = SPIGAFramework(ModelConfig('wflw'), gpus=[0])


imgdir = './data/unaligned'

img_list = sorted(os.listdir(imgdir))
for image_name in img_list:
    img_path = os.path.join(imgdir, image_name)

    im = cv2.imread(img_path)
    cv2.imshow('img', im)

    cv2.circle(im, (50, 50), 2, (255, 255, 255), -1)
    cv2.imshow('img', im)
    print('aaa')


    fig, axs = plt.subplots(1, 1, figsize=(12, 12))
    # axs.imshow(torchvision.utils.make_grid(xx)[0])
    axs.imshow(im)


    # 打开图像并转换为RGBA格式
    img = Image.open(img_path)
    img_np = np.array(img)
    cv2.circle(img_np, (50, 50), 2, (255, 255, 255), 4)
    cv2.imshow('img', img_np)
    print('aaa')



    img = transform(img)  # 3,1024,1024
    #把tensor图像 缩小为指定尺寸
    img_tensor = F.interpolate(img.unsqueeze(0),size=(256,256),mode='bilinear',align_corners=False)
    # resized_img = img_tensor.squeeze(0).permute(1,2,0)
    # resized_img = resized_img.clamp(0,1)
    # resized_img = resized_img.numpy()
    # plt.imshow(resized_img)
    # print('aaa')
    img_tensor = img_tensor.cuda()
    img_tensor.requires_grad = True

    img_tensor_new = img_tensor.clone().squeeze(0).permute(1, 2, 0)
    img_tensor_np = img_tensor_new.detach().cpu().numpy().copy() # 要加.copy()，不共享以前的
    img_tensor_np = (img_tensor_np*255).astype(np.uint8)


    ldm_ref = get_landmark(img_tensor, face_predictor)

    ldm_ref_np = ldm_ref.clone().detach().cpu().numpy().astype(np.uint8)


    for b in ldm_ref_np:
        cv2.circle(img_np, (int(b[0]), int(b[1])), 2, (255, 255, 255), -1)
        cv2.circle(img_tensor_np, (int(b[0]), int(b[1])), 2, (255, 255, 255), -1)


    cv2.imshow("bg", img_tensor_np)
    cv2.waitKey(0)







#
# REFERENCE_FACIAL_POINTS = [        # default reference facial points for crop_size = (112, 112); should adjust REFERENCE_FACIAL_POINTS accordingly for other crop_size
#     [30.29459953+8,  51.69630051],
#     [65.53179932+8,  51.50139999],
#     [48.02519989+8,  71.73660278],
#     [33.54930115+8,  92.3655014],
#     [62.72990036+8,  92.20410156]
# ]

#bg = np.zeros([112,112,3])



