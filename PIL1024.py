from PIL import Image
import os
import numpy as np
import PIL
import PIL.Image
import scipy
import scipy.ndimage
import cv2
from matplotlib import pyplot as plt
from spiga.inference.config import ModelConfig
from spiga.inference.framework import SPIGAFramework



face_predictor = SPIGAFramework(ModelConfig('wflw'), gpus=[0])

def get_landmark(img, p):

    """get landmark with dlib
    :return: np.array shape=(98, 2)
    """
    # detector = dlib.get_frontal_face_detector()
    # dets = detector(img, 1)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    dets = face_cascade.detectMultiScale(img,scaleFactor = 1.1, minNeighbors = 5)

    for k, det in enumerate(dets):
        feature = p.inference(img, [det])


    #     shape = predictor(img, d)
    #
    # t = list(shape.parts())
    # a = []
    # for tt in t:
    #     a.append([tt.x, tt.y])
    lm = np.array(feature['landmarks'][0])
    return lm

def align_face(img, predictor, output_size):
    """
    :param filepath: str
    :return: PIL Image
    """
    lm = get_landmark(img, predictor)

    # 新加的看看landmark效果
    # for b in lm:
    #     cv2.circle(img, (int(b[0]), int(b[1])), 2, (255, 255, 255), 4)
    #
    # fig, axs = plt.subplots(1, 1, figsize=(12, 12))
    # axs.imshow(img)
    # print('aaa')
    # 结束看效果

    lm_eye_left = lm[60: 68]  # left-clockwise
    # lm_eye_left2 = lm[96:97]
    # lm_eye_left = np.concatenate([lm_eye_left1,lm_eye_left2],axis=0)
    lm_eye_right = lm[68: 76]  # left-clockwise
    # lm_eye_right2 = lm[97:98]
    # lm_eye_right = np.concatenate([lm_eye_right1,lm_eye_right2],axis=0)
    lm_mouth_outer = lm[76: 88]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    # img = PIL.Image.open(filepath)
    img = PIL.Image.fromarray(img)

    transform_size = output_size
    enable_padding = True

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
        img = img.resize((1024, 1024), PIL.Image.ANTIALIAS)


    # Return aligned image.
    return img

imgdir = './data/unaligned'

# a0 = cv2.imread("./data/examples/aligned.png")
# print(a0)


img_list = sorted(os.listdir(imgdir))
for image_name in img_list:
    img_path = os.path.join(imgdir, image_name)

    # 打开图像并转换为RGBA格式
    img = Image.open(img_path).convert("RGBA")

    # 获取图像数据
    pixel_data = img.load()
    # 创建一个新的RGB像素数组
    rgb_data = []
    # 遍历原始图像的所有像素，并从中提取R、G、B值
    for y in range(img.size[1]):
        for x in range(img.size[0]):
            r, g, b, a = pixel_data[x, y]
            rgb_data.append((r, g, b))
    # 创建新的RGB图像
    new_img = Image.new("RGB", img.size, color=0)
    new_img.putdata(rgb_data)

    fig,axs = plt.subplots(1,1,figsize=(12,12))
    # axs.imshow(torchvision.utils.make_grid(xx)[0])
    axs.imshow(new_img)
    # 保存新图像
    # new_img.save("./test_rgb.png")
    new_img = np.array(new_img)

    # starting ... ...
    original_image = np.array(align_face(new_img, face_predictor, 1024))

    fig, axs = plt.subplots(1, 1, figsize=(12, 12))
    # axs.imshow(torchvision.utils.make_grid(xx)[0])
    axs.imshow(original_image)
    print('aaa')




    im = Image.fromarray(original_image)
    im.save("./data/examples/filename.png")





