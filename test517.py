
import copy

import cv2
import numpy as np
from PIL import Image
from imageio import imwrite, mimwrite
import torch
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from matplotlib import pyplot as plt
import sys

from spiga000.inference.config import ModelConfig
from spiga000.inference.framework import SPIGAFramework
from spiga000.demo.visualize.plotter import Plotter

def get_landmark(img, p):
    """get landmark with dlib
    :return: np.array shape=(98, 2)
    """
    feature = p.inference(img)
    # lm = np.array(feature['landmarks'][0])
    lm = feature['landmarks'][0]
    return lm

# img = cv2.imread('data/examples/aligned.png') # np [1024,1024,3]
img = cv2.imread('data/examples/aligned.png') # np [1024,1024,3]
img = cv2.resize(img,(512, 512))

b,g,r = cv2.split(img)
img = cv2.merge([r,g,b])

fig, axs = plt.subplots(1, 1, figsize=(12, 12))
axs.imshow(img)


img_gen = torch.tensor(img,dtype=torch.float).cuda() # tensor [1024,1024,3]

face_predictor = SPIGAFramework(ModelConfig('wflw'), gpus=[0])

ldm_ref = get_landmark(img_gen, face_predictor)
landmarkss = np.array(ldm_ref.cpu().detach()) # np [98,2]
# landmarkss = ldm_ref.cpu().detach().numpy()

canvas = copy.deepcopy(img) # np:[1024,1024,3]

ploter = Plotter()
canvas = ploter.landmarks.draw_landmarks(canvas, landmarkss)

fig, axs = plt.subplots(1, 1, figsize=(12, 12))
axs.imshow(canvas)
print('aaa')