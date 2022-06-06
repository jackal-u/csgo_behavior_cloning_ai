import re
import random
import time
from multiprocessing.dummy import freeze_support
import cv2
from network import Net
import argparse
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torch.nn as nn
from torchvision import transforms
from PIL import Image



def get_frame_from_mp4(path, skip_frames_num, show_frames_num = False): # , size=(200, 150)
    """
    读取图像并做图像变换
    :param path:
    :param skip_frames_num:
    :return:
    """

    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    frames = []
    # Preprocess image
    tfms = transforms.Compose([transforms.Resize(250), transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    # torch.Size([3, 224, 298])
    m = 0
    std =0
    while success:
        success, image = vidcap.read()
        if image is None:
            continue

        m = + image.mean()/255
        std =+ image.std()/255

        image = tfms(Image.fromarray(image))
        image = transforms.functional.adjust_contrast(image, contrast_factor=1.8)
        # image = cv2.resize(image, size)
        frames.append(image)
        #print(image.shape)
    print(m/len(frames), std / len(frames))
    if show_frames_num:
        print(len(frames))

    frames = frames[skip_frames_num-1:]
    return frames


path = r'./data/g151-c-20220402212855368210429_de_dust2_round1_t_tick_1159_5297_player_76561198355750091.mp4'
fs = get_frame_from_mp4(path, 360,True)
tran = transforms.ToPILImage()

print(type(tran(fs[12])))
tran(fs[12]).show()