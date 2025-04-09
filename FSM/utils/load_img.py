# -*- encoding: utf-8 -*-
#Time        :2020/12/13 19:25:28
#Author      :Chen
#FileName    :load_img.py
#Version     :1.0

import PIL.Image as Image
import numpy as np
import torchvision.transforms as transforms

img_size = 512

def load_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((512, 512))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    return img


def load_img_Pro(img_path):
    img = Image.open(img_path)
    img = img.resize((512, 512))

    # Convert to RGB if the image is not in RGB format
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_npy = np.array(img)

    # Check if the image has the expected number of channels
    if img_npy.ndim != 3 or img_npy.shape[2] != 3:
        raise ValueError(f"Image at {img_path} does not have 3 channels. Shape: {img_npy.shape}")

    img_npy = img_npy.transpose(2, 0, 1).astype(np.float32)

    for i in range(img_npy.shape[0]):
        img_npy[i] = (img_npy[i] - img_npy[i].mean()) / img_npy[i].std()

    return img_npy


def show_img(img):
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    img.show()