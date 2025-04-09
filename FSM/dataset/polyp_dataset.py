# -*- encoding: utf-8 -*-
#Time        :2020/12/19 21:18:08
#Author      :Chen
#FileName    :polyp_dataset.py
#Version     :2.0

import os
import torch
import os.path as osp
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .transform import *
from PIL import Image, ImageEnhance, ImageOps, ImageFile, ImageFilter
from utils.convert_csv_to_list import convert_labeled_list_Unlab, convert_labeled_list_lab
# Eq. (8): define compactness metric by quotient between area and perimeter of pseudo labels
def calculateCompact(image):
    #image = Image.open(image).convert('L')
    edge = image.filter(ImageFilter.FIND_EDGES)
    #edge.save('/home/cyang/SFDA/edge.png')
    edge = np.asarray(edge, np.float32)
    image = np.asarray(image, np.float32)
    image = image / 255
    edge = edge / 255
    S = np.sum(image)
    C = np.sum(edge)
    #print(C, S, edge[0])
    return np.nan_to_num(np.asarray(100 * S / C ** 2, np.float32))

def randomRotation(image, label):

    random_angle = np.random.randint(1, 60)
    return image.rotate(random_angle, Image.BICUBIC), label.rotate(random_angle, Image.NEAREST)

def randomColor(image):

    random_factor = np.random.randint(0, 31) / 10.  
    color_image = ImageEnhance.Color(image).enhance(random_factor)  
    random_factor = np.random.randint(10, 21) / 10. 
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  
    random_factor = np.random.randint(10, 21) / 10.  
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  
    random_factor = np.random.randint(0, 31) / 10. 
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor) 

def randomGaussian(image, mean=0.2, sigma=0.3):


    def gaussianNoisy(im, mean=0.2, sigma=0.3):

        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im


    img = np.asarray(image)
    #img.flags.writeable = True  
    width, height = img.shape[:2]
    img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
    img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
    img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
    img[:, :, 0] = img_r.reshape([width, height])
    img[:, :, 1] = img_g.reshape([width, height])
    img[:, :, 2] = img_b.reshape([width, height])
    return Image.fromarray(np.uint8(img))

# EndoScene Dataset
class Polyp(Dataset):
    def __init__(self, root, data_dir, mode='train', is_mirror=True, is_pseudo=None, max_iter=None, is_inverse=None, is_cbst=None):
        #super(Polyp, self).__init__()
        self.root = root
        self.data_dir = data_dir
        self.is_pseudo = is_pseudo
        self.is_mirror = is_mirror
        self.is_inverse = is_inverse
        self.is_cbst = is_cbst
        self.mode = mode
        self.imglist = []
        self.gtlist = []

        self.fine_list, self.mask_list = convert_labeled_list_lab([self.data_dir])
        self.files = []
        for i in range(len(self.fine_list)):
            if self.is_inverse:
                img_file = osp.join(self.root, self.fine_list[i])
            # else:
            #     img_file = osp.join(self.root, "images/%s" % name)
            # if self.is_pseudo:
            #     label_file = osp.join(self.root, "pseudo/%s" % name)
            # elif self.is_cbst:
            #     label_file = osp.join(self.root, "pseudo_cbst/%s" % name)
            # else:
            label_file = osp.join(self.root, self.mask_list[i])
            self.files.append({
                "image": img_file,
                "label": label_file,
                "name": self.fine_list[i].split('/')[-1]
            })
        #print(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]["image"]
        gt_path = self.files[index]["label"]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512), Image.BICUBIC)
        gt = Image.open(gt_path).convert('L')
        gt = gt.resize((512, 512), Image.NEAREST)
        if self.mode == 'train':
            #img = randomColor(img)
            img, gt = randomRotation(img, gt)
            #img = randomGaussian(img)

        img = np.asarray(img, np.float32)
        # img = img / 255
        gt = np.asarray(gt, np.float32)
        # gt = gt / 255

        # img = img[:, :, ::-1]  # change to BGR
        img = img.transpose((2, 0, 1))

        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            img = img[:, :, ::flip]
            gt = gt[:, ::flip]
        for i in range(img.shape[0]):
            img[i] = (img[i] - img[i].mean()) / img[i].std()
        mask = np.zeros_like(gt)
        mask[gt > 0] = 1
        mask[gt == 128] = 2
        gt = mask
        data = {'image': img.copy(), 'label': gt.copy()}
        # if self.transform:
        #     data = self.transform(data)

        return data, gt_path

    def __len__(self):
        return len(self.files)


class Pseudo_Polyp(Dataset):
    def __init__(self, root, data_dir, mode='train', is_mirror=True, is_pseudo=None, max_iter=None, is_inverse=None, is_cbst=None):
        #super(Polyp, self).__init__()
        self.root = root
        self.data_dir = data_dir
        self.is_pseudo = is_pseudo
        self.is_mirror = is_mirror
        self.is_inverse = is_inverse
        self.is_cbst = is_cbst
        self.mode = mode
        self.imglist = []
        self.gtlist = []

        self.img_ids = [i_id.strip() for i_id in open(self.data_dir)]
        if not max_iter == None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iter) / len(self.img_ids)))
        self.files = []
        for name in self.img_ids:
            if self.is_inverse:
                img_file = osp.join(self.root, "inversed/%s" % name)
            else:
                img_file = osp.join(self.root, "images/%s" % name)
            if self.is_pseudo:
                label_file = osp.join(self.root, "pseudo/%s" % name)
            elif self.is_cbst:
                label_file = osp.join(self.root, "pseudo_cbst/%s" % name)
            else:
                label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "image": img_file,
                "label": label_file,
                "name": name
            })
        #print(self.files)

    def __getitem__(self, index):
        img_path = self.files[index]["image"]
        gt_path = self.files[index]["label"]
        img = Image.open(img_path).convert('RGB')
        img = img.resize((512, 512), Image.BICUBIC)
        gt = Image.open(gt_path).convert('L')
        gt = gt.resize((512, 512), Image.NEAREST)
        weight = calculateCompact(gt)
        if self.mode == 'train':
            img = randomColor(img)
            img, gt = randomRotation(img, gt)
            #img = randomGaussian(img)

        img = np.asarray(img, np.float32)
        img = img / 255
        gt = np.asarray(gt, np.float32)
        gt = gt / 255

        img = img[:, :, ::-1]  # change to BGR
        img = img.transpose((2, 0, 1))

        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            img = img[:, :, ::flip]
            gt = gt[:, ::flip]

        data = {'image': img.copy(), 'label': gt.copy()}
        # if self.transform:
        #     data = self.transform(data)

        return data, weight, gt_path

    def __len__(self):
        return len(self.files)

class Contrast_Polyp(Dataset):
    def __init__(self, root, data_dir, mode='train', is_mirror=True, is_pseudo=None, max_iter=None, is_inverse=None):
        #super(Polyp, self).__init__()
        self.root = root
        self.data_dir = data_dir
        self.is_pseudo = is_pseudo
        self.is_mirror = is_mirror
        self.is_inverse = is_inverse
        self.mode = mode

        self.image_list, self.fine_list, self.pseudo_list = convert_labeled_list_Unlab([self.data_dir])
        self.files = []
        for i in range(len(self.image_list)):
            source_imgfile = osp.join(self.root, self.fine_list[i])

            target_imgfile = osp.join(self.root, self.image_list[i])
            if self.is_pseudo:
                label_file = osp.join(self.root, self.pseudo_list[i])
            # else:
            #     label_file = osp.join(self.root, "labels/%s" % name)
            self.files.append({
                "source": source_imgfile,
                "target": target_imgfile,
                "label": label_file,
                "name": self.image_list[i].split('/')[-1]
            })
        #print(self.files)

    def __getitem__(self, index):
        source_imgpath = self.files[index]["source"]
        target_imgpath = self.files[index]["target"]
        gt_path = self.files[index]["label"]
        source_img = Image.open(source_imgpath).convert('RGB')
        source_img = source_img.resize((512, 512), Image.BICUBIC)

        target_img = Image.open(target_imgpath).convert('RGB')
        target_img = target_img.resize((512, 512), Image.BICUBIC)

        gt = Image.open(gt_path).convert('L')
        gt = gt.resize((512, 512), Image.NEAREST)
        # weight = calculateCompact(gt)

        random_angle = np.random.randint(1, 360)
        source_img_rotate = source_img.rotate(random_angle, Image.BICUBIC)
        target_img_rotate = target_img.rotate(random_angle, Image.BICUBIC)
        gt_rotate = gt.rotate(random_angle, Image.NEAREST)

        weight = calculateCompact(gt)
        weight_rotate = calculateCompact(gt_rotate)

        source_img = np.asarray(source_img, np.float32)
        # source_img = source_img / 255

        source_img_rotate = np.asarray(source_img_rotate, np.float32)
        # source_img_rotate = source_img_rotate / 255

        target_img = np.asarray(target_img, np.float32)
        # target_img = target_img / 255

        target_img_rotate = np.asarray(target_img_rotate, np.float32)
        # target_img_rotate = target_img_rotate / 255


        gt = np.asarray(gt, np.float32)
        # gt = gt / 255

        gt_rotate = np.asarray(gt_rotate, np.float32)
        # gt_rotate = gt_rotate / 255

        # source_img = source_img[:, :, ::-1]  # change to BGR
        source_img = source_img.transpose((2, 0, 1))

        # source_img_rotate = source_img_rotate[:, :, ::-1]  # change to BGR
        source_img_rotate = source_img_rotate.transpose((2, 0, 1))

        # target_img = target_img[:, :, ::-1]  # change to BGR
        target_img = target_img.transpose((2, 0, 1))

        # target_img_rotate = target_img_rotate[:, :, ::-1]  # change to BGR
        target_img_rotate = target_img_rotate.transpose((2, 0, 1))

        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            source_img = source_img[:, :, ::flip]
            source_img_rotate = source_img_rotate[:, :, ::flip]
            target_img = target_img[:, :, ::flip]
            target_img_rotate = target_img_rotate[:, :, ::flip]
            gt = gt[:, ::flip]
            gt_rotate = gt_rotate[:, ::flip]
        for i in range(source_img.shape[0]):
            source_img[i] = (source_img[i] - source_img[i].mean()) / source_img[i].std()
        for i in range(source_img_rotate.shape[0]):
            source_img_rotate[i] = (source_img_rotate[i] - source_img_rotate[i].mean()) / source_img_rotate[i].std()
        for i in range(target_img.shape[0]):
            target_img[i] = (target_img[i] - target_img[i].mean()) / target_img[i].std()
        for i in range(target_img_rotate.shape[0]):
            target_img_rotate[i] = (target_img_rotate[i] - target_img_rotate[i].mean()) / target_img_rotate[i].std()
        mask = np.zeros_like(gt)
        mask[gt > 0] = 1
        mask[gt == 128] = 2
        gt = mask
        mask = np.zeros_like(gt_rotate)
        mask[gt_rotate > 0] = 1
        mask[gt_rotate == 128] = 2
        gt_rotate = mask
        data = {'source_image': source_img.copy(), 'source_image_rotate': source_img_rotate.copy(), 'target_image': target_img.copy(), 'target_image_rotate': target_img_rotate.copy(), 'label': gt.copy(), 'label_rotate': gt_rotate.copy()}
        # if self.transform:
        #     data = self.transform(data)

        return data, weight, gt_path

    def __len__(self):
        return len(self.files)

if __name__ == '__main__':
    Source_data = Polyp(root='/home/cyang/SFDA/data/EndoScene', data_dir='/home/cyang/SFDA/dataset/EndoScene_list/train.lst', mode='train', max_iter=15000)
    print(Source_data.__len__())
    # for i in range(15000):
    #     print(Source_data[i])
    train_loader = torch.utils.data.DataLoader(Source_data, batch_size=8, shuffle=True, num_workers=4)
    print(np.max(Source_data[0][0]['image']), np.min(Source_data[0][0]['image']))
