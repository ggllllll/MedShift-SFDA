# -*- encoding: utf-8 -*-
#Time        :2020/12/13 19:19:15
#Author      :Chen
#FileName    :domain_inversion.py
#Version     :1.0

import os
import torch
import torch.nn as nn
from utils.load_img import load_img_Pro, show_img
import scipy.misc
from PIL import Image
import numpy as np
import torchvision.models as models
import torch.optim as optim
from torch.autograd import Variable
from models.deeplab import Deeplab
from torchvision import transforms
from utils.fda import FDA_source_to_target_np
from utils.pilutil import toimage
from models.unet import UNet
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# Eq. (1): style loss between noise image and BN statistics of source model
class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        #forcing mean and variance to match between two distributions
        #other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no output

    def close(self):
        self.hook.remove()

# Eq. (2): Content loss between noise image and target image
class Content_Loss(nn.Module):
    def __init__(self, target, weight):
        super(Content_Loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * self.weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        out = input.clone()
        return out

    def backward(self):
        self.loss.backward(retain_graph = True)
        return self.loss

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def main(image_name, target_dir, coarse_source_dir):
    # model = Deeplab(num_classes=2, pretrained=False, inversion=True).cuda()
    model = UNet(inversion=True)
    # load well trained source model
    # model.load_state_dict(torch.load('/home/cyang53/CED/Ours/SFDA-FSM/checkpoint/Endo_best.pth'))
    model.load_state_dict(torch.load(r' ')['model_state_dict'])
    model.apply(fix_bn)
    model.cuda()

    criterion = nn.MSELoss()
    content_layers_default = ['conv_4', 'conv_5', 'conv_6']

    content_img = load_img_Pro(os.path.join(target_dir, image_name))
    content_img = torch.from_numpy(content_img).cuda().to(dtype=torch.float32).unsqueeze(0)
    wce = model(content_img)
    
    # initialize noise image with Gaussian distribution or content image
    input_img = Variable(torch.randn(content_img.size()), requires_grad=True).cuda()
    # input_img = content_img.clone()

    loss_r_feature_layers = []

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))


    input_param = nn.Parameter(input_img.data)
    #optimizer = optim.LBFGS([input_param])
    optimizer = optim.Adam([input_param], lr=0.01, betas=[0.5, 0.9], eps=1e-8)

    for i in range(100):
        optimizer.zero_grad()
        output = model(input_param)
        loss_content = criterion(output[1], wce[1]) + criterion(output[2], wce[2]) + criterion(output[0], wce[0])
        loss_r_feature = sum([mod.r_feature for (idx, mod) in
                              enumerate(loss_r_feature_layers[1:24])])
        
        # Eq. (10): combine style loss and content loss to optimize input noise image 
        loss = loss_r_feature + 0.5 * loss_content
        
        loss.backward(retain_graph=True)
        optimizer.step()

    input_param.data.clamp_(0, 1)

    save_pic = transforms.ToPILImage()(input_param.data.cpu().squeeze(0))

    save_pic.save(os.path.join(coarse_source_dir, image_name))

if __name__ == '__main__':
    target_dir = r' '
    coarse_source_dir = r' '
    fine_source_dir = r' '
    dir_list = os.listdir(target_dir)
    for i in dir_list:
        main(i, target_dir, coarse_source_dir)
        im_src = Image.open(os.path.join(target_dir, i)).convert('RGB')
        im_trg = Image.open(os.path.join(coarse_source_dir, i)).convert('RGB')

        im_src = im_src.resize((512, 512), Image.BICUBIC)
        im_trg = im_trg.resize((512, 512), Image.BICUBIC)

        im_src = np.asarray(im_src, np.float32)
        im_trg = np.asarray(im_trg, np.float32)

        im_src = im_src.transpose((2, 0, 1))
        im_trg = im_trg.transpose((2, 0, 1))
        
        # Eq. (3): mutual fourier transform between coarse generation and target image
        src_in_trg = FDA_source_to_target_np(im_src, im_trg, L=0.05)

        src_in_trg = src_in_trg.transpose((1, 2, 0))
        toimage(src_in_trg, cmin=0.0, cmax=255.0).save(os.path.join(fine_source_dir, i))
