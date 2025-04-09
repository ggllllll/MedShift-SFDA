
#!/usr/bin/env python

import argparse
import os
import os.path as osp
import torch.nn.functional as F

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import tqdm
from dataloaders import fundus_dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
# from scipy.misc import imsave
from matplotlib.pyplot import imsave
from utils.Utils import *
from utils.metrics import *
from plot.evaluate_mu import eval_print_all_CU
from datetime import datetime
import pytz
import networks.deeplabv3 as netd
import networks.deeplabv3_eval as netd_eval
import cv2
import torch.backends.cudnn as cudnn
import random
from tensorboardX import SummaryWriter
from networks.unet import UNet
bceloss = torch.nn.BCELoss(reduction='none')
seed = 3377
savefig = False
get_hd = True
model_save = True
if True:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
import time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default=r'')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--source', type=str, default='1')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--root', default=r'')
    parser.add_argument('--data-dir', default=r'')
    parser.add_argument('--out-stride',type=int,default=16)
    parser.add_argument('--sync-bn',type=bool,default=True)
    parser.add_argument('--freeze-bn',type=bool,default=False)
    args = parser.parse_args()
    inference_tag = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file
    root_folder = args.root
    # 1. dataset
    composed_transforms_train = transforms.Compose([
        tr.Resize(512),
        tr.add_salt_pepper_noise(),
        tr.adjust_light(),
        tr.eraser(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    composed_transforms_test = transforms.Compose([
        tr.Resize(512),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    db_train = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='train/ROIs', transform=composed_transforms_train)
    db_test = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='test/ROIs', transform=composed_transforms_test)
    db_source = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.source, split='train/ROIs', transform=composed_transforms_test)

    train_loader = DataLoader(db_train, batch_size=8, shuffle=False, num_workers=1)
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    source_loader = DataLoader(db_source, batch_size=1, shuffle=False, num_workers=1)

    model = UNet()
    model_eval = UNet()
    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.train()
    visualization_folder = r''
    if args.dataset=="":
        npfilename = r''
    npdata = np.load(npfilename, allow_pickle=True)
    pseudo_label_dic = npdata['arr_0'].item()
    uncertain_dic = npdata['arr_1'].item()
    proto_pseudo_dic = npdata['arr_2'].item()
    var_list = model.named_parameters()
    optim_gen = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.99))
    best_val_cup_dice = 0.0;
    best_val_disc_dice = 0.0;
    best_avg = 0.0
    start_time = time.time()
    iter_num = 0
    for epoch_num in tqdm.tqdm(range(100), ncols=75):
        model.train()
        epoch_start_time = time.time()
        for batch_idx, (sample) in enumerate(train_loader):
            data, target, img_name = sample['image'], sample['map'], sample['img_name']
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            prediction, _, feature = model(data)
            prediction = torch.sigmoid(prediction)

            pseudo_label = [pseudo_label_dic.get(key) for key in img_name]
            uncertain_map = [uncertain_dic.get(key) for key in img_name]
            proto_pseudo = [proto_pseudo_dic.get(key) for key in img_name]

            pseudo_label = torch.from_numpy(np.asarray(pseudo_label)).float().cuda()
            uncertain_map = torch.from_numpy(np.asarray(uncertain_map)).float().cuda()
            proto_pseudo = torch.from_numpy(np.asarray(proto_pseudo)).float().cuda()

            for param in model.parameters():
                param.requires_grad = True
            optim_gen.zero_grad()

            target_0_obj = F.interpolate(pseudo_label[:,0:1,...], size=feature.size()[2:], mode='nearest')
            target_1_obj = F.interpolate(pseudo_label[:, 1:, ...], size=feature.size()[2:], mode='nearest')
            target_0_bck = 1.0 - target_0_obj;target_1_bck = 1.0 - target_1_obj

            mask_0_obj = torch.zeros([pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
            mask_0_bck = torch.zeros([pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
            mask_1_obj = torch.zeros([pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
            mask_1_bck = torch.zeros([pseudo_label.shape[0], 1, pseudo_label.shape[2], pseudo_label.shape[3]]).cuda()
            mask_0_obj[uncertain_map[:, 0:1, ...] < 0.05] = 1.0
            mask_0_bck[uncertain_map[:, 0:1, ...] < 0.05] = 1.0
            mask_1_obj[uncertain_map[:, 1:, ...] < 0.05] = 1.0
            mask_1_bck[uncertain_map[:, 1:, ...] < 0.05] = 1.0
            mask = torch.cat((mask_0_obj*pseudo_label[:,0:1,...] + mask_0_bck*(1.0-pseudo_label[:,0:1,...]), mask_1_obj*pseudo_label[:,1:,...] + mask_1_bck*(1.0-pseudo_label[:,1:,...])), dim=1)

            mask_proto = torch.zeros([data.shape[0], 2, data.shape[2], data.shape[3]]).cuda()
            mask_proto[pseudo_label==proto_pseudo] = 1.0

            mask = mask*mask_proto

            loss_seg_pixel = bceloss(prediction, pseudo_label)
            loss_seg = torch.sum(mask * loss_seg_pixel) / torch.sum(mask)
            loss_seg.backward()
            optim_gen.step()
            iter_num = iter_num + 1
        model_eval.train()
        pretrained_dict = model.state_dict()
        model_dict = model_eval.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_eval.load_state_dict(pretrained_dict)

        val_cup_dice = 0.0;val_disc_dice = 0.0;datanum_cnt = 0.0
        cup_hd = 0.0; disc_hd = 0.0;datanum_cnt_cup = 0.0;datanum_cnt_disc = 0.0
        with torch.no_grad():
            for batch_idx, (sample) in enumerate(test_loader):
                data, target, img_name = sample['data'], sample['seg'], sample['name']
                data = torch.from_numpy(data).to(dtype=torch.float32)
                target = torch.from_numpy(target).to(dtype=torch.float32)
                target = F.one_hot(target.to(torch.int64)).transpose(1, 4).squeeze(-1)[:, 1:3, ...]
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                prediction, _ = model_eval(data, rfeat=True)
                prediction = torch.sigmoid(prediction)

                target_numpy = target.data.cpu()
                prediction = prediction.data.cpu()

                case_seg = np.zeros((512, 512))
                case_seg[prediction[0][0] > 0.75] = 255
                # case_seg[prediction[0][1] > 0.75] = 128
                case_seg_f = Image.fromarray(case_seg.astype(np.uint8)).resize((512, 512), resample=Image.NEAREST)
                case_seg_f.save(os.path.join(os.path.join(visualization_folder, str(epoch_num)),
                                             img_name[0].split('/')[-1].replace('.tif', '-1.tif')))

                prediction[prediction > 0.75] = 1;
                prediction[prediction <= 0.75] = 0
        os.makedirs(os.path.join(visualization_folder, str(epoch_num)), exist_ok=True)
        savefile = os.path.join(os.path.join(visualization_folder, str(epoch_num)), 'checkpoint.pth.tar')
        if model_save:
            torch.save({
                'model_state_dict': model.state_dict(),
            }, savefile)

        model.train()



