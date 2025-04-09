# /usr/bin/env python3.6
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
import re
import argparse
import warnings

warnings.filterwarnings("ignore")
from pathlib import Path
from operator import itemgetter
from shutil import copytree, rmtree
import typing
from typing import Any, Callable, List, Tuple
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from dice3d import dice3d
from networks.networks import weights_init
from dataloader import get_loaders
from utils import map_, save_dict_to_file
from utils import dice_coef, dice_batch, save_images, save_images_p, save_be_images, tqdm_, save_images_ent
from utils import probs2one_hot, probs2class, mask_resize, resize, haussdorf
from utils import exp_lr_scheduler
import datetime
from itertools import cycle
from time import sleep
from bounds import CheckBounds
import matplotlib.pyplot as plt
from itertools import chain
import platform
from networks.unet import UNet
from PIL import Image

from plot.evaluate_mu import eval_print_all_CU


def normalize_image(img_npy):
    """
    :param img_npy: b, c, h, w
    """
    for b in range(img_npy.shape[0]):
        for c in range(img_npy.shape[1]):
            img_npy[b, c] = (img_npy[b, c] - img_npy[b, c].mean()) / img_npy[b, c].std()
    return img_npy

def setup(args, n_class, dtype) -> Tuple[
    Any, Any, Any, List[Callable], List[float], List[Callable], List[float], Callable]:
    print(">>> Setting up")
    cpu: bool = args.cpu or not torch.cuda.is_available()
    if cpu:
        print("WARNING CUDA NOT AVAILABLE")
    device = torch.device("cpu") if cpu else torch.device("cuda")
    n_epoch = args.n_epoch
    if args.model_weights:
        if cpu:
            net = torch.load(args.model_weights, map_location='cpu')
        else:
            net = UNet()
            checkpoint = torch.load(args.model_weights)['model_state_dict']
            net.load_state_dict(checkpoint)
    else:
        net_class = getattr(__import__('networks'), args.network)
        net = net_class(1, n_class).type(dtype).to(device)
        net.apply(weights_init)
    net.to(device)
    if args.saveim:
        print("WARNING SAVING MASKS at each epc")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.l_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    if args.adamw:
        optimizer = torch.optim.AdamW(net.parameters(), lr=args.l_rate, betas=(0.9, 0.999))

    # print(args.target_losses)
    losses = eval(args.target_losses)
    loss_fns: List[Callable] = []
    for loss_name, loss_params, _, bounds_params, fn, _ in losses:
        loss_class = getattr(__import__('losses'), loss_name)
        loss_fns.append(loss_class(**loss_params, dtype=dtype, fn=fn))
        # print("bounds_params", bounds_params)
        if bounds_params != None:
            bool_predexist = CheckBounds(**bounds_params)
            # print(bool_predexist,"size predictor")
            if not bool_predexist:
                n_epoch = 0

    loss_weights = map_(itemgetter(5), losses)

    if args.scheduler:
        scheduler = getattr(__import__('scheduler'), args.scheduler)(**eval(args.scheduler_params))
    else:
        scheduler = ''

    return net, optimizer, device, loss_fns, loss_weights, scheduler, n_epoch


def do_epoch(args, mode: str, net: Any, device: Any, epc: int,
             loss_fns: List[Callable], loss_weights: List[float],
             new_w: int, C: int, metric_axis: List[int], savedir: str = "",
             optimizer: Any = None, target_loader: Any = None, best_dice3d_val: Any = None):
    assert mode in ["train", "val"]
    L: int = len(loss_fns)
    indices = torch.tensor(metric_axis, device=device)
    if mode == "train":
        net.train()
        desc = f">> Training   ({epc})"
    elif mode == "val":
        net.eval()
        desc = f">> Validation ({epc})"

    total_it_t, total_images_t = len(target_loader), len(target_loader.dataset)
    total_iteration = total_it_t
    total_images = total_images_t

    if args.debug:
        total_iteration = 10
    pho = 1
    dtype = eval(args.dtype)

    tq_iter = tqdm_(enumerate(target_loader), total=total_iteration, desc=desc)
    done: int = 0
    n_warmup = args.n_warmup
    mult_lw = [pho ** (epc - n_warmup + 1)] * len(loss_weights)
    mult_lw[0] = 1
    loss_weights = [a * b for a, b in zip(loss_weights, mult_lw)]
    losses_vec, source_vec, target_vec, baseline_target_vec = [], [], [], []
    pen_count = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        count_losses = 0
        for j, target_data in tq_iter:
            img_file, img_npy, mask, bounds = target_data
            img = torch.from_numpy(normalize_image(img_npy.numpy())).cuda().to(dtype=torch.float32)

            filenames_target, target_image, target_gt = img_file, img, mask
            bounds = bounds
            # filenames_target = [f.split('.nii')[0] for f in filenames_target]
            B = len(target_image)
            # Reset gradients
            if optimizer:
                # adjust_learning_rate(optimizer, 1, args.l_rate, args.power)
                optimizer.zero_grad()
            # Forward
            with torch.set_grad_enabled(mode == "train"):
                pred_logits: Tensor = net(target_image)
                pred_probs = torch.sigmoid(pred_logits)
                # predicted_mask: Tensor = probs2one_hot(pred_probs)  # Used only for dice computation

            if epc < n_warmup:
                loss_weights = [0] * len(loss_weights)
            loss: Tensor = torch.zeros(1, requires_grad=True).to(device)
            loss_vec = []
            loss_kw = []
            for loss_fn, w, bound in zip(loss_fns, loss_weights, bounds):
                if w > 0:
                    if eval(args.target_losses)[0][0] == "EntKLProp":
                        loss_1, loss_cons_prior, est_prop = loss_fn(pred_probs, 'label',
                                                                    bound.cuda().to(dtype=torch.float32))
                        loss = loss_1 + loss_cons_prior
                    else:
                        loss = loss_fn(pred_probs, 'label', bound)
                        loss = w * loss
                        loss_1 = loss
                    loss_kw.append(loss_1.detach())
            # Backward
            if optimizer:
                loss.backward()
                optimizer.step()
            # Save images
            if savedir and args.saveim and mode == "val":
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    warnings.simplefilter("ignore")
                    pred_probs = pred_probs.cpu()
                    for i in range(pred_probs.shape[0]):
                        case_seg = np.zeros((512, 512))
                        case_seg[pred_probs[i][0] > 0.5] = 255
                        case_seg_f = Image.fromarray(case_seg.astype(np.uint8)).resize((512, 512),
                                                                                       resample=Image.NEAREST)
                        case_seg_f.save(
                            os.path.join(savedir, filenames_target[i].split('/')[-1].replace('.tif', '-1.tif')))
        if savedir and args.saveim and mode == "val":
            eval_print_all_CU(savedir + '/', os.path.join(args.root, 'RIGA-mask', args.inference_tag, 'Labeled'))
            saved_model = {
                'model_state_dict': net.state_dict(),
            }
            print('  Saving model_{}.model...'.format('latest'))
            torch.save(saved_model, os.path.join(savedir + '/', 'model.model'))
            # predicted_class: Tensor = probs2class(pred_probs)
            # save_images(predicted_class, filenames_target, savedir, mode, epc, False)
            # if args.entmap:
            #     ent_map = torch.einsum("bcwh,bcwh->bwh", [-pred_probs, (pred_probs+1e-10).log()])
            #     save_images_ent(ent_map, filenames_target, savedir,'ent_map', epc)

    return losses_vec, target_vec, source_vec


def run(args: argparse.Namespace) -> None:
    d = vars(args)
    d['time'] = str(datetime.datetime.now())
    d['server'] = platform.node()
    save_dict_to_file(d, args.workdir)
    temperature: float = 0.1
    n_class: int = args.n_class
    metric_axis: List = args.metric_axis
    lr: float = args.l_rate
    dtype = eval(args.dtype)
    savedir: str = args.workdir
    n_epoch: int = args.n_epoch
    net, optimizer, device, loss_fns, loss_weights, scheduler, n_epoch = setup(args, n_class, dtype)
    shuffle = True
    # print(args.target_folders)

    target_loader, target_loader_val = get_loaders(args, args.target_dataset, args.target_folders,
                                                   args.batch_size, n_class,
                                                   args.debug, args.in_memory, dtype, shuffle, "target",
                                                   args.val_target_folders)

    # print("metric axis",metric_axis)
    best_dice_pos: Tensor = np.zeros(1)
    best_dice: Tensor = np.zeros(1)
    best_hd3d_dice: Tensor = np.zeros(1)
    best_3d_dice: Tensor = 0
    best_2d_dice: Tensor = 0
    # print("Results saved in ", savedir)
    # print(">>> Starting the training")

    for i in range(n_epoch):
        savedir_epoch = os.path.join(savedir + '/', str(i))
        os.makedirs(savedir_epoch, exist_ok=True)
        print("Results saved in ", savedir_epoch)

        if args.mode == "makeim":
            with torch.no_grad():

                val_losses_vec, val_target_vec, val_source_vec = do_epoch(args, "val", net, device,
                                                                          i, loss_fns,
                                                                          loss_weights,
                                                                          args.resize,
                                                                          n_class, metric_axis,
                                                                          savedir=savedir,
                                                                          target_loader=target_loader_val,
                                                                          best_dice3d_val=best_3d_dice)

                tra_losses_vec = val_losses_vec
                tra_target_vec = val_target_vec
                tra_source_vec = val_source_vec
        else:
            tra_losses_vec, tra_target_vec, tra_source_vec = do_epoch(args, "train", net, device,
                                                                      i, loss_fns,
                                                                      loss_weights,
                                                                      args.resize,
                                                                      n_class, metric_axis,
                                                                      savedir=savedir_epoch,
                                                                      optimizer=optimizer,
                                                                      target_loader=target_loader,
                                                                      best_dice3d_val=best_3d_dice)

            with torch.no_grad():
                val_losses_vec, val_target_vec, val_source_vec = do_epoch(args, "val", net, device,
                                                                          i, loss_fns,
                                                                          loss_weights,
                                                                          args.resize,
                                                                          n_class, metric_axis,
                                                                          savedir=savedir_epoch,
                                                                          target_loader=target_loader_val,
                                                                          best_dice3d_val=best_3d_dice)

        if args.flr == False:
            exp_lr_scheduler(optimizer, i, args.lr_decay, args.lr_decay_epoch)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--target_dataset', default=[r''])
    parser.add_argument('--test_dataset', default=[r''])
    parser.add_argument('--inference_tag', default='')
    parser.add_argument('--root', default=r'')
    parser.add_argument("--workdir", type=str, default=r'')
    parser.add_argument("--target_losses", type=str,
                        help="List of (loss_name, loss_params, bounds_name, bounds_params, fn, weight)",
                        default="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sizefile':'sizes/REFUGE_1_unlabeled.csv'},'norm_soft_size',1)]")
    parser.add_argument("--target_folders", type=str,
                        help="List of (subfolder, transform, is_hot)",
                        default="[('Inn', png_transform, False), ('GT', gtpng_transform, False),('GT', gtpng_transform, False)]")
    parser.add_argument("--val_target_folders", type=str,
                        help="List of (subfolder, transform, is_hot)",
                        default="[('Inn', png_transform, False), ('GT', gtpng_transform, False),('GT', gtpng_transform, False)]")
    parser.add_argument("--network", type=str, help="The network to use", default='UNet')
    parser.add_argument("--grp_regex", type=str, default='Subj_\d+_\d+')
    parser.add_argument("--n_class", type=int, default=2)
    parser.add_argument("--mode", type=str, default="learn")
    parser.add_argument("--lin_aug_w", action="store_true")
    parser.add_argument("--both", action="store_true")
    parser.add_argument("--trainval", action="store_true")
    parser.add_argument("--valonly", action="store_true")
    parser.add_argument("--flr", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--mix", type=bool, default=True)
    parser.add_argument("--do_hd", type=bool, default=90)
    parser.add_argument("--saveim", type=bool, default=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--csv", type=str, default='metrics.csv')
    parser.add_argument("--source_metrics", action="store_true")
    parser.add_argument("--adamw", action="store_true")
    parser.add_argument("--dice_3d", action="store_true")
    parser.add_argument("--ontest", action="store_true")
    parser.add_argument("--ontrain", action="store_true")
    parser.add_argument("--best_losses", action="store_true")
    parser.add_argument("--pprint", action="store_true")
    parser.add_argument("--entmap", action="store_true", default=True)
    parser.add_argument("--model_weights", type=str,
                        default=r'')
    parser.add_argument("--cpu", action='store_true')
    parser.add_argument("--in_memory", action='store_true')
    parser.add_argument("--resize", type=int, default=0)
    parser.add_argument("--pho", nargs='?', type=float, default=1,
                        help='augment')
    parser.add_argument("--n_warmup", type=int, default=0)
    parser.add_argument('--n_epoch', nargs='?', type=int, default=1,
                        help='# of the epochs')
    parser.add_argument('--l_rate', nargs='?', type=float, default=0.000001,
                        help='Learning Rate')
    parser.add_argument('--lr_decay', nargs='?', type=float, default=0.9),
    parser.add_argument('--lr_decay_epoch', nargs='?', type=float, default=20),
    parser.add_argument('--weight_decay', nargs='?', type=float, default=1e-4,
                        help='L2 regularisation of network weights')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument("--dtype", type=str, default="torch.float32")
    parser.add_argument("--scheduler", type=str, default="DummyScheduler")
    parser.add_argument("--scheduler_params", type=str, default="{}")
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--metric_axis", type=int, nargs='*', help="Classes to display metrics. \
        Display only the average of everything if empty", default=1)
    args = parser.parse_args()
    print(args)

    return args


if __name__ == '__main__':
    run(get_args())
