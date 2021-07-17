""" Train RPMNet

Example usage:
    python train.py --noise_type crop
    python train.py --noise_type jitter --train_batch_size 4
"""
from collections import defaultdict
import os
import random
from typing import Dict, List

from matplotlib.pyplot import cm as colormap
import numpy as np
import open3d  # Ensure this is imported before pytorch
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
import time
from arguments import rpmnet_train_arguments
from common.colors import BLUE, ORANGE
from common.misc import prepare_logger
from common.torch import dict_all_to_device, CheckPointManager, TorchDebugger
from common.math_torch import se3
from data_loader.datasets import get_train_datasets
from eval import compute_metrics, summarize_metrics, print_metrics
from models.rpmnet_cls import *
from common.torch import to_numpy
from models.pointnet_util import square_distance, angle_difference
from torch.autograd import Variable
from sklearn.neighbors import NearestNeighbors
from models.pointnet_util import square_distance
import torch.nn.functional as F
# Set up arguments and logging
parser = rpmnet_train_arguments()
_args = parser.parse_args()
# _args.method_type = "ar"
if _args.method_type == "voxnet":
    _args.num_points = 2048
_args.task = 'cls'

_logger, _log_path = prepare_logger(_args)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
_EPS = 1e-8  # To prevent division by zero
def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.2
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss
def main():

    train_set, val_set = get_train_datasets(_args)
    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=_args.train_batch_size, shuffle=True, num_workers=_args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=_args.val_batch_size, shuffle=True, num_workers=_args.num_workers)
    run(train_loader, val_loader)

def loss(x, mask):
    # temp0 = mask.data.cpu().numpy()
    temp = (mask-x).data.cpu().numpy()
    idx1 = np.where(temp>170/255)
    idx0 = np.where(temp<-80/255)
    return (idx1[0].shape[0])/(idx0[0].shape[0]+0.001)*F.mse_loss((mask)*x, mask)+F.mse_loss((1-mask)*x,mask)

def process(data, model, align_model=None):
    target = Variable(data['label_src'].type(torch.LongTensor)).to(device)
    if _args.method_type == 'voxnet':
        label_ref, _, _, _, _, classes_src, classes_ref, gt_label, label_src, _, vox_src, _ = [Variable(data[k].type(torch.FloatTensor)).to(device) for k in data]
        pred, glob_feat = model(vox_src,  method_type=_args.method_type)
        losses = cal_loss(glob_feat, target)
    elif _args.method_type == 'autonet':
        label_ref, _, _, _, _, classes_src, classes_ref, gt_label, label_src, _, vox_src, _ = [Variable(data[k].type(torch.FloatTensor)).to(device) for k in data]
        pred, out_vox, glob_feat = model(vox_src,  method_type=_args.method_type)
        losses = cal_loss(glob_feat, target) + loss(out_vox, vox_src)

    elif _args.method_type == 'ar':
        _, points_src, _, _, _, _, _, gt_label, _, _ = [Variable(data[k].type(torch.FloatTensor)).to(device) for k in data]
        feat_src, pooled_feat_src, new_feat_src = align_model(points_src,  method_type=_args.method_type)
        glob_feat = model(feat_src, points_src, pooled_feat_src, new_feat_src)
        pred = glob_feat
        # pred = F.softmax(glob_feat, dim=1)
        losses = cal_loss(pred, target)
    else:
        _, points_src, _, _, _, _, _, gt_label, _, _ = [Variable(data[k].type(torch.FloatTensor)).to(device) for k in data]
        pred, glob_feat = model(points_src,  method_type=_args.method_type)
        losses = cal_loss(glob_feat, target)

    return losses, pred, target
def validate(data_loader, model, align_model=None):
    """Perform a single validation run, and saves results into tensorboard summaries"""
    _logger.info('Starting validation run...')
    pred_set = []
    target_set = []
    with torch.no_grad():
        all_val_losses = defaultdict(list)
        for i, data in enumerate(data_loader) or i > 5:
            if data['points_ref'].shape[0] < _args.val_batch_size:
                continue
            if _args.method_type == "ar":
                losses, pred, target = process(data, model, align_model)
            else:
                losses, pred, target = process(data, model)
            all_val_losses['total'].append(losses.detach().cpu().numpy())
            pred = torch.max(pred, dim=1)[1]
            pred = pred.detach().cpu().numpy(),
            target = target.detach().cpu().numpy(),
            pred_set.append( pred )
            target_set.append( target )

        all_val_losses = np.array([all_val_losses[k] for k in all_val_losses])
        mean_val_losses = np.mean(all_val_losses)
    print_metrics(_logger, mean_val_losses, 'Validation results')
    pred_array = np.array(pred_set).reshape(-1)
    target_array = np.array(target_set).reshape(-1)
    idx1 = np.where(pred_array-target_array == 0)
    mean_val_losses = idx1[0].shape[0] / pred_array.shape[0]
    print_metrics(_logger, mean_val_losses, 'success ratio')
    return mean_val_losses

def run(train_loader, val_loader):
    """Main train/val loop"""
    _logger.debug('Trainer (PID=%d), %s', os.getpid(), _args)
    if _args.method_type=="ar":
        align_model = get_model(_args)
        if torch.cuda.device_count() > 1:
            align_model = nn.DataParallel(align_model, device_ids=[0, 1, 2, 3]).cuda()
        align_model.to(device)
        saver_model0 = CheckPointManager(os.path.join(_log_path, 'ckpt', 'model'), keep_checkpoint_every_n_hours=0.5)
        global_step = saver_model0.load('/root/RPMNet/logs/210114_142504/ckpt/model-best.pth', align_model)
        align_model.eval()

        model = GFNet(_args)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
        model.to(device)
    else:
        model = get_model(_args)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
        model.to(device)

    # optimizer
    model_optimizer = torch.optim.Adam(model.parameters(), lr=_args.lr)
    # Checkpoint manager
    saver_model = CheckPointManager(os.path.join(_log_path, 'ckpt', 'model'+ '_' + _args.method_type), keep_checkpoint_every_n_hours=0.5)
    if _args.resume is not None:
        global_step = saver_model.load(_args.resume, model, model_optimizer)
    else:
        global_step = 0

    # trainings
    torch.autograd.set_detect_anomaly(_args.debug)
    model.train()

    for epoch in range(0, _args.epochs):
        _logger.info('Begin epoch {} (steps {} - {})'.format(epoch, global_step, global_step + len(train_loader)))
        tbar = tqdm(total=len(train_loader), ncols=100)
        for idx, data in enumerate(train_loader):
            global_step += 1
            model_optimizer.zero_grad()
            # Forward through neural network
            if data['points_ref'].shape[0] < _args.train_batch_size:
                continue

            if _args.method_type == "ar":
                losses, pred, target = process(data, model, align_model)
            else:
                losses, pred, target = process(data, model)

            if _args.debug:
                with TorchDebugger():
                    losses.backward()
            else:
                losses.backward()

            model_optimizer.step()
            tbar.set_description('Loss:{:.3g}'.format(losses))
            tbar.update(1)

        model.eval()
        score = validate(val_loader, model, align_model=align_model)
        saver_model.save(model, model_optimizer, step=global_step,  score=score)
        model.train()
        tbar.close()

    _logger.info('Ending training. Number of steps = {}.'.format(global_step))

if __name__ == '__main__':

    main()
