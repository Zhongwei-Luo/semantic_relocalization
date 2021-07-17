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
from models.rpmnet_rel import *
from common.torch import to_numpy
from models.pointnet_util import square_distance, angle_difference
from torch.autograd import Variable
from sklearn.neighbors import NearestNeighbors
from models.pointnet_util import square_distance
import torch.nn.functional as F
import shutil
from loss import *

# Set up arguments and logging
parser = rpmnet_train_arguments()
_args = parser.parse_args()
if _args.method_type=="voxnet":
    _args.num_points = 2048
_args.task = "retrival"
if _args.method_type == "rpmnet":
    my_file = '../datasets/' + _args.method_type + '_' + 'validation_data/'
    if os.path.exists(my_file):
        shutil.rmtree(my_file)
        os.makedirs(my_file)
    else:
        os.makedirs(my_file)

_logger, _log_path = prepare_logger(_args)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
_EPS = 1e-8  # To prevent division by zero
_thresh = 255
_S = 20

def main():
    train_set, val_set = get_train_datasets(_args)
    # dataloaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=_args.train_batch_size, shuffle=True, num_workers=_args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=_args.val_batch_size, shuffle=True, num_workers=_args.num_workers)
    run(train_loader, val_loader)

def process(data, model, epoch=0):
    if _args.method_type == 'voxnet':
        gt_label = Variable(construct_gt_label(data['gt_label'])).to(device)
        vox_ref = Variable(data['vox_ref'].type(torch.FloatTensor)).to(device)
        vox_src = Variable(data['vox_src'].type(torch.FloatTensor)).to(device)

        glob_feat_src, glob_feat_ref = model(vox_src, vox_ref, type='test', method_type=_args.method_type)
        logit = square_distance(glob_feat_src, glob_feat_ref)
        losses = compute_losses(logit, gt_label)
        return losses, glob_feat_src, glob_feat_ref

def validate(train_loader, data_loader, model: torch.nn.Module, align_model=None):
    """Perform a single validation run, and saves results into tensorboard summaries"""
    _logger.info('Starting validation run...')
    glob_feat_src_set = []
    glob_feat_ref_set = []
    label_ref = []
    label_src = []
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            if data['points_ref'].shape[0] < _args.train_batch_size:
                continue
            losses, glob_feat_src, glob_feat_ref = process(data, model, epoch=i+1)

            glob_feat_ref = glob_feat_ref.detach().cpu().numpy()
            if isinstance(glob_feat_ref, tuple):
                glob_feat_ref_set.append(glob_feat_ref[0])
            else:
                glob_feat_ref_set.append(glob_feat_ref)
            label_ref.append(data["gt_label"].cpu().numpy())
            # __import__('ipdb').set_trace()
        all_val_losses = defaultdict(list)
        for i in range(1):
            for j, data in enumerate(data_loader):
                if data['points_ref'].shape[0] < _args.val_batch_size:
                    continue
                losses, glob_feat_src, glob_feat_ref = process(data, model, epoch=j+i+1)
                all_val_losses['total'].append(losses.detach().cpu().numpy())
                glob_feat_src = glob_feat_src.detach().cpu().numpy(),
                glob_feat_ref = glob_feat_ref.detach().cpu().numpy(),
                if isinstance(glob_feat_ref, tuple):
                    glob_feat_ref_set.append(glob_feat_ref[0])
                else:
                    glob_feat_ref_set.append(glob_feat_ref)
                label_ref.append(data["gt_label"].cpu().numpy())

                if isinstance(glob_feat_src, tuple):
                    glob_feat_src_set.append(glob_feat_src[0])
                else:
                    glob_feat_src_set.append(glob_feat_src)
                label_src.append(data["gt_label"].cpu().numpy())

        all_val_losses = np.array([all_val_losses[k] for k in all_val_losses])
        mean_val_losses = np.mean(all_val_losses)
    print_metrics(_logger, mean_val_losses, 'Validation results')

    global_feat_ref_array = np.concatenate(glob_feat_ref_set, axis=0)
    global_feat_src_array = np.concatenate(glob_feat_src_set, axis=0)
    label_src = np.concatenate(label_src, axis=0)
    label_ref = np.concatenate(label_ref, axis=0)
    nbrs = NearestNeighbors(n_neighbors=20, algorithm='kd_tree').fit(global_feat_ref_array)
    distances, indices0 = nbrs.kneighbors(global_feat_src_array)

    indices = indices0[:, 0]
    idx = label_ref[indices] - label_src
    idx1 = np.where(idx == 0)
    mean_val_losses = idx1[0].shape[0] / idx.shape[0]
    print_metrics(_logger, mean_val_losses, 'success ratio top-1: ')

    idx = np.min(np.abs(label_ref[indices0] - label_src.reshape(-1,1)), axis=1)
    a = label_ref[indices0] - label_src.reshape(-1,1)
    idx1 = np.where(idx == 0)
    mean_val_losses_5 = idx1[0].shape[0] / idx.shape[0]
    print_metrics(_logger, mean_val_losses_5, 'success ratio top-5: ')

    return mean_val_losses

def run(train_loader, val_loader):
    """Main train/val loop"""
    _logger.debug('Trainer (PID=%d), %s', os.getpid(), _args)

    model = get_model(_args)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
    model.to(device)
    # optimizer
    model_optimizer = torch.optim.Adam(model.parameters(), lr=_args.lr)
    # Checkpoint manager
    name = _args.model_data + '_' + _args.loss + '_' + str(_args.lamb) + '_' + _args.method_type + '_' + str(_args.object) + '_' + str(_args.rot_mag) + '_' + str(_args.partial[0]*100)
    saver_model = CheckPointManager(os.path.join(_log_path, 'ckpt', 'model' + '_' +  name), keep_checkpoint_every_n_hours=0.5)
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

            losses, glob_feat_src, glob_feat_ref = process(data, model)
            losses.backward()
            model_optimizer.step()

            tbar.set_description('Loss:{:.3g}'.format(losses))
            tbar.update(1)

        model.eval()

        score = validate(train_loader, val_loader, model)
        saver_model.save(model, model_optimizer, step=global_step,  score=score)
        model.train()
        tbar.close()

    _logger.info('Ending training. Number of steps = {}.'.format(global_step))


if __name__ == '__main__':
    object_list = ['03001627', '04256520', '04379243', '02818832']
    for obj in object_list:
        _args.model_data = obj
        _args.save_data = 0
        _args.dataset_path = "/p300/Scan2CAD/data/object_model/"
        generate_data(_args.dataset_path)
        _args.dataset_type = 'scan2cad'
        _args.epochs = 200
        object_set = [8]
        _args.loss = "sraa"
        _args.rot_mag = 0
        _args.partial = [0.65, 1]
        _args.lamb = 0.001
        _logger.info('---------setting-------: rot_mag -->{}, partial-->[{},{}]'.format(_args.rot_mag, _args.partial[0], _args.partial[1]))
        if _args.method_type == "dgcnn" or _args.method_type == "rpmnet":
            loss_type = ['sraa', 'normal' ]
            for type in loss_type:
                _args.loss = type
                main()
        else:
            main()



