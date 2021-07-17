"""Feature Extraction and Parameter Prediction networks
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet_util import sample_and_group_multi
import time
import open3d as o3d
_raw_features_sizes = {'xyz': 3, 'dxyz': 3, 'ppf': 4}
_raw_features_order = {'xyz': 0, 'dxyz': 1, 'ppf': 2}
from collections import OrderedDict


class FeatExtraction1(nn.Module):
    """Feature extraction Module that extracts hybrid features"""
    def __init__(self,  feature_dim=96, radius=0.3, num_neighbors=64, task='cls',features=[ 'ppf', 'dxyz', 'xyz'], nclass=128):
        super().__init__()
        self.task = task
        self.radius = radius
        self.n_sample = num_neighbors
        self.feature_dim = feature_dim
        self.features = sorted(features, key=lambda f: _raw_features_order[f])
        raw_dim = np.sum([_raw_features_sizes[f] for f in self.features])
        self.prepool = self.get_prepool(raw_dim, feature_dim * 4)
        self.postpool = self.get_postpool(feature_dim * 4, feature_dim*2)
        self.global_postpool = self.get_global_postpool(feature_dim*2, nclass)

    def get_prepool(self, in_dim, out_dim):
        """Shared FC part in PointNet before max pooling"""
        net = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.2),

            nn.Conv2d(out_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_dim, out_dim, 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        return net

    def get_postpool(self, in_dim, out_dim):
        """Linear layers in PointNet after max pooling
        """
        net = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, 1),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_dim, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),

            nn.Conv1d(out_dim, out_dim, 1),
        )
        return net

    def get_global_postpool(self, in_dim, out_dim):
        net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),

            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),

            nn.Linear(out_dim, out_dim),
        )
        return net
    def forward(self, xyz, normals, class_vec=None):
        """Forward pass of the feature extraction network
        Args:
            xyz: (B, N, 3)
            normals: (B, N, 3)
        Returns:
            cluster features (B, N, C)
        """
        features = sample_and_group_multi(self.radius, self.n_sample, xyz, normals)
        features['xyz'] = features['xyz'][:, :, None, :]
        # Gate and concat
        concat = []
        for i in range(len(self.features)):
            f = self.features[i]
            expanded = (features[f]).expand(-1, -1, self.n_sample, -1)
            concat.append(expanded)
        features = torch.cat(concat, -1)
        features = features.permute(0, 3, 2, 1)  # [B, 10, n_sample, N]
        features = self.prepool(features)

        point_feat = torch.max(features, 2)[0]  # Max pooling (B, C, N)

        point_feat = self.postpool(point_feat)   # Post pooling dense layers
        # point_feat = point_feat.permute(0, 2, 1)

        # x = torch.max(point_feat, 1, keepdim=True)[0]

        glob_feat = torch.max(point_feat, 2)[0]
        # glob_feat = self.global_postpool(glob_feat)
        point_feat = point_feat.permute(0, 2, 1)


        return glob_feat.detach().numpy() , point_feat.detach().numpy()  # (B, N, C)


class rpmnet():
    def __init__(self, save_path):
        self.model = FeatExtraction1()
        state = torch.load(save_path, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        if 'state_dict' in state and self.model is not None:
            for keys, v in state['state_dict'].items():
                keys = keys[22:]
                new_state_dict[keys] = v
            self.model.load_state_dict(new_state_dict)
        self.model.eval()
    def sample(self, ptc):
        if self.cad:
            ref_size = 1024
        else:
            ref_size = 666
        rand_idxs = self._resample(ptc, ref_size)
        return ptc[rand_idxs, :], rand_idxs

    @staticmethod
    def _resample(points, k):
        """Resamples the points such that there is exactly k points.

        If the input point cloud has <= k points, it is guaranteed the
        resampled point cloud contains every point in the input.
        If the input point cloud has > k points, it is guaranteed the
        resampled point cloud does not contain repeated point.
        """
        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return rand_idxs
        else:
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
            return rand_idxs
    def normal_estimation(self, ptc):
        points_src = ptc
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_src)
        o3d.estimate_normals(pcd,
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.25, max_nn=400))
        return np.array(pcd.normals)
    def normaliza_model(self, model):
        v_mean = np.mean(model, axis=0)
        model -= v_mean
        scale = np.max(np.linalg.norm(model - np.min(model, axis=0), axis=1)) / 2
        model /= scale
        return model
    def data_process(self, ptc):
        ptc = self.normaliza_model(ptc)
        normal = self.normal_estimation(ptc)
        ptc = np.concatenate((ptc, normal), axis=1)
        ptc, ptc_idx = self.sample(ptc)
        return ptc[:, :3], ptc[:, 3:], ptc_idx

    def run(self, ptc, cad = False):
        self.cad = cad
        pt, normal, ptc_idx = self.data_process(ptc)
        pt = pt[np.newaxis, :]
        normal = normal[np.newaxis, :]

        glob_feat, point_feat = self.model(torch.from_numpy(pt).type(torch.FloatTensor), torch.from_numpy(normal).type(torch.FloatTensor))
        return ptc_idx,  np.squeeze(point_feat), glob_feat



if __name__ == '__main__':


    model = rpmnet("/home/lan/Desktop/weijx_luozhw/network/code/model/rpmnet_model0/model_000000_sraa_0.001_rpmnet_8_0_65.0-179655.pth" )
    xyz = np.random.randn(5524, 3)

    src_embedding = model.run(xyz)
    print(src_embedding[0].shape, src_embedding[1].shape,src_embedding[2].shape)