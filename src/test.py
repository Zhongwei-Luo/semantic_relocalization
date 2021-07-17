import numpy as np
from sklearn.neighbors import NearestNeighbors
from tools import rigid_transform, show_pc, points_distance,show_pair_pc,downsample
import os
class voxnet():
    def __init__(self,  save_path=None):

        self.range = 2.4
        self.voxel_shape = 32
        self.voxel_size = self.range / self.voxel_shape
        self.voxel_margin = 3 * self.voxel_size
        self.x_range = [-self.range / 2, self.range / 2]
        self.y_range = [-self.range / 2, self.range / 2]
        self.z_range = [-self.range / 2, self.range / 2]
    def pcd2tdf(self, pcd):
        xv, yv, zv = np.meshgrid(
            np.linspace(self.x_range[0] + self.voxel_size / 2, self.x_range[1] - self.voxel_size / 2, self.voxel_shape),
            np.linspace(self.y_range[0] + self.voxel_size / 2, self.y_range[1] - self.voxel_size / 2, self.voxel_shape),
            np.linspace(self.z_range[0] + self.voxel_size / 2, self.z_range[1] - self.voxel_size / 2, self.voxel_shape))
        xv = np.asarray(xv.flat).reshape(-1, 1)
        yv = np.asarray(yv.flat).reshape(-1, 1)
        zv = np.asarray(zv.flat).reshape(-1, 1)
        grid = np.concatenate([yv, xv, zv], axis=1)

        knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(pcd)
        dist, _ = knn.kneighbors(X=grid, return_distance=True)

        df = dist.reshape(self.voxel_shape, self.voxel_shape, self.voxel_shape)

        tdf = 1.0 - np.where(df / (self.voxel_size * 0.9) < 1.0, 0, 1.0)
        return tdf[np.newaxis, :]
    def vox2pcd(self,vox):
        id = np.where(vox>0)
        pcd = np.concatenate((id[1].reshape(-1,1),id[2].reshape(-1,1),id[3].reshape(-1,1)), axis=1)
        return pcd
    def normaliza_model(self, model):
        v_mean = np.mean(model, axis=0)
        model -= v_mean
        scale = np.max(np.linalg.norm(model - np.min(model, axis=0), axis=1)) / 2
        model /= scale
        return model
    def run(self, ptc, class_id=0):

        ptc = ptc[:,:3]
        ptc = self.normaliza_model(ptc)
        vox = self.pcd2tdf(ptc[:, :3])
        pcd = self.vox2pcd(vox)
        return pcd



if __name__ == '__main__':
    data_path = "/media/lan/Samsung_T5/scan2cad/data/object_model/03001627/"
    list_data_path = os.listdir(data_path)
    for f in list_data_path:
        cad2_data_path = os.path.join(data_path,f)
        # cad2_data_path = '/media/lan/Samsung_T5/scan2cad/data/object_model/02747177/scene0000_00_02747177_93d5f21e94ac5fb68a606c63493ddb7c_4.npy'
        cad2_data = np.load(cad2_data_path, allow_pickle=True).item()
        cad2_model = cad2_data["model"][:, :3]
        model = voxnet()
        target = model.run(cad2_data["model"][:, :3])
        surce = model.run(cad2_data["object"][:, :3])
        show_pair_pc(surce, target)



