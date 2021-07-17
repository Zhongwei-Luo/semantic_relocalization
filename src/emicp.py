import numpy as np
from scipy.optimize import least_squares
from tools import rotationMatrixToEulerAngles, eulerAnglesToRotationMatrix, show_pair_pc, rigid_transform, downsample, points_denoise
from sklearn.neighbors import NearestNeighbors
import math
import concurrent.futures
import open3d as o3d
import scipy.io as scio
class robust_icp():
    def __init__(self):
        self.delta = 0.02
        self.rotm_axis = "y"

    def em_icp(self, source, target):
        s = 1
        if 0:
            v = np.zeros(4)
            res_1 = least_squares(self.cost_function, v, args=(source, target))
            v = res_1.x
            error = res_1.cost
            R = self.generate_rotm(v[0])
            t = v[1:4]
            T = np.identity(4)
            T[:3, :3] = R
            T[:3, 3] = t

            src = rigid_transform(source, T)
            v = np.zeros(4)
            v[:1] = 1.0
            res_1 = least_squares(self.scale_funtion, v, args=(src, target))
            error = res_1.cost
            x = res_1.x
            t_new = x[1:]
            s = x[:1]
            T[:3, 3] = t + t_new

        if 1:
            R, t, s, error = self.iterative(source, target)
            T = np.identity(4)
            T[:3, :3] = R
            T[:3, 3] = t

        return T, error, s
    def SVD_T(self, source, target):
        mean_s = np.mean(source, axis=0)
        mean_t = np.mean(target, axis=0)
        source_m = source - mean_s
        target_m = target - mean_t
        S = target_m.transpose().dot(source_m)
        U, A, V = np.linalg.svd(S)
        R = U.dot(V)

        if np.linalg.det(R)<0:
            R = -1*R
        src = source.dot(R.transpose())
        S = np.sum(src*src)
        Q = np.sum(src*target)
        c = np.sum(src, axis=0)
        d = np.sum(target, axis=0)
        A = target.shape[0]*np.eye(4)
        A[0, 0] = S
        A[0, 1:] = c
        A[1:, 0] = c
        b = np.zeros(4)
        b[0] = Q
        b[1:] = d
        x = np.linalg.inv(A).dot(b)
        s = x[0]
        t = x[1:]
        s=1
        return R, t, s
    def SVD_T_1(self, source, target):
        mean_s = np.mean(source, axis=0)
        mean_t = np.mean(target, axis=0)
        source_m = source - mean_s
        target_m = target - mean_t
        s = self.estimate_scale(source_m, target_m)
        source_m[:, 1] = 0
        target_m[:, 1] = 0
        S = target_m.transpose().dot(source_m.dot(np.diag(s)))
        U, A, V = np.linalg.svd(S)
        R = U.dot(V)

        if np.linalg.det(R)<0:
            R = -1*R
        t = mean_t - np.diag(s).dot(R.dot(mean_s))
        return R, t, s
    def estimate_scale(self, src, tar):
        s_max = [2,  2,  2]
        s_min = [0.9, 0.9, 0.9]
        s = np.ones(3)
        for i in range(3):
            temp = np.zeros_like(src)
            temp[:, i] = src[:, i]
            ss = np.sum(np.sum(tar * temp, axis=1)) / np.sum(np.sum(temp * temp, axis=1))
            if ss > s_max[i]:
                ss = s_max[i]
            if ss < s_min[i]:
                ss = s_min[i]
            s[i] = ss
        return s
    def first_k(self, dist, k=0.95):
        idx=np.argsort(dist[:,0])
        length = int(dist.shape[0]*k)
        idx = idx[:length]
        return idx
    def iterative(self, source, target):
        R = np.eye(3)
        t = np.zeros(3)
        s = 1
        for iter in range(40):
            src = source.dot(R.transpose()) + t
            tar_scale = np.max(np.linalg.norm(target - np.min(target, axis=0), axis=1))
            src_scale = np.max(np.linalg.norm(src - np.min(src, axis=0), axis=1))

            ratio = tar_scale / src_scale
            src *= ratio
            nbrs1 = NearestNeighbors(algorithm='kd_tree', n_neighbors=1).fit(target)
            distance1, indices = nbrs1.kneighbors(src)

            nbrs2 = NearestNeighbors(algorithm='kd_tree', n_neighbors=1).fit(src)
            distance2, indices2 = nbrs2.kneighbors(target)

            tar = target[indices[:, 0], :]
            idx = self.first_k(distance1, k=0.95)
            tar0 = tar[idx, :]
            src0 = source[idx, :]

            src = source[indices[:, 0], :]
            idx = self.first_k(distance2, k=0.8)
            src1 = src[idx, :]
            tar1 = target[idx, :]

            src0 = np.concatenate((src1, src0))
            tar0 = np.concatenate((tar1, tar0))

            R, t, s = self.SVD_T_1(src0, tar0)

        src = (s * source.dot(R.transpose()) + t)
        error = self._distance(src, target, 1)
        # show_pair_pc(src, target)
        return R, t, s, error

    def _distance(self, source, target, k=0.9):
        nbrs1 = NearestNeighbors(algorithm='kd_tree', n_neighbors=1).fit(source)
        distance1, _ = nbrs1.kneighbors(target)
        errors = np.mean(self.huber_loss(distance1))

        nbrs1 = NearestNeighbors(algorithm='kd_tree', n_neighbors=1).fit(target)
        distance1, _ = nbrs1.kneighbors(source)
        idx = self.first_k(distance1, k)
        errors += np.mean(distance1[idx, 0])
        return errors
    def cost_function(self, x, *args):
        source = args[0]
        target = args[1]
        R = self.generate_rotm(x[0])
        t = x[1:]
        # s = x[4:]

        src = source.dot(R.transpose()) + t
        # src = R.dot(source.transpose()).transpose() + t

        nbrs1 = NearestNeighbors(algorithm='kd_tree', n_neighbors=1).fit(target)
        distance1, indices = nbrs1.kneighbors(src)

        # tar = target[indices[:,0],:]
        # src = src[distance1[:,0]<self.delta ]
        # tar = tar[distance1[:,0]<self.delta ]
        # s = self.
        # error = np.linalg.norm(src-tar, axis=0)
        errors = self.huber_loss(distance1[:,0])


        return errors

    def registration(self, x, *args):
        source = args[0]
        target = args[1]
        R = eulerAnglesToRotationMatrix([x[0], x[1], x[2]])
        t = x[3:6]
        src = source.dot(R.transpose()) + t
        nbrs1 = NearestNeighbors(algorithm='kd_tree', n_neighbors=1).fit(target)
        distance1, indices = nbrs1.kneighbors(src)
        errors = self.huber_loss(distance1[:,0])
        return errors

    def scale_funtion(self, x, *args):
        source = args[0]
        target = args[1]
        t = x[1:]
        s = x[:1]
        src = s*source + t
        nbrs1 = NearestNeighbors(algorithm='kd_tree', n_neighbors=1).fit(src)
        distance1, indices = nbrs1.kneighbors(target)
        errors = self.huber_loss(distance1[:,0])
        # nbrs1 = NearestNeighbors(algorithm='kd_tree', n_neighbors=1).fit(target)
        # distance1, indices = nbrs1.kneighbors(src)
        # errors = np.concatenate((self.huber_loss(distance1[:,0]),errors), axis=0)
        return errors

    def huber_loss(self, errors):
        delta = 0.04
        temp = errors
        errors[temp <= delta] = np.sqrt(0.5) * errors[temp <= delta]
        errors[temp > delta] = np.sqrt(delta * errors[temp > delta] - 0.5 * delta * delta)
        return errors
    def generate_rotm(self, rot_deg):
        if self.rotm_axis == "x":
            rotm = eulerAnglesToRotationMatrix([rot_deg, 0, 0])
        elif self.rotm_axis == "y":
            rotm = eulerAnglesToRotationMatrix([0, rot_deg, 0])
        else:
            rotm = eulerAnglesToRotationMatrix([0, 0, rot_deg])
        return rotm

class global_em_icp(robust_icp):
    def __init__(self, steps, rotm_axis="y"):
        """input
         --steps: divide the axis into how many bins
         --rotm_axis: the rotation ia along which axis
         output
         --trans: transformation from the source to target
         --error: registration error
        """
        gaps = 2*math.pi/steps
        self.rot_deg_list = [gaps*i + gaps/2 for i in range(steps)]
        self.rotm_axis = rotm_axis
    def _resample(self, points, k=400):

        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs, :]
        elif points.shape[0] == k:
            return points
        else:
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
            return points[rand_idxs, :]

    def downsample(self, ptc, voxel_size):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ptc[:, :3])
        if ptc.shape[1] == 6:
            pcd.normals = o3d.utility.Vector3dVector(ptc[:, 3:])
            downpcd = pcd.voxel_down_sample(voxel_size)
            ptc = np.array(downpcd.points)
            normals = np.array(downpcd.normals)
            return np.concatenate((ptc, normals), axis=1)
        else:
            downpcd = o3d.voxel_down_sample(pcd, voxel_size)
            ptc = np.array(downpcd.points)
            return ptc

    def run(self, source, target):


        self.source_down = self._resample(source)
        self.target_down = self._resample(target)
        self.tar_scale =  np.max(np.linalg.norm(self.target_down -np.min(self.target_down, axis=0), axis=1))
        # self.source = self.source
        # self.target  = self.target

        dist_error = []
        T_set = []
        s_set = []

        executor = concurrent.futures.ProcessPoolExecutor(max_workers=6)
        for rot_deg, result in zip(self.rot_deg_list, executor.map(self.multi_process, self.rot_deg_list)):
            T_set.append(result[0])
            dist_error.append(result[1])
            # s_set.append(result[2])
        executor.shutdown(wait=True)

        # for i in range(len(self.rot_deg_list)):
        #     result = self.multi_process(self.rot_deg_list[i])
        #     T_set.append(result[0])
        #     dist_error.append(result[1])
        #     s_set.append(result[2])

        idx = np.argmin(dist_error)
        trans = T_set[idx]
        error = dist_error[idx]
        # show_pair_pc(rigid_transform(self.source_down,trans), self.target_down)
        # s = s_set[idx]


        return trans, error


    def multi_process(self, rot_deg):

        tar_mean = np.mean(self.target_down, axis=0)
        tar = self.target_down - tar_mean
        tar_scale = np.max(np.linalg.norm(tar -np.min(tar, axis=0), axis=1)) #[:,[0,2]]
        # tar_scale = np.max(np.sum((tar[:,[0,2]] -np.min(tar[:,[0,2]] , axis=0))*(tar[:,[0,2]] -np.min(tar[:,[0,2]] , axis=0)) , axis=1))

        rotm = self.generate_rotm(rot_deg)
        src = self.source_down.dot(rotm.transpose())
        src_mean = np.mean(src, axis=0)
        src = src - src_mean

        src_scale = np.max(np.linalg.norm(src  - np.min(src, axis=0), axis=1))
        # src_scale = np.max(np.sum((src[:,[0,2]]  - np.min(src[:,[0,2]] , axis=0))*(src[:,[0,2]]  - np.min(src[:,[0,2]] , axis=0))  , axis=1))

        ratio = tar_scale / src_scale
        src = src * ratio


        trans, error, s = self.em_icp(src, tar)
        # show_pair_pc(rigid_transform(s*src, trans), tar)
        s *= ratio
        trans[:3,  3] = -(trans[:3, :3]*s).dot(src_mean) + trans[:3, 3] + tar_mean
        trans[:3, :3] = (trans[:3, :3]*s).dot(rotm)
        # show_pair_pc(rigid_transform(self.source_down, trans), self.target_down)

        return (trans, error, s)



if __name__=="__main__":
    source = np.random.random((1024, 3))
    R = eulerAnglesToRotationMatrix([0, 3.14/2, 0])
    target = source.dot(R.transpose())
    em_icp = global_em_icp(12)
    trans, error = em_icp.run(source, target)
    show_pair_pc(target, rigid_transform(source, trans) + np.array([0,0.1,1]))

