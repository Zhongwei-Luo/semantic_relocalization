import numpy as np
import os
import quaternion
from tools import rigid_transform, show_pc, points_distance,show_pair_pc,downsample
from scipy.spatial.transform import Rotation
from voxnet_process import voxnet
from rpmnet_process import rpmnet
from sklearn.neighbors import NearestNeighbors
from emicp import global_em_icp
#import xlrd
from tools import rotationMatrixToEulerAngles, eulerAnglesToRotationMatrix, show_pair_pc, rigid_transform, downsample, points_denoise
import math
from generate_model_set import *
def random_transform():
    rand_rot_deg = np.random.random() * 180
    rand_rot = Rotation.from_euler('y', rand_rot_deg, degrees=True).as_dcm()
    rand_SE3 = np.pad(rand_rot, ((0, 0), (0, 1)), mode='constant').astype(np.float32)
    rand_SE3 = np.concatenate([rand_SE3, np.array([[0, 0, 0, 1]])], axis=0)
    return rand_SE3

def show_map(map,color_set):
    t = np.max(map["object_list"][0], axis = 0)/5
    t[1] = 0
    pt = []
    color = []
    for i, object in enumerate(map["object_list"]):
        if map["label_list"][i] > 37 or map["label_list"][i] <= 2:
            pt.append(object + np.array([4, 0, 4]))
            color.append(color_set[0, :].reshape(1, 3) * np.ones_like(object))
        else:
            pt.append(object + np.array([4, 0, 4]))
            color.append(color_set[i, :].reshape(1, 3) * np.ones_like(object))
    map_pt = np.concatenate(pt, axis=0)
    map_color = np.concatenate(color, axis=0)

    for j in range(4):
        object = projection(map_pt, j * 2 * math.pi / 4)
        idx = np.where((object[:, 0] > 0) & (object[:, 2] > 0))
        object = object[idx[0], :]
        color = map_color[idx[0], :]
        show_pc(object, color)
def projection(pt, angle, t=0):
    rotm = eulerAnglesToRotationMatrix([0, angle, 0])
    pt = pt.dot(rotm.transpose())
    t =  np.min(pt, axis=0)
    t[1] = 0
    pt -= t
    return pt


if __name__ == "__main__":
    model = voxnet("/home/lzw/desktop/new_project/code_21_7_10/model/model0/model_000000_sraa_0.001_voxnet_8_0_65.0-37296.pth")
    model_classes = excel2dict()
    scene_model_path = "/home/lzw/desktop/new_project/data_21_7_10/scene_model"
    #generate_model_set(model, model_classes, scene_model_path)
    # all_cad_model_set = np.load("./model_set.npy", allow_pickle=True).item()
    color_set = np.loadtxt("/home/lzw/desktop/new_project/color_label/new_color.txt")/255
    id_scan = 'scene0122_00'
    id_scan_list = os.listdir(scene_model_path)
    for id_scan in id_scan_list:
        print(id_scan)
        # if not id_scan=='scene0223_00':
            # continue
        id_scan_path = os.path.join(scene_model_path, id_scan)
        map_path = os.path.join(id_scan_path, 'map') + '/' + id_scan + '.npy'
        object_model_path = os.path.join(id_scan_path, 'object_model')
        object_model_list = os.listdir(object_model_path)
        if not os.path.exists(map_path):
            continue
        map = np.load(map_path, allow_pickle=True).item()
        # show_map(map, color_set)
        ###obtain the models and objects from the scene map
        if 1:
            em_icp = global_em_icp(18)
            pt = []
            color = []
            object_set = []
            model_points = []
            model_feat = []
            object_feat = []
            errors = np.zeros((len(object_model_list),2))
            for i in range(len(object_model_list)):
                pair_name = os.path.join(object_model_path, object_model_list[i])
                pair = np.load(pair_name, allow_pickle=True).item()
                cate_id = model_classes[pair_name.split('/')[-1].split('_')[-3]]
                cad_model = pair["model"]
                object = pair["object"]

                t = pair["trs"]["translation"]
                q = pair["trs"]["rotation"]
                s = pair["trs"]["scale"]

                q = np.quaternion(q[0], q[1], q[2], q[3])
                T = np.eye(4)
                T[0:3, 3] = t
                R = np.eye(4)
                R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
                T = T.dot(R)


                # show_pair_pc(object, cad_model)
                object_trans = rigid_transform(object, np.linalg.inv(T))
                cad_model_trans = rigid_transform(cad_model, np.linalg.inv(T))
                random_trans = random_transform()

                # show_pair_pc(object_trans, cad_model_trans)

                errors[i, 0] = points_distance(cad_model_trans, object_trans)
                feat = model.run(cad_model_trans, 0) + cate_id*10
                model_feat.append(feat)
                model_points.append(cad_model)

                feat = model.run(rigid_transform(object_trans, random_trans), 0) + cate_id*10
                object_feat.append(feat)
                object_set.append(object)

                pt.append(object + np.array([0, 0, 0]))
                color.append(color_set[i, :].reshape(1, 3)*np.ones_like(object))

            if len(object_feat)==0:
                continue
            object_feat = np.concatenate(object_feat, axis=0)
            # if 0:   ### all the cad model
            #     model_feat = [all_cad_model_set[key][0] for key in all_cad_model_set.keys()]
            #     model_points = [all_cad_model_set[key][1] for key in all_cad_model_set.keys()]
            #     model_feat = np.concatenate(model_feat, axis=0)

            if 1:   ### model for the current scene
                model_feat = np.concatenate(model_feat, axis=0)

            ###search the corresponding cad model
            nbrs1 = NearestNeighbors(algorithm='kd_tree', n_neighbors=1).fit(model_feat)
            distance1, indices = nbrs1.kneighbors(object_feat)

            ###registration  for the cad_model and object
            for i in range(indices.shape[0]):
                print(i)
                cad_model = model_points[indices[i, 0]]
                object = object_set[i]
                # show_pair_pc(cad_model, object)
                trans, errors= em_icp.run(object, cad_model)
                trans = np.linalg.inv(trans)
                cad_model = rigid_transform(cad_model, trans)
                # errors[i, 1] = points_distance(object, cad_model)
                # show_pair_pc(cad_model, object)
                pt.append(cad_model + np.array([0, 0, 0]))
                color.append(color_set[i, :].reshape(1, 3)*np.ones_like(cad_model))

            if len(pt) == 0:
                continue
            map_pt = np.concatenate(pt, axis=0)
            map_color = np.concatenate(color, axis=0)

            show_pc(map_pt, map_color)
            print("a")


def load_model_set():
    model_set = np.load("/media/lan/Samsung_T5/scan2cad/data/model_set.npy", allow_pickle=True).item()
    return model_set





