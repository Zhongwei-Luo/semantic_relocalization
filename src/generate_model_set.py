import numpy as np
import os
import quaternion
from tools import rigid_transform, show_pc, points_distance,show_pair_pc,downsample
from scipy.spatial.transform import Rotation
from voxnet_process import voxnet
from sklearn.neighbors import NearestNeighbors
from emicp import global_em_icp
import xlrd
def excel2dict():
    filename = "/home/lzw/desktop/new_project/scan2cad/model_classes.xlsx"
    data = xlrd.open_workbook(filename)
    table = data.sheet_by_name('Sheet1')
    row_num = table.nrows  # 行数
    # col_num = table.ncols  # 列数
    model_classes = dict([])  # 这步也要转字典类型
    for i in range(row_num):
        xx = dict([table.row_values(i)])  # 这一步就要给它转字典类型，不然update没法使用
        model_classes.update(xx)
    return model_classes
def make_M_from_tqs(t, q, s):
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    M = T.dot(R).dot(S)
    return M
def generate_model_set(model, model_classes, scene_model_path):
    id_scan_list = os.listdir(scene_model_path)
    model_set = {}
    for id_scan in id_scan_list:
        id_scan_path = os.path.join(scene_model_path, id_scan)
        map_path = os.path.join(id_scan_path, 'map') + '/' + id_scan + '.npy'
        object_model_path = os.path.join(id_scan_path, 'object_model')
        object_model_list = os.listdir(object_model_path)
        if not os.path.exists(map_path):
            continue

        for i in range(len(object_model_list)):
            pair_name = os.path.join(object_model_path, object_model_list[i])
            mod_id = pair_name.split('/')[-1].split('_')[-2]
            if mod_id not in model_set.keys():
                pair = np.load(pair_name, allow_pickle=True).item()
                cate_id = model_classes[pair_name.split('/')[-1].split('_')[-3]]
                cad_model = pair["model"]
                t = pair["trs"]["translation"]
                q = pair["trs"]["rotation"]
                s = pair["trs"]["scale"]

                q = np.quaternion(q[0], q[1], q[2], q[3])
                T = np.eye(4)
                T[0:3, 3] = t
                R = np.eye(4)
                R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
                T = T.dot(R)
                S = np.eye(4)
                S[0:3, 0:3] = np.diag(s)
                M = T.dot(S)
                trans = np.linalg.inv(M)
                # Mcad = make_M_from_tqs(pair["trs"]["translation"], pair["trs"]["rotation"], pair["trs"]["scale"])
                # trans = np.linalg.inv(Mcad)
                cad_model_trans = rigid_transform(cad_model, trans)
                feat = model.run(cad_model_trans, 0)
                feat += 10*cate_id
                model_set[mod_id] = [feat, cad_model_trans, pair_name]

    np.save("/media/lan/Samsung_T5/scan2cad/data/model_set.npy", model_set)

def load_model_set():
    model_set = np.load("./model_set.npy", allow_pickle=True).item()
    return model_set

# def visualization_model_distribution()


