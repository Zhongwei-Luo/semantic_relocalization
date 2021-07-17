import os
import numpy as np
import torch
import torch.nn as nn
from common.torch import to_numpy
from torch.autograd import Variable
from collections import OrderedDict
from sklearn.neighbors import NearestNeighbors
from tools import *
class VoxNet(nn.Module):
    def __init__(self,  n_classes=128, input_shape=(32, 32, 32), ):
        super(VoxNet, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.feat = torch.nn.Sequential(OrderedDict([
            ('conv3d_1', torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2)),
            ('relu1', torch.nn.ReLU()),
            ('drop1', torch.nn.Dropout(p=0.2)),

            ('conv3d_3', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
            ('relu3', torch.nn.ReLU()),
            ('pool3', torch.nn.MaxPool3d(2)),
            # ('drop3', torch.nn.Dropout(p=0.3))
        ]))
        x = self.feat(torch.autograd.Variable(torch.rand((1, 1) + input_shape)))
        dim_feat = 1
        for n in x.size()[1:]:
            dim_feat *= n

        self.mlp = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(dim_feat, 512)),
            ('relu1', torch.nn.ReLU()),
            # ('drop3', torch.nn.Dropout(p=0.1)),
            ('fc2', torch.nn.Linear(512, self.n_classes))
        ]))

    def forward(self, x, classes_vec=None):
        x = self.feat(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x
class VoxNet1(nn.Module):
    def __init__(self,  n_classes=128, input_shape=(32, 32, 32), ):
        super(VoxNet1, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.feat = torch.nn.Sequential(OrderedDict([
            ('conv3d_1', torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2)),
            ('relu1', torch.nn.ReLU()),
            ('drop1', torch.nn.Dropout(p=0.2)),

            ('conv3d_3', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
            ('relu3', torch.nn.ReLU()),
            ('pool3', torch.nn.MaxPool3d(2)),
            # ('drop3', torch.nn.Dropout(p=0.3))
        ]))
        x = self.feat(torch.autograd.Variable(torch.rand((1, 1) + input_shape)))
        dim_feat = 1
        for n in x.size()[1:]:
            dim_feat *= n

        self.mlp = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(dim_feat, 128)),
            ('relu1', torch.nn.ReLU()),
            # ('drop3', torch.nn.Dropout(p=0.1)),
            ('fc2', torch.nn.Linear(128, self.n_classes))
        ]))
        self.mlp1 = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(self.n_classes*2, 128)),
            ('relu1', torch.nn.ReLU()),
            # ('drop3', torch.nn.Dropout(p=0.1)),
            ('fc2', torch.nn.Linear(128, self.n_classes))
        ]))
        self.class_encoder = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(40, 80)),
            ('relu1', torch.nn.ReLU()),
            # ('drop3', torch.nn.Dropout(p=0.1)),
            ('fc2', torch.nn.Linear(80, self.n_classes))
        ]))

    def forward(self, x, classes_feat):
        x = self.feat(x)
        x = x.view(x.size(0), -1)
        obj_feat = self.mlp(x)
        x = torch.cat((obj_feat, self.class_encoder(classes_feat)), dim=1)
        x = self.mlp1(x)
        return obj_feat, x
class VoxNet2(nn.Module):
    def __init__(self,  n_classes=128, input_shape=(32, 32, 32)):
        super(VoxNet2, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.feat = torch.nn.Sequential(OrderedDict([
            ('conv3d_1', torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=7, stride=2)),
            ('relu1', torch.nn.ReLU()),
            ('drop1', torch.nn.Dropout(p=0.1)),

            ('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=5, stride=2)),
            ('relu2', torch.nn.ReLU()),
            ('drop2', torch.nn.Dropout(p=0.1)),
            # ('pool3', torch.nn.MaxPool3d(2)),

            ('conv3d_3', torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1)),
            ('relu3', torch.nn.ReLU()),
            # ('pool3', torch.nn.MaxPool3d(2)),
            ('drop3', torch.nn.Dropout(p=0.1)),

            ('conv3d_4', torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3)),
            ('relu4', torch.nn.ReLU()),

            ('conv3d_5', torch.nn.Conv3d(in_channels=256, out_channels=512, kernel_size=1)),
            ('relu5', torch.nn.ReLU()),


        ]))
        x = self.feat(torch.autograd.Variable(torch.rand((1, 1) + input_shape)))
        dim_feat = 1
        for n in x.size()[1:]:
            dim_feat *= n
        print(dim_feat)
        self.mlp = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(dim_feat, 1024)),
            ('relu1', torch.nn.ReLU()),
            # ('drop3', torch.nn.Dropout(p=0.1)),
            ('fc2', torch.nn.Linear(1024, self.n_classes))
        ]))


    def forward(self, x, classes_feat=None):
        x = self.feat(x)
        # print(x.size())
        x = x.view(x.size(0), -1)
        obj_feat = self.mlp(x)

        return obj_feat

class VoxNet3(nn.Module):
    def __init__(self,  n_classes=128, input_shape=(32, 32, 32)):
        super(VoxNet3, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.feat = torch.nn.Sequential(OrderedDict([
            ('conv3d_1', torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=5, stride=2)),
            # ('bn3d_1', torch.nn.BatchNorm3d(32)),
            ('relu1', torch.nn.ReLU()),
            ('drop1', torch.nn.Dropout(p=0.1)),

            ('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=2)),
            # ('bn3d_2', torch.nn.BatchNorm3d(64)),
            ('relu2', torch.nn.ReLU()),
            # ('drop2', torch.nn.Dropout(p=0.1)),
            # ('pool3', torch.nn.MaxPool3d(2)),

            ('conv3d_3', torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1)),
            # ('bn3d_3', torch.nn.BatchNorm3d(128)),
            ('relu3', torch.nn.ReLU()),
            # ('pool3', torch.nn.MaxPool3d(2)),
            # ('drop3', torch.nn.Dropout(p=0.1)),

            ('conv3d_4', torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3)),
            # ('bn3d_4', torch.nn.BatchNorm3d(256)),
            ('relu4', torch.nn.ReLU()),

            ('conv3d_5', torch.nn.Conv3d(in_channels=256, out_channels=512, kernel_size=1)),
            # ('pool5', torch.nn.MaxPool3d(2)),
            # ('relu5', torch.nn.ReLU()),

        ]))
        x = self.feat(torch.autograd.Variable(torch.rand((1, 1) + input_shape)))
        dim_feat = 1
        for n in x.size()[1:]:
            dim_feat *= n
        print(dim_feat)
        self.mlp = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(dim_feat, 512)),
            # ('bn1', torch.nn.BatchNorm1d(num_features=512)),
            ('relu1', torch.nn.ReLU()),
            # ('drop3', torch.nn.Dropout(p=0.1)),
            ('fc2', torch.nn.Linear(512, self.n_classes))
        ]))
        # self.mlp1 = torch.nn.Sequential(OrderedDict([
        #     ('fc1', torch.nn.Linear(2*self.n_classes, self.n_classes)),
        #     ('bn1', torch.nn.BatchNorm1d(self.n_classes)),
        #     ('relu1', torch.nn.ReLU()),
        #     # ('drop3', torch.nn.Dropout(p=0.1)),
        #     ('fc2', torch.nn.Linear(self.n_classes, self.n_classes))
        # ]))
        # self.class_encoder = torch.nn.Sequential(OrderedDict([
        #     ('fc1', torch.nn.Linear(40, 80)),
        #     ('relu1', torch.nn.ReLU()),
        # #     # ('drop3', torch.nn.Dropout(p=0.1)),
        #     ('fc2', torch.nn.Linear(80, self.n_classes))
        # ]))

class voxnet():
    def __init__(self,  save_path):
        self.model = VoxNet()
        state = torch.load(save_path, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        if 'state_dict' in state and self.model is not None:
            for keys, v in state['state_dict'].items():
                keys = keys[22:]
                new_state_dict[keys] = v
            self.model.load_state_dict(new_state_dict)
        self.model.eval()
        self.range = 2.2
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
    def normaliza_model(self, model):
        v_mean = np.mean(model, axis=0)
        model -= v_mean
        scale = np.max(np.linalg.norm(model - np.min(model, axis=0), axis=1)) / 2
        model /= scale
        return model
    def run(self, ptc, class_id=0):

        ptc = self.normaliza_model(ptc[:,:3])

        vox = self.pcd2tdf(ptc[:, :3])
        vox = torch.from_numpy(vox).unsqueeze(0).type(torch.FloatTensor)
        if class_id > 0:
            classes_vec = np.zeros((1, 40))
            classes_vec[0, int(class_id)] = 1
            classes_vec = torch.from_numpy(classes_vec).type(torch.FloatTensor)
            obj_feat, glob_feat = self.model(vox, classes_vec)
        else:
            glob_feat = self.model(vox)
        return glob_feat.detach().numpy()   # 1*96

def generate_data_txt(object_model_path, cat_id=None):
    save_path = "./data_loader/"

    if cat_id == None:
        cat_id_list = os.listdir(object_model_path)
        train_f = open(os.path.join(save_path, 'train.txt'), "w")
        validation_f = open(os.path.join(save_path, 'validation.txt'), "w")
        for cat_id in cat_id_list:
            if cat_id == "04379243":
                continue
            cat_id_path = os.path.join(object_model_path, cat_id)
            file_list = os.listdir(cat_id_path)

            for obj in file_list:
                file_name = os.path.join(cat_id_path, obj)
                if np.random.random() > 0.02:
                    train_f.write(file_name)
                    train_f.write("\n")

                else:
                    validation_f.write(file_name)
                    validation_f.write("\n")

        train_f.close()
        validation_f.close()

    else:
        train_f = open(os.path.join(save_path, 'train.txt'), "w")
        validation_f = open(os.path.join(save_path, 'validation.txt'), "w")
        cat_id_path = os.path.join(object_model_path, cat_id)
        file_list = os.listdir(cat_id_path)

        for obj in file_list:
            file_name = os.path.join(cat_id_path, obj)
            if np.random.random() > 0.04:
                train_f.write(file_name)
                train_f.write("\n")
            else:
                validation_f.write(file_name)
                validation_f.write("\n")

        train_f.close()
        validation_f.close()

def read_data_txt(path):
    name_list = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip('\n')
            name_list.append(line)
    return name_list

def voxnet_test():
    model = voxnet("/home/lzw/desktop/new_project/code_21_7_10/model/model3/model_000000_sraa_0.001_voxnet_8_0_65.0-32016.pth")
    test_name_list = read_data_txt('./data_loader/validation.txt')
    object_point = []
    object_feat = []
    model_point = []
    model_feat = []
    for obj in test_name_list:
        data = np.load(obj, allow_pickle=True).item()

        t = data["trs"]["translation"]
        q = data["trs"]["rotation"]
        s = data["trs"]["scale"]
        Mcad = make_M_from_tqs(t, q, s)
        trans = np.linalg.inv(Mcad)

        cad_model = data["model"][:,:3]
        cad_model = rigid_transform(cad_model, trans)


        object = data["object"][:,:3]
        object = rigid_transform(object, trans)
        object = rigid_transform(object, random_transform())
        glob_feat_src, ptc = model.run(object)
        object_point.append(ptc)
        object_feat.append(glob_feat_src)
        glob_feat_ref, ptc = model.run(cad_model)
        model_point.append(ptc)
        model_feat.append(glob_feat_ref)
    object_feat = np.concatenate(object_feat, axis=0)
    model_feat = np.concatenate(model_feat, axis=0)
    K = 5

    nbrs = NearestNeighbors(n_neighbors=K, algorithm='kd_tree').fit(model_feat)
    distances, indices0 = nbrs.kneighbors(object_feat)
    for i in range(object_feat.shape[0]):
        pt = []
        pt.append(object_point[i])
        for j in range(K):
            pt.append(model_point[indices0[i, j]] + (j+1)*np.array([1, 0, 1]))
            # pt.append(model_point[indices0[i, j]] )
        pt = np.concatenate(pt, axis=0)
        show_pc(pt)

def test_list():
    model = voxnet(
         "/home/lzw/desktop/new_project/code_21_7_10/model/model0/model_000000_sraa_0.001_voxnet_8_0_65.0-best.pth")
    path="/home/lzw/desktop/new_project/data_21_7_10/object_model"
    done_count=0
    succ_write_count=0
    skip_count=0
    for dir_number in os.listdir(path):
        dir_path=path+'/'+dir_number


        npy_list=os.listdir(dir_path)
        print("-------",'\n',dir_path,'\n',"---------------",'\n')
        #npy_list.remove("validation_data")
        #npy_list.remove("train_data")
        #npy_list.remove("vis_data")

        for npy_name in npy_list:

            obj_path=path+'/'+dir_number+'/'+npy_name

            with open("/home/lzw/desktop/new_project/data_21_7_10/error_file.txt", "r") as f:
                error_list = f.readlines()
            if obj_path+"\n" in error_list:#+"\n"

                print('\n',"error file "+obj_path,'\n')
                skip_count=skip_count+1
                print('\n',"skip_count: ",skip_count,'\n')
                continue



            with open("/home/lzw/desktop/new_project/data_21_7_10/done_file_list.txt", "r") as f:
                done_list = f.readlines()
            if obj_path+"\n" in done_list:
                print('\n', "done file " + obj_path, '\n')
                skip_count=skip_count+1
                print('\n',"skip_count: ",skip_count,'\n')
                continue




            print("-------", '\n', "    ",obj_path, '\n', "---------------", '\n')

            data = np.load(obj_path, allow_pickle=True).item()
            # object = data["object"][:,:3]
            # model = data["model"][:,:3]
            # show_pair_pc(object, model)
            glob_feat_obj = model.run(data["object"])
            data["glob_feat_obj"] = glob_feat_obj
            glob_feat_model = model.run(data["model"])
            data["glob_feat_model"] = glob_feat_model
            np.save(obj_path, data)







            print('\n',"done with ",obj_path,'\n')
            done_count=done_count+1
            print('\n',done_count,'\n')
            #mode set to  "a" is very important,it mean append info to the tail of the file
            with open("/home/lzw/desktop/new_project/data_21_7_10/done_file_list.txt", "a") as f:  # 打开文件
                f.write(obj_path)  # 读取文件
                f.write('\n')
                succ_write_count=succ_write_count+1
                print('\n',"succ_write_count: ",succ_write_count,'\n')



def hl_generate_feature():
    # generate_data_txt("/home/lzw/desktop/new_project/data_21_7_10/object_model/", cat_id="03001627")
    # voxnet_test()
    model = voxnet(
        "/home/lzw/desktop/new_project/code_21_7_10/model/model0/model_000000_sraa_0.001_voxnet_8_0_65.0-best.pth")
    # test_name_list = read_data_txt('./data_loader/validation.txt')
    print('successffully init voxel net')
    test_name_list = []

    for obj in test_name_list:
        data = np.load(obj, allow_pickle=True).item()
        # object = data["object"][:,:3]
        # model = data["model"][:,:3]
        # show_pair_pc(object, model)
        glob_feat_obj = model.run(data["object"])
        data["glob_feat_obj"] = glob_feat_obj
        glob_feat_model = model.run(data["model"])
        data["glob_feat_model"] = glob_feat_model
        np.save(obj, data)
        # __import__('ipdb').set_trace()


if __name__ == "__main__":
    test_list()
