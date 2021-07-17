import sklearn
import numpy as np

import matplotlib.pyplot as plt
from sklearn import manifold,datasets
import os

import seaborn as sns
import pandas as pd

from open3d import*

def draw_for_several_big_class():
    pass

def draw_for_two_big_class():
    n_class_feat = []
    list_color_num = []
    all_obj_feat = []
    all_model_feat = []

    path = "/home/lzw/desktop/new_project/data_21_7_10/object_model"  # object_model

    two_test_cat=['02747177','02773838']
    skip_count = 0
    #for dir_number in os.listdir(path):
    for dir_number in two_test_cat:
        dir_path = path + '/' + dir_number

        npy_list = os.listdir(dir_path)
        print("-------", '\n', dir_path, '\n', "---------------", '\n')

        # if you use new_data/new_path  you need to comment the following three lines:
        # npy_list.remove("validation_data")
        # npy_list.remove("train_data")
        # npy_list.remove("vis_data")

        l_model_glob_feat = []  # in each iteration,this varible store feat of object of one  directory,and then set to null
        l_object_glob_feat = []
        for npy_name in npy_list:

            obj_path = path + '/' + dir_number + '/' + npy_name
            with open("/home/lzw/desktop/new_project/data_21_7_10/error_file.txt", "r") as f:
                error_list = f.readlines()
            if obj_path + "\n" in error_list:  # +"\n"

                print('\n', "error file " + obj_path, '\n')
                skip_count = skip_count + 1
                print('\n', "skip_count: ", skip_count, '\n')
                continue
            print("-------", '\n', obj_path, '\n', "---------------", '\n')
            data = np.load(obj_path, allow_pickle=True).item()
            l_object_glob_feat.append(np.squeeze(data["glob_feat_obj"]))
            print('\n', "l_object_glob_feat length: ", len(l_object_glob_feat), '\n')
            l_model_glob_feat.append(np.squeeze(data["glob_feat_model"]))
            print('\n', "l_model_glob_feat length: ", len(l_model_glob_feat), '\n')


        list_color_num.append(len(l_model_glob_feat))
        # eahc object of one directory will had a number,this is used for plt.scatter 's parameter
        # link feat of object of each directory into one list


        all_model_feat = all_model_feat + l_model_glob_feat

        all_obj_feat = all_obj_feat + l_object_glob_feat

        # to do  :for different class,use different color
        list_color = []  # one class last for a very long sequences,
    # generate color list according to different,the elements which in the same directory will
    # =get the same RGB value
    RGB_value = 0
    for i in range(len(list_color_num)):
        num_count = list_color_num[i]
        for j in range(num_count):
            list_color.append(RGB_value)
        RGB_value = RGB_value + 1

    all_model_feat = np.array(all_model_feat)
    all_obj_feat = np.array(all_obj_feat)

    low_d_feat = get_low_D_feat(all_model_feat)

    plot_two_catgory(low_d_feat, list_color, list_color_num)

    # establish  X with  glob_feat_list

    # X = n_object_glob_feat  # N*96,N number of samples, 96 the dimension of feature
    #
    #
    #
    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    # X_tsne = tsne.fit_transform(X)
    # data_x=X_tsne[:,0]
    # data_y=X_tsne[:,1]

    # plt.plot(data_x,data_y,'o')
    # plt.show()
    # plt.figure(figsize=(10, 10))
    # plt.scatter(data_x, data_y, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    # plt.show()
    # df_subset = pd.DataFrame(X_tsne, columns=2)#create pandans  dataframe
    # df_subset['tsne-2d-one']=X_tsne[:,0]
    # df_subset['tsne-2d-two']=X_tsne[:,1]
    # plt.figure(figsize=(16, 10))
    # sns.scatterplot(
    #     x="tsne-2d-one", y="tsne-2d-two",
    #     hue="y",
    #     palette=sns.color_palette("hls", 10),
    #     data=df_subset,
    #     legend="full",
    #     alpha=0.3
    # )
    #
    #
    #
    # print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))
    #
    # '''嵌入空间可视化'''
    # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    # X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    # plt.figure(figsize=(8, 8))
    # for i in range(50):
    #     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
    #              fontdict={'weight': 'bold', 'size': 12})
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

def demo_for_vis_high_D():

    pass



def draw_del_featrue_for_all_big_class():
    n_class_feat = []
    list_color_num = []
    all_obj_feat = []
    all_model_feat = []

    path = "/home/lzw/desktop/new_project/data_21_7_10/object_model"  # object_model

    #two_test_cat = ['02747177', '02773838']
    skip_count = 0
    dir_list=os.listdir(path)
    #print(dir_list)


    #note that the data we use is subset of shapenet core v2,so dont confuse with the
    #number of the dires.
    for dir_number in os.listdir(path):
    #for dir_number in two_test_cat:
        dir_path = path + '/' + dir_number

        npy_list = os.listdir(dir_path)
        print("-------", '\n', dir_path, '\n', "---------------", '\n')

        # if you use new_data/new_path  you need to comment the following three lines:
        # npy_list.remove("validation_data")
        # npy_list.remove("train_data")
        # npy_list.remove("vis_data")

        l_model_glob_feat = []  # in each iteration,this varible store feat of object of one  directory,and then set to null
        l_object_glob_feat = []
        for npy_name in npy_list:

            obj_path = path + '/' + dir_number + '/' + npy_name
            with open("/home/lzw/desktop/new_project/data_21_7_10/error_file.txt", "r") as f:
                error_list = f.readlines()
            if obj_path + "\n" in error_list:  # +"\n"

                print('\n', "error file " + obj_path, '\n')
                skip_count = skip_count + 1
                print('\n', "skip_count: ", skip_count, '\n')
                continue
            print("-------", '\n', obj_path, '\n', "---------------", '\n')
            data = np.load(obj_path, allow_pickle=True).item()
            l_object_glob_feat.append(np.squeeze(data["glob_feat_obj"]))
            print('\n', "l_object_glob_feat length: ", len(l_object_glob_feat), '\n')
            l_model_glob_feat.append(np.squeeze(data["glob_feat_model"]))
            print('\n', "l_model_glob_feat length: ", len(l_model_glob_feat), '\n')

        list_color_num.append(len(l_model_glob_feat))
        # eahc object of one directory will had a number,this is used for plt.scatter 's parameter
        # link feat of object of each directory into one list

        all_model_feat = all_model_feat + l_model_glob_feat

        all_obj_feat = all_obj_feat + l_object_glob_feat

        # to do  :for different class,use different color
        list_color = []  # one class last for a very long sequences,
    # generate color list according to different,the elements which in the same directory will
    # =get the same RGB value
    RGB_value = 0.1
    for i in range(len(list_color_num)):
        num_count = list_color_num[i]
        for j in range(num_count):
            list_color.append(RGB_value)
        RGB_value = RGB_value + 0.1

    all_model_feat = np.array(all_model_feat)
    all_obj_feat = np.array(all_obj_feat)
    #
    low_d_feat = get_low_D_feat(all_model_feat)


    #in debug mode,we just store the ndarray in hard disk
    #to speed up debug
    #np.save("/home/lzw/desktop/new_project/code_21_7_10/low_d_feature/low_d_feature_7_17.npy", low_d_feat)
    #low_d_feat = np.load("/home/lzw/desktop/new_project/code_21_7_10/low_d_feature/low_d_feature_7_17.npy")




    #we want to draw 2 class in one picture,and then 3 class in one picture
    #and so on
    #draw_num in 2,3,4,...,len(list_color_num)+1-1.
    for draw_num in range(2,len(list_color_num)+1):

        #so once we sure about which number of class we want to draw,we need
        #do it iterable
        # draw_index in 0,1,2,...,draw_num-1.
        #then draw class 0,1,2,...,draw_num-1 in list_color_num.

        list_global_draw_x=[]
        list_global_draw_y=[]
        list_global_draw_c=[]
        for draw_index in range(draw_num):
            #we need to know the start and end in the list_color
            previous_color_num=0
            color_start=0
            color_end=0
            #draw the first class
            if draw_index == 0:
                color_start=0
                color_end=list_color_num[0]-1
            else :
                #it_count_color_num in 0,1,2,...,draw_index-1.
                for it_count_color_num in range(draw_index):
                    color_start=color_start+list_color_num[it_count_color_num]
                color_end=color_start+list_color_num[draw_index]-1

            #the color start,end is directly  index!!!!  not order,order=index+1
            #ause return of  list[a,b] wont include b,so,,,need +1.
            list_draw_c=list_color[color_start:color_end+1]
            # row_index in color_start,3,4,...,color_end+1-1.
            list_draw_x=[]
            list_draw_y=[]
            for row_index in range(color_start,color_end+1):
                list_draw_x.append(low_d_feat[row_index][0])
                list_draw_y.append(low_d_feat[row_index][1])

            list_global_draw_x=list_global_draw_x+list_draw_x
            list_global_draw_y=list_global_draw_y+list_draw_y
            list_global_draw_c=list_global_draw_c+list_draw_c
        plt.scatter(list_global_draw_x,list_global_draw_y, c=list_global_draw_c,s=4 ,alpha=0.7)

        plt.savefig('/home/lzw/desktop/new_project/code_21_7_10/img/draw_'+str(draw_num)+'.jpg')
        #to do :draw vis picture in a inverse order
        #more specificlly,find the id which feature are quite close solely,check carefully




def test_load():
    obj_path=""
    data = np.load(obj_path, allow_pickle=True).item()

def get_low_D_feat(X):
    #tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    # X_tsne = tsne.fit_transform(X)

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne =tsne.fit_transform(X)
    #data_x = X_tsne[:, 0]
    #data_y = X_tsne[:, 1]
    #data_list=[data_x,data_y]
    return X_tsne


def test_list_range():
    a=[1,2,3,4,5,6,7,8,89]
    #index start from 0,and this expression is [left closed,right open]
    b=a[3:6]
    print('5')
def test_figure():
    np.random.seed(0)  # 执行多次每次获取的随机数都是一样的

    x1 = np.random.rand(6)draw_model_featrue_for_all_big_class
    y1 = np.random.rand(6)
    x2 = np.random.rand(3)
    y2 = np.random.rand(3)

    c1 = [5,6,6,6,7,8]
    c1=np.array(c1)
      # 生成10种大小
    c2=[2,2,3]
    plt.scatter(x1, y1, c=c1, alpha=0.7)

    plt.show()

    plt.scatter(x2,y2,c=c2,alpha=0.7)
    plt.show()

def test_for_in():
    l=[1 ,2 ,3,4,5,6]
    for i in range(2,len(l)):
        print(i)

def test_list_in_list():
    a=[1, 2, 3]
    b=[]
    for i in range(3):
        b=b+a
    print(b)

def test_range():
    #i in 0,1,2,3,4:
    for i in range(5):
        print(i)
def get_one_plot_list(low_d_feat):
    num1 = color_num_list[0]
    num2 = color_num_list[1]
    # num3=color_num_list[2]

    # convert ndarray to list

    f1 = low_d_feat[0:num1]
    f1_row = f1.shape[0]
    f1_col = f1.shape[1]
    x1_list = []
    y1_list = []
    for row in range(f1_row):
        x1_list.append(f1[row][0])
        y1_list.append(f1[row][1])


def  plot_two_catgory(low_d_feat,list_color,color_num_list):


    num1=color_num_list[0]
    num2=color_num_list[1]
    #num3=color_num_list[2]

    #convert ndarray to list

    f1=low_d_feat[0:num1]
    f1_row = f1.shape[0]
    f1_col=f1.shape[1]
    x1_list=[]
    y1_list=[]
    for row in range(f1_row):
        x1_list.append(f1[row][0])
        y1_list.append(f1[row][1])






    c1=c=list_color[0:num1]

    #convert ndarray to list

    f2=low_d_feat[num1:num2+num1]

    f2_row = f2.shape[0]
    f2_col = f2.shape[1]

    x2_list=[]
    y2_list=[]
    for row in range(f2_row):
        x2_list.append(f2[row][0])
        y2_list.append(f2[row][1])


    c2=list_color[num1:num1+num2]

    #plt.subplot(211)
    #plt.scatter(x1_list,y1_list, c=c1, alpha=0.7)
    #print("num1: ",num1)
    #print(low_d_feat[0][0:num1])

    #plt.subplot(212)
    plt.scatter(x1_list+x2_list, y1_list+y2_list, c=c1+c2, alpha=0.7)

    plt.savefig('/home/lzw/desktop/new_project/code_21_7_10/img/two_cat.jpg')
    plt.show()
    #plt.show()
    #to do :wirte code to print the num of each class ,each line contain 5~10 elements

#'02747177','02773838
def see_object_details():
    class_path="/home/lzw/desktop/new_project/data_21_7_10/object_model/02747177/"


    npy_list=os.listdir(class_path)
    for npy_path in npy_list:

        data = np.load(class_path+npy_path, allow_pickle=True).item()
        vis_pcd(data['object'][:,0:3])
    print("1")


def see_model_details():
    class_path = "/media/lzw/big/dataset/ShapeNetCore.v2/02747177/"

    pcd_list = os.listdir(class_path)
    for pcd_path in pcd_list:

        pcd = open3d.io.read_point_cloud(class_path+pcd_path+"/models/model_normalized.pcd")
        open3d.visualization.draw_geometries([pcd])
    print("1")

#draw_for_all_big_class()

def vis_pcd(pcd):
    source_data=pcd
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(source_data)
    open3d.visualization.draw_geometries([point_cloud])

#test_figure()
if __name__ == "__main__":
    #see_model_details()
    #test_figure()
    #draw_for_two_big_class()
    draw_model_featrue_for_all_big_class()
    #test_for_in()
    #test_list_range()
    #test_range()