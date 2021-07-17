import numpy as np
import os
from emicp import global_em_icp
import concurrent.futures
from tools import rigid_transform, show_pc, points_distance,show_pair_pc,downsample

def multi_process(cad_name):
    em_icp = global_em_icp(2)
    # graph = {}
    graph_save_name = os.path.join("/media/lan/Samsung_T5/scan2cad/data/object_model_similarity", cad_name)
    if os.path.exists(os.path.join(graph_save_name, "similarity_graph.npy")):
        graph = np.load(os.path.join(graph_save_name, "similarity_graph.npy"), allow_pickle=True).item()
    else:
        graph = {}
    if not os.path.exists(graph_save_name):
        os.makedirs(graph_save_name)
    data_path1 = os.path.join(data_path, cad_name)
    cad_inner_list = os.listdir(data_path1)
    for i in range(len(cad_inner_list)):
        cad1 = cad_inner_list[i]
        cad1_data_path = os.path.join(data_path1, cad1)
        cad1_name = cad1_data_path.split('/')[-1].split('_')[-2]
        cad1_data = np.load(cad1_data_path, allow_pickle=True).item()
        cad1_model = cad1_data["model"][:, :3]

        for j in range(i + 1, len(cad_inner_list)):
            cad2 = cad_inner_list[j]
            # for cad2 in cad_inner_list:
            cad2_data_path = os.path.join(data_path1, cad2)
            cad2_name = cad2_data_path.split('/')[-1].split('_')[-2]
            if cad1_name == cad2_name:
                continue
            # if name == "2c7fd96b46b2b5b5efc579970fcfc006" and name1 == "2d3a484f14ec3d4d7b11ae648ea92233":
            # print("a")
            # else:
            # continue
            keys = cad1_name + "-" + cad2_name
            keys1 = cad2_name + "-" + cad1_name
            if keys in graph.keys() or keys1 in graph.keys():
                continue
            cad2_data = np.load(cad2_data_path, allow_pickle=True).item()
            cad2_model = cad2_data["model"][:, :3]
            # print(data_path2, data_path22)
            # show_pair_pc(model, model1)
            trans, error = em_icp.run(cad1_model, cad2_model)
            # trans = np.linalg.inv(trans)
            # cad2_model = rigid_transform(cad2_model, trans)


            # show_pair_pc(cad1_model, cad2_model)
            # print(error)

            graph[keys] = error
            graph[keys1] = error
    np.save(os.path.join(graph_save_name, "similarity_graph.npy"), graph)
    print(cad_name)
    return 1


if __name__ == '__main__':
    data_path = "/media/lan/Samsung_T5/scan2cad/data/object_model"

    cad_classes_list = os.listdir(data_path)
    # cad_classes_list = ["02880940"]
    # graph = np.load('/media/lan/Samsung_T5/scan2cad/data/graph.npy', allow_pickle=True).item()
    # graph = {}

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=8)
    for cad_name, result in zip(cad_classes_list, executor.map(multi_process, cad_classes_list)):
        print(cad_name)
    executor.shutdown(wait=True)

    if 0:
        em_icp = global_em_icp(2)
        for ii, cad_name in enumerate(cad_classes_list):
            print(cad_name)
            if not cad_name == "04379243":
                continue
            graph = {}
            graph_save_name = os.path.join("/media/lan/Samsung_T5/scan2cad/data/object_model_similarity", cad_name)
            # if not os.path.exists(graph_save_name):
                # os.makedirs(graph_save_name)
            # graph = np.load(os.path.join(graph_save_name, "similarity_graph.npy"), allow_pickle=True).item()

            data_path1 = os.path.join(data_path, cad_name)
            cad_inner_list = os.listdir(data_path1)
            for i in range(1, 2):
                cad1 = cad_inner_list[i]
                cad1_data_path = os.path.join(data_path1, cad1)
                cad1_name = cad1_data_path.split('/')[-1].split('_')[-2]
                cad1_data = np.load(cad1_data_path, allow_pickle=True).item()
                cad1_model = cad1_data["model"][:, :3]

                for j in range(i+1, len(cad_inner_list),500):
                    cad2 = cad_inner_list[j]
                    cad2_data_path = os.path.join(data_path1, cad2)
                    cad2_name = cad2_data_path.split('/')[-1].split('_')[-2]
                    if cad1_name == cad2_name:
                        continue

                    keys = cad1_name + "-" + cad2_name
                    keys1 = cad2_name + "-" + cad1_name
                    if keys in graph.keys() or keys1 in graph.keys():
                        continue
                    cad2_data = np.load(cad2_data_path, allow_pickle=True).item()
                    cad2_model = cad2_data["model"][:, :3]
                    # print(data_path2, data_path22)
                    # show_pair_pc(model, model1)
                    trans, error = em_icp.run(cad1_model, cad2_model)
                    trans = np.linalg.inv(trans)
                    cad2_model = rigid_transform(cad2_model, trans)
                    print(error)
                    show_pair_pc(cad1_model, cad2_model)


                    graph[keys] = error
                    graph[keys1] = error
            # np.save(os.path.join(graph_save_name, "similarity_graph.npy"), graph)
            print(cad_name)












