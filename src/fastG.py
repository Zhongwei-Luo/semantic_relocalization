# examples/Python/Advanced/fast_global_registration.py

import open3d as o3d
from global_registration import *
import numpy as np
import copy
import sys

import time


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.registration.registration_fast_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold, iteration_number=1000, division_factor=3))
    return result


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 2
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

def registration_ransac_based_on_correspondence(source, target, corr, voxel_size):
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target)
    corres = o3d.utility.Vector2iVector(corr)
    result = o3d.registration.registration_ransac_based_on_correspondence(source_pcd, target_pcd, corres, voxel_size,)
    return result


def FGR(source, target,voxel_size=0.03):
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target)
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             # [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source_pcd.transform(trans_init)
    # draw_registration_result(source_pcd, target_pcd, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
    # start = time.time()
    result_fast = execute_global_registration(source_down, target_down,
                                                   source_fpfh, target_fpfh,
                                                   voxel_size)
    # print("Fast global registration took %.3f sec.\n" % (time.time() - start))
    # print(result_fast)
    # draw_registration_result(source_down, target_down,
    #                          result_fast.transformation)
    return result_fast.transformation



if __name__ == "__main__":

    voxel_size = 0.008  # means 5cm for the dataset

    path = sys.argv[1]
    file_txt = path +'object.txt'
    with open(file_txt, 'r') as file:
        allDirs = file.readlines()
    allDirs = [x.strip() for x in allDirs]
    for i in range(len(allDirs)):
        for j in range(len(allDirs)):
            if i == j:
                continue
            modelFName = path + 'mat/'
            modelFName += allDirs[i]
            dataFName = path + 'mat/'
            dataFName += allDirs[j]

            outputFname = path + 'output/fgr/'
            outputFname += 'outliers_'

            outputFname += allDirs[i][:-4]
            outputFname += '_'
            outputFname += allDirs[j]

            source, target, source_down, target_down, source_fpfh, target_fpfh = \
                    prepare_dataset(dataFName, modelFName, voxel_size)


            # start = time.time()
            result_fast = execute_fast_global_registration(source_down, target_down,
                                                           source_fpfh, target_fpfh,
                                                           voxel_size)
            # print("Fast global registration took %.3f sec.\n" % (time.time() - start))
            # print(result_fast)
            # draw_registration_result(source_down, target_down,
            #                          result_fast.transformation)
            trans = np.zeros((7, 3))
            trans[1:4, :] = result_fast.transformation[:3, :3].transpose()
            trans[4:, 0] = result_fast.transformation[:3, 3]



            np.savetxt(outputFname, trans)