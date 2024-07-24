# 当执行配准模块时，输入通常只考虑待配准点云对，这极大的降低了其可复用性，也就是可迭代性。
# 因此，新构建的配准模块引入基本变换矩阵，当没有时则为单位矩阵。
# 以下所述代码基于open3d的基本架构进行构建。

import open3d as o3d
import numpy as np


def pairwise_registration(source, target, init_trans=np.identity(4)):

    # add for estimate normals by cucu at 2024/07/24
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # step01: coarse registraion
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, init_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    # step02: fine registration
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    # final trans 
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp # class:numpy(4,4) , class:numpy(6,6)


def main():
    global max_correspondence_distance_coarse
    global max_correspondence_distance_fine
    


if __name__ == "__main__":
    main()