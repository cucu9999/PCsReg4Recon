import open3d as o3d
import  numpy as np
from max_clique import infer as reg2max


# 两两配准
def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    # add for estimate normals by cucu at 2024/07/24
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())  # 用一个初始4*4转换矩阵，粗配准
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())  # 用粗配准的出的旋转矩阵，进行再次配准
    transformation_icp = icp_fine.transformation  # 配准后的出的旋转矩阵
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)  # 从变换矩阵计算信息矩阵
    return transformation_icp, information_icp



# 全局配准
def full_registration(pcds, voxel_size):
    pose_graph = o3d.pipelines.registration.PoseGraph()  # 姿势图
    odometry = np.identity(4)  # 对角线是1的方阵，4*4
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))  # 不懂这句，odometry是边
    n_pcds = len(pcds)
    for source_id in range(n_pcds):  # 遍历每个点云
        for target_id in range(source_id + 1, n_pcds):  # 遍历两两点云

            print(f"正在配准第{source_id}和{source_id +1}点云对--------------------------------------")
            # if source_id ==2 or source_id ==12 or source_id ==13 or source_id ==14 or source_id ==18 or source_id == 20:
            #     voxel_size = 0.05
            # elif source_id == 6:
            #     voxel_size = 0.02
            # else:
            #     voxel_size = 0.03

            transformation_icp, information_icp = pairwise_registration(pcds[source_id], pcds[target_id])  # 进行两两配对，获得转换矩阵、信息矩阵
            # _, transformation_icp, information_icp = reg2max(pcds[source_id], pcds[target_id], voxel_size)

            # print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case 在姿势图相邻的两个点云
                odometry = np.dot(transformation_icp, odometry)  # 两个矩阵相乘
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(np.linalg.inv(odometry)))  # 求逆矩阵，往图里添加node
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=False))  # uncertain用于区分是Odometry edges
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=True))  # 添加边Loop closure edges，非邻居的点云的边
    return pose_graph





# 返回的是一个list，里面都是降采样的（如果体素大小为0则源点云）
def load_point_clouds(voxel_size=0.0):
    pcds = []
    for i in range(3):
        pcd = o3d.io.read_point_cloud("/Users/cucu/Desktop/Xcode/reg4recon/test/RGBPoints_%d.ply" % i)
        
        # -------------------- 点云尺寸归一化 -------------------
        # pcd_points = np.asarray(pcd.points)
        # pcd_points /= 1000.0 # unit --> 1000.0
        # pcd.points = o3d.utility.Vector3dVector(pcd_points)
        # ----------------------------------------------------

        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)  # 在reg2max模块中会进行降采样操作
        # pcds.append(pcd)
        pcds.append(pcd_down)

    return pcds


def main():
    global max_correspondence_distance_coarse
    global max_correspondence_distance_fine

    # ----------------------------------- 加载数据 ----------------------------------
    voxel_size = 0.05  # default = 0.05
    pcds_down = load_point_clouds(voxel_size)  # 获得一个经过降采样的点云list


    print("----------------------------------- 逐一配准 ----------------------------------")
    max_correspondence_distance_coarse = voxel_size * 15  # 粗配准距离阈值
    max_correspondence_distance_fine = voxel_size * 1.5   # 精配准距离阈值

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        pose_graph = full_registration(pcds_down, voxel_size)



    print("---------------------------------- 全局优化 ------------------------------------")
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph, o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(), option)  


    print("---------------------------------- 合并结果 ------------------------------------")
    pcds = load_point_clouds(voxel_size)
    pcd_combined = o3d.geometry.PointCloud()
    for point_id in range(len(pcds)):
        pcds[point_id].transform(pose_graph.nodes[point_id].pose)
        pcd_combined += pcds[point_id]   

    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    o3d.io.write_point_cloud("multiway_registration.pcd", pcd_combined_down)

    # o3d.visualization.draw_geometries([pcd_combined_down])
    o3d.visualization.draw_geometries([pcd_combined])
    


if __name__ == "__main__":
    main()

