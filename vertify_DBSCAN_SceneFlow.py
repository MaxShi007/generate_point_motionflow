import numpy as np
from icecream import ic
from numpy.core.shape_base import hstack
import open3d as o3d

def parse_calibration(filename):
        """ read calibration file with given filename

            Returns
            -------
            dict
                Calibration matrices as 4x4 numpy arrays.
        """
        calib = {}

        calib_file = open(filename)
        for line in calib_file:
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()

        return calib

def parse_poses(filename, calibration):
        """ read poses file with per-scan poses from given filename

            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses


def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        if len(color) != len(xyz):
            color = np.tile(color, (len(xyz), 1))
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


if False:
    seq = 11
    f_id = 1

    f_id_str = "{:06d}".format(f_id)
    pc1_path = f"/home1/datasets/semantic_kitti/dataset/sequences/{seq}/velodyne/000000.bin"
    pc2_path = f"/home1/datasets/semantic_kitti/dataset/sequences/{seq}/velodyne/{f_id_str}.bin"

    pose_file = f"/home1/datasets/semantic_kitti/dataset/sequences/{seq}/poses.txt"
    calib_file = f"/home1/datasets/semantic_kitti/dataset/sequences/{seq}/calib.txt"


    calib_file = parse_calibration(calib_file)
    # print(calib_file)

    poses = parse_poses(pose_file, calib_file)
    ic(len(poses))

    pc1_data = np.fromfile(pc1_path, dtype=np.float32).reshape((-1, 4))
    pc2_data = np.fromfile(pc2_path, dtype=np.float32).reshape((-1, 4))
    pcd1 = make_open3d_point_cloud(pc1_data[:, :3], color=[0, 1, 0]) # green

    T_frame2world = poses[f_id]
    new_point = (T_frame2world @ np.hstack((pc2_data[:, :3], np.ones((pc2_data.shape[0], 1)))).T).T
    pcd2 = make_open3d_point_cloud(new_point[:, :3], color=[0, 0, 1])

    # pcd2 = make_open3d_point_cloud(pc2_data[:, :3], color=[0, 0, 1])

    reg = o3d.registration.registration_icp(pcd1, pcd2, 0.2, np.eye(4),
                            o3d.registration.TransformationEstimationPointToPoint(),
                            o3d.registration.ICPConvergenceCriteria(max_iteration=200))

    ic(reg.transformation)
    ic(np.asarray(reg.correspondence_set))  # 数量比pc1_data的少
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=35)

    o3d.visualization.draw_geometries([pcd1, pcd2, mesh])


seq = 11
rigid3Dsceneflow_npz_path = f"/home1/sunjiadai/Repo/Rigid3DSceneFlow/data/semantic_kitti/{seq}/000000_000001.npz"

data = np.load(rigid3Dsceneflow_npz_path)
ic(data.files)  # ['pc1', 'pc2', 'pose_s', 'pose_t']

pc1_data = data['pc1']
pc2_data = data['pc2']
pcd1 = make_open3d_point_cloud(pc1_data[:, :3], color=[0, 1, 0]) # green

T_frame2world = data['pose_t']
new_point = (np.linalg.inv(T_frame2world) @ np.hstack((pc2_data[:, :3], np.ones((pc2_data.shape[0], 1)))).T).T
pcd2 = make_open3d_point_cloud(new_point[:, :3], color=[0, 0, 1])

# pcd2 = make_open3d_point_cloud(pc2_data[:, :3], color=[0, 0, 1])

reg = o3d.registration.registration_icp(pcd1, pcd2, 0.2, np.eye(4),
                        o3d.registration.TransformationEstimationPointToPoint(),
                        o3d.registration.ICPConvergenceCriteria(max_iteration=200))

ic(reg.transformation)
ic(np.asarray(reg.correspondence_set))  # 数量比pc1_data的少

o3d.visualization.draw_geometries([pcd1, pcd2])