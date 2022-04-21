"""
@Time    : 11/1/20 5:28 PM
@Author  : Jiadai Sun
"""

import numpy as np
from sklearn.cluster import dbscan
import os, sys

sys.path.append('..')
sys.path.append('../tools')
from vis_utils import *
# from io_utils import *
import yaml
from icecream import ic

# from sceneflow_hdf5_load import VirtualKitti
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

def labels2colors(labels, colormap_data):
    # FIXME: the max of labels may greater to keys of colormap_dict
    try:
        colors = [colormap_data[i + 1] for i in labels]
        return np.asarray(colors) / 255.
    except:
        print("Unexpected error:", sys.exc_info()[0])


def dbscan_labels(pointcloud, epsilon, minpoints, rgb_weight=0,
                  algorithm='ball_tree'):
    if rgb_weight > 0:
        X = pointcloud.to_array()
        X[:, 3:] *= rgb_weight
    else:
        X = pointcloud

    _, labels = dbscan(X, eps=epsilon, min_samples=minpoints,
                       algorithm=algorithm)
    return np.asarray(labels)


if __name__ == '__main__':

    # the colormap to render different class
    colormap_path = os.path.join(os.path.dirname(__file__), 'instance_colormap.yaml')
    colormap_data = yaml.safe_load(open(colormap_path, 'r'))['colormap']

    switch = 3
    if switch == 0:
        pc_xyz = np.random.random(size=(2048, 3))
    elif switch == 1:
        # https://blog.csdn.net/ChanningLau/article/details/104898232
        d = 4
        mesh = o3d.geometry.TriangleMesh.create_tetrahedron().translate((-d, 0, 0))
        mesh += o3d.geometry.TriangleMesh.create_octahedron().translate((0, 0, 0))
        mesh += o3d.geometry.TriangleMesh.create_icosahedron().translate((d, 0, 0))
        mesh += o3d.geometry.TriangleMesh.create_torus().translate((-d, -d, 0))
        mesh += o3d.geometry.TriangleMesh.create_moebius(twists=1).translate((0, -d, 0))
        mesh += o3d.geometry.TriangleMesh.create_moebius(twists=2).translate((d, -d, 0))
        ##apply k means on this point cloud
        point_cloud = mesh.sample_points_uniformly(int(1e3))
        ##transfer point cloud into array
        pc_xyz = np.asarray(point_cloud.points)
    elif switch == 2:
        vkitti_data = VirtualKitti('/data/VirtualKitti/vkitti_2.0.3_sceneflow_filter')
    elif switch == 3:
        seq = "02"
        f_id = 10

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
    else:
        raise ValueError("Need to be implemented.")

    if switch == 2:
        # Don’t know which Scene** belong to
        for i in range(len(vkitti_data)):
            pc_1, pc_2, sceneflow, instance_id, movinginstance_id = vkitti_data[i]
            # colors_1 = np.asarray([colormap_data[i] for i in instance_id]) / 255.0
            # colors_1 = np.asarray([colormap_data[i] for i in movinginstance_id]) / 255.0
            pc_1 = pc_1[np.random.permutation(pc_1.shape[0])][:2048]
            labels = dbscan_labels(pc_1, epsilon=0.4, minpoints=10)
            colors_1 = labels2colors(labels, colormap_data)

            vis_pcd_1 = generate_point_cloud(pc_1, each_points_color=colors_1)
            # vis_pcd_2 = generate_point_cloud(pc_2, uniform_color=[0, 1.0, 0])  # each_points_color=colors_2
            # vis_flow_lineset = genereate_correspondence_line_set(pc_1, pc_2)
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            vis_pcd_1.transform(flip_transform)
            # vis_pcd_2.transform(flip_transform)
            # vis_flow_lineset.transform(flip_transform)
            coordinate_frame.transform(flip_transform)
            o3d.visualization.draw_geometries([vis_pcd_1, coordinate_frame])
            # o3d.visualization.draw_geometries([vis_pcd_1, vis_pcd_2, coordinate_frame])
            # o3d.visualization.draw_geometries([vis_pcd_1, vis_pcd_2, vis_flow_lineset, coordinate_frame])
    elif switch == 3:
        non_ground_point = pc1_data[:, 2] > -1
        radius_threshold = 20
        # near_point_mask  = (pc1_data[:, 0] < radius_threshold) & (pc1_data[:, 0] > -radius_threshold) & (pc1_data[:, 1] < radius_threshold) & (pc1_data[:, 1] > -radius_threshold)
        near_point_mask = np.sqrt(pc1_data[:, 0] **2 + pc1_data[:, 1] **2) < radius_threshold
        valid_mask = near_point_mask & non_ground_point


        vis_pcd_1 = generate_point_cloud(pc1_data[:, :3][valid_mask])
        o3d.visualization.draw_geometries([vis_pcd_1])

        valid_mask = non_ground_point
        tmp_data = pc1_data[valid_mask]
        labels = dbscan_labels(tmp_data, epsilon=0.75, minpoints=30)
        ic(labels.shape, labels.max(), labels.min())
        total_labels = np.zeros(shape=(pc1_data.shape[0]))
        total_labels[valid_mask] = labels
        total_labels[~valid_mask] = -1

        colors_1 = labels2colors(total_labels, colormap_data)

        
        vis_pcd_1 = generate_point_cloud(pc1_data[:, :3], each_points_color=colors_1)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([vis_pcd_1, coordinate_frame])

        import pdb; pdb.set_trace()
        if False:
            pc_1, pc_2, sceneflow, instance_id, movinginstance_id = vkitti_data[i]
            # colors_1 = np.asarray([colormap_data[i] for i in instance_id]) / 255.0
            # colors_1 = np.asarray([colormap_data[i] for i in movinginstance_id]) / 255.0
            pc_1 = pc_1[np.random.permutation(pc_1.shape[0])][:2048]
            labels = dbscan_labels(pc_1, epsilon=0.4, minpoints=10)
            colors_1 = labels2colors(labels, colormap_data)

            vis_pcd_1 = generate_point_cloud(pc_1, each_points_color=colors_1)
            # vis_pcd_2 = generate_point_cloud(pc_2, uniform_color=[0, 1.0, 0])  # each_points_color=colors_2
            # vis_flow_lineset = genereate_correspondence_line_set(pc_1, pc_2)
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            vis_pcd_1.transform(flip_transform)
            # vis_pcd_2.transform(flip_transform)
            # vis_flow_lineset.transform(flip_transform)
            coordinate_frame.transform(flip_transform)
            o3d.visualization.draw_geometries([vis_pcd_1, coordinate_frame])
            # o3d.visualization.draw_geometries([vis_pcd_1, vis_pcd_2, coordinate_frame])
            # o3d.visualization.draw_geometries([vis_pcd_1, vis_pcd_2, vis_flow_lineset, coordinate_frame])

    else:
        labels = dbscan_labels(pc_xyz, epsilon=0.5, minpoints=5)
        colors = labels2colors(labels, colormap_data)
        pcd = generate_point_cloud(pc_xyz, each_points_color=colors)
        o3d.visualization.draw_geometries([pcd])
