import sys
import numpy as np
import open3d as o3d
from sklearn.cluster import dbscan


def labels2colors(labels, colormap_data):
    # FIXME: the max of labels may greater to keys of colormap_dict
    try:
        colors = [colormap_data[i + 1] for i in labels]
        return np.asarray(colors) / 255.
    except:
        print("Unexpected error:", sys.exc_info()[0])


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