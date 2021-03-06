import numpy as np
import open3d as o3d
import os
from icecream import ic
from datetime import datetime
import time
from utils import parse_calibration, parse_poses

# flow=np.load('/share/sgb/semantic_kitti/dataset/sequences/08/motionflow/000000.flow.npy')
# data1=np.fromfile('/share/sgb/semantic_kitti/dataset/sequences/08/velodyne/000000.bin',dtype=np.float32).reshape(-1,4)
# data2=np.fromfile('/share/sgb/semantic_kitti/dataset/sequences/08/velodyne/000001.bin',dtype=np.float32).reshape(-1,4)

# print(flow.shape)
# # print(data1.shape)
# pc=o3d.geometry.PointCloud()
# pc.points=o3d.utility.Vector3dVector(data1[:,:3])
# pc2=o3d.geometry.PointCloud()
# pc2.points=o3d.utility.Vector3dVector(data2[:,:3])

# vis=o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(pc)
# vis.add_geometry(pc2)
# vis.run()

# flow_paths = '/share/sunjiadai/semantic_kitti/dataset/sequences/08/motionflow_all_instance_1'

# data = np.load(os.path.join(flow_paths, '000000.flow.npy'))
# data2 = np.load(os.path.join(flow_paths, '000002.flow.npy'))
# ic(data.shape, np.unique(data, return_counts=True))
# print((data == 0).sum() / 3)

velo = np.fromfile("/share/sgb/semantic_kitti/dataset/sequences/38/velodyne/000020.bin", dtype=np.float32).reshape(-1, 4)
print(velo.shape)