"""
  @Author  : Jiadai Sun

"""
import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plot


def vis_four_result(disp1, disp1_occ, disp1_change, flow):
    """
    :param disp1:
    :param disp1_occ:
    :param disp1_change:
    :param flow:
    :return:
    """
    plot.subplot(2, 2, 1)
    plot.imshow(disp1)
    plot.title('disp1')
    plot.subplot(2, 2, 2)
    plot.imshow(disp1_occ)
    plot.title('disp1_occ')
    plot.subplot(2, 2, 3)
    plot.imshow(disp1_change)
    plot.title('disp1_change')
    plot.subplot(2, 2, 4)
    flow_vis = np.concatenate((flow, np.zeros((flow.shape[0], flow.shape[1], 1))), axis=2)
    plot.imshow(flow_vis)
    plot.title('flow_vis')
    plot.show()


def generate_point_cloud(points, uniform_color=None, each_points_color=None):
    """
    :param points: [1024, 3]
    :param uniform_color:  [3, ]
    :param each_points_color: [1024, 3]
    :return: o3d.geometry.PointCloud()
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if uniform_color is None and each_points_color is None:
        pcd.paint_uniform_color([1.0, 0, 0])
    elif uniform_color is not None:
        pcd.paint_uniform_color(uniform_color)
    elif each_points_color is not None:
        # float64 array of shape (num_points, 3) range [0, 1]
        pcd.colors = o3d.utility.Vector3dVector(each_points_color)
    # pcd.estimate_normals()
    return pcd


flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
def draw_geometries_flip(pcds):
    pcds_transform = []
    for pcd in pcds:
        pcd_temp = copy.deepcopy(pcd)
        pcd_temp.transform(flip_transform)
        pcds_transform.append(pcd_temp)
    o3d.visualization.draw_geometries(pcds_transform)


def genereate_correspondence_line_set(src, target, uniform_line_color=None, line_colors=None):
    """
    Args:
        src: (num_points, 3) | xyz
        target: (num_points, 3) | xyz
        uniform_line_color:
        line_colors:

    Returns:

    """
    # src = src[0::10]
    # target = target[0::10]
    points = np.concatenate((src, target), axis=0)

    lines = np.arange(src.shape[0] * 2).reshape(-1, 2, order='F')
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    if uniform_line_color is not None:
        colors = np.expand_dims(uniform_line_color, 0).repeat(len(lines), axis=0)
    elif line_colors is not None:
        colors = line_colors
    else:
        colors = np.expand_dims([0.47, 0.53, 0.7], 0).repeat(len(lines), axis=0)
        # colors = np.zeros((len(lines), 3))
        # colors[:] = [0.47, 0.53, 0.7]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def visualize_pc_with_motion_flow(pc1, pc2, flow):
    pcd1 = generate_point_cloud(pc1, uniform_color=[0.47, 0.53, 0.7])
    pcd2 = generate_point_cloud(pc2, uniform_color=[0.85, 0.65, 0.125])
    non_zero_mask = (np.sum(flow == 0, axis=1) != 3)
    # import pdb; pdb.set_trace()
    print(non_zero_mask.shape)
    _xyz1 = pc1[non_zero_mask]
    _xyz2 = pc1[non_zero_mask] + flow[non_zero_mask]
    vis_flow_lineset = genereate_correspondence_line_set(_xyz1, _xyz2, uniform_line_color=[1, 0, 0])
    corrds = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
    o3d.visualization.draw_geometries([pcd1, pcd2, vis_flow_lineset, corrds])

    return