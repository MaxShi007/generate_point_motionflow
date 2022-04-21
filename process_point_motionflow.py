from datetime import time
import os
from re import L, T
import yaml
os.environ["OMP_NUM_THREADS"] = "8"
import numpy as np
from icecream import ic
from tqdm import trange
import open3d as o3d
from utils import parse_calibration, parse_poses, make_open3d_point_cloud, dbscan_labels, labels2colors
from vis_utils import genereate_correspondence_line_set, generate_point_cloud, visualize_pc_with_motion_flow
import glob
import time

splits = ["train", "valid", "test"]

class ProcessSemanticKITTI:
    def __init__(self, process_split = "train") -> None:
        
        self.semkitti_cfg = '/home1/sunjiadai/Repo/Cylinder3D/generate_point_motionflow/semantic-kitti.yaml'
        self.color_map_cfg = '/home1/sunjiadai/Repo/Cylinder3D/generate_point_motionflow/instance_colormap.yaml'
        self.data_path = '/home1/datasets/semantic_kitti/dataset'
        self.cfg = yaml.safe_load(open(self.semkitti_cfg, 'r'))
        assert process_split in splits
        self.process_split = process_split
        self.process_sequences_list = self.cfg["split"][process_split]

        self.moving_class = [252, 253, 254, 255, 256, 257, 258, 259]
        self.mini_pcnumber_for_one_instance = 50
        self.debug = FLAG_debug_ICP
        # ic(self.process_sequences_list)

    def judge_process_sequences(self):
        if self.process_split in ['train', 'valid']:
            self.process_sequences_from_gt()
        elif self.process_split in ['test']:
            self.process_sequences_from_dbscan()

    def get_frame_data(self, frame_id):
        pc_data = np.fromfile(self.seq_scan_names[frame_id], dtype=np.float32).reshape((-1, 4))
        label = np.fromfile(self.seq_label_names[frame_id], dtype=np.uint32).reshape((-1))
        sem_label = label & 0xFFFF
        ins_label = label >> 16
        velodyne_pose = self.seq_poses[frame_id]

        return pc_data, sem_label, ins_label, velodyne_pose
    
    def get_frame_data_with_dbscan(self, frame_id):
        pc_data = np.fromfile(self.seq_scan_names[frame_id], dtype=np.float32).reshape((-1, 4))
        
        # TODO: 1. remove ground points
        #       2. remove far points
        #       2. dbscan
        
        # 1. the non-ground points mask
        use_GndNet_mask = True
        if use_GndNet_mask == True:
            label_path = self.seq_scan_names[frame_id].replace("velodyne", "ground_masks")[:-4] + ".label"
            ground_labels = np.fromfile(label_path, dtype=np.uint32)
            non_ground_mask = (ground_labels == 2) # 0 is the unlabel data, 1: ground, 2: non-ground points
        else:
            non_ground_mask = pc_data[:, 2] > -1.4  # z-axis

        # 2. The mask of the close/near points
        tmp_dis = np.linalg.norm(pc_data[:, 0:2], axis=1) # np.sqrt(pc_data[:, 0] **2 + pc_data[:, 1] **2)
        near_point_mask = (tmp_dis < 25) & (tmp_dis > 2)  # radius_threshold = 25 # .m distance
        valid_mask = near_point_mask & non_ground_mask

        labels = dbscan_labels(pc_data[valid_mask], epsilon=0.75, minpoints=50)
        # ic(pc_data.shape, labels.shape, labels.max(), labels.min())
        
        ins_merge_labels = -np.ones(shape=(pc_data.shape[0]))
        ins_merge_labels[valid_mask] = labels
        
        if False:
            # the colormap to render different class
            colormap_data = yaml.safe_load(open(self.color_map_cfg, 'r'))['colormap']
            colors_by_dbscan = labels2colors(ins_merge_labels, colormap_data)        
            vis_pcd = generate_point_cloud(pc_data[:, :3], each_points_color=colors_by_dbscan)
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            o3d.visualization.draw_geometries([vis_pcd, coordinate_frame])

        velodyne_pose = self.seq_poses[frame_id]

        return pc_data, ins_merge_labels, velodyne_pose

    def caculate_the_releative_pose(self, pc1_xyz, pc2_xyz, flags=None):
        # Here, maybe pc1_xyz is f1_ins_xyzi[:, :3], pc2_xyz = f2_part_xyz[:, :3]
        # make pointcloud class for open3D registraiton_icp
        if self.debug:
            pcd1 = make_open3d_point_cloud(pc1_xyz, color=[1,0,0])
            pcd2 = make_open3d_point_cloud(pc2_xyz, color=[0,0,1])
        else:
            pcd1 = make_open3d_point_cloud(pc1_xyz)#, color=[1,0,0])
            pcd2 = make_open3d_point_cloud(pc2_xyz)#, color=[0,0,1])
        # define the initialize matrix used for icp
        init_matrix = np.eye(4)
        if flags == "two_instance":
            init_matrix[0:3, 3] = pc2_xyz.mean(axis=0) - pc1_xyz.mean(axis=0)
        
        trans = o3d.registration.registration_icp(pcd1, pcd2, 0.2, init_matrix, #np.eye(4), # max_correspondence_distance, init
                                    o3d.registration.TransformationEstimationPointToPoint(), # estimation_method
                                    o3d.registration.ICPConvergenceCriteria(max_iteration=200)) # criteria
        if trans.fitness < 0.4:
            # TODO: It may be possible to separate two mislabels that are farther apart. DBCSCAN? by find center
            # ic(trans.fitness, "modify the init matrix")
            trans = o3d.registration.registration_icp(pcd1, pcd2, 0.2, np.eye(4), # max_correspondence_distance, init
                                    o3d.registration.TransformationEstimationPointToPoint(), # estimation_method
                                    o3d.registration.ICPConvergenceCriteria(max_iteration=200)) # criteria
        
        if self.debug:
            # evaluation = o3d.registration.evaluate_registration(pcd1, pcd2, 0.2, np.eye(4))
            # ic(trans.transformation)
            # R_ref = trans.transformation[0:3,0:3].astype(np.float32)
            # t_ref = trans.transformation[0:3,3:4].astype(np.float32)
            tmp_tranform_xyz = (trans.transformation @ np.hstack((pc1_xyz[:, :3], np.ones((pc1_xyz.shape[0], 1)))).T).T
            pcd3 = make_open3d_point_cloud(tmp_tranform_xyz[:, :3], color=[0,1,0])
            ic(trans.transformation)
            ic(trans.fitness, trans.inlier_rmse)
            ic(pc1_xyz.shape, pc2_xyz.shape)
            corrds = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
            vis_flow_lineset = genereate_correspondence_line_set(pc1_xyz, tmp_tranform_xyz[:, :3])
            o3d.visualization.draw_geometries([pcd1, pcd2, pcd3, vis_flow_lineset]) #corrds,
            # tmp_tranform_xyz = (trans.transformation @ np.hstack((f1_ins_xyzi[:, :3], np.ones((f1_ins_xyzi.shape[0], 1)))).T).T
            # f1_ins_flow_xyz = tmp_tranform_xyz[:, :3] - f1_ins_xyzi[:, :3]
            # f1_moitonflow[f1_ins_mask] = f1_ins_flow_xyz

        return trans

    def save_motionflow_vector_to_file(self, f_id, motionflow_xyz, folder_name=None):
        scan_path = self.seq_scan_names[f_id]
        assert folder_name is not None
        motionflow_dir = os.path.split(scan_path)[0].replace('velodyne', folder_name)
        if not os.path.exists(motionflow_dir):
            os.makedirs(motionflow_dir)
        frame_path = os.path.join(motionflow_dir, os.path.split(scan_path)[1].replace('.bin', '.flow'))
        np.save(frame_path, motionflow_xyz)
        ic(frame_path)
        pass

    def process_sequences_from_gt(self):
        folder_name = "motionflow"
        for sequence in self.process_sequences_list:
            sequence = '{0:02d}'.format(int(sequence))
            print(f"Process seq {sequence} ....")

            # get scan paths
            scan_paths = os.path.join(self.data_path, "sequences", str(sequence), "velodyne")
            label_paths = os.path.join(self.data_path, "sequences", str(sequence), "labels")
            pose_file = os.path.join(self.data_path, "sequences", str(sequence), "poses.txt")
            calib_file = os.path.join(self.data_path, "sequences", str(sequence), "calib.txt")

            calib_file = parse_calibration(calib_file)
            self.seq_poses = parse_poses(pose_file, calib_file)

            # populate the scan names and label names
            self.seq_scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_paths)) for f in fn if ".bin" in f]
            self.seq_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_paths)) for f in fn if ".label" in f]
            self.seq_scan_names.sort()
            self.seq_label_names.sort()

            assert len(self.seq_label_names) == len(self.seq_scan_names) and len(self.seq_scan_names) == len(self.seq_poses)
            
            for f_id in trange(len(self.seq_scan_names)):
                # if f_id < 620: continue
                # load the frame_1 data 
                f1_xyzi, f1_semlabel, f1_inslabel, f1_pose = self.get_frame_data(f_id)
                f1_moitonflow = np.zeros(shape=(f1_xyzi.shape[0], 3))
                f1_moving_label_mask = (f1_semlabel > 250)

                # if there is no moving object in frame_1 or frame_1 is the last frame of this sequence.
                if (f_id == len(self.seq_scan_names) - 1) or (f1_moving_label_mask.any() is False):
                    if FLAG_save_file:
                        self.save_motionflow_vector_to_file(f_id, f1_moitonflow, folder_name)
                    continue
                
                # else load the frame_2 data 
                f2_xyzi, f2_semlabel, f2_inslabel, f2_pose = self.get_frame_data(f_id + 1)
                f2_moving_label_mask = (f2_semlabel > 250)
                transform_f2_xyz = (np.linalg.inv(f1_pose) @ \
                                    (f2_pose @ np.hstack((f2_xyzi[:, :3], np.ones((f2_xyzi.shape[0], 1)))).T)).T

                # Count the number and id of moving instances
                f1_moving_instance_ids, f1_moving_instance_pcnum = np.unique(f1_inslabel[f1_moving_label_mask], return_counts=True)
                f2_moving_instance_ids = np.unique(f2_inslabel[f2_moving_label_mask])
                for ins_id, ins_pcnum in zip(f1_moving_instance_ids, f1_moving_instance_pcnum):
                    if ins_pcnum < self.mini_pcnumber_for_one_instance: 
                        continue

                    f1_ins_mask = (f1_inslabel == ins_id)
                    f1_ins_xyzi = f1_xyzi[f1_ins_mask]
                    if (ins_id in f2_moving_instance_ids) and ((f2_inslabel == ins_id).sum() > 0.2 * self.mini_pcnumber_for_one_instance):
                        f2_part_xyz = transform_f2_xyz[f2_inslabel == ins_id]
                        flags = "two_instance"
                    else: 
                        ic("!!!!!!!!")
                        ic((ins_id in f2_moving_instance_ids), ((f2_inslabel == ins_id).sum() > self.mini_pcnumber_for_one_instance)) 
                        ic((f2_inslabel == ins_id).sum()) 
                        # self.debug = True
                        # TODO: Need to fixQ!!!!!!!!!
                        f2_part_xyz = transform_f2_xyz[(f2_semlabel > 250)]
                        flags = "f2_moving"

                    # the point cloud registration function.
                    trans = self.caculate_the_releative_pose(f1_ins_xyzi[:, :3], f2_part_xyz[:, :3], flags)
                    if (trans.transformation == np.eye(4)).all(): 
                        continue

                    # caculate the moiton flow of each object
                    tmp_tranform_xyz = (trans.transformation @ np.hstack((f1_ins_xyzi[:, :3], np.ones((f1_ins_xyzi.shape[0], 1)))).T).T
                    f1_ins_flow_xyz = tmp_tranform_xyz[:, :3] - f1_ins_xyzi[:, :3]
                    f1_moitonflow[f1_ins_mask] = f1_ins_flow_xyz
                    # self.debug = False
                if FLAG_save_file:
                    self.save_motionflow_vector_to_file(f_id, f1_moitonflow, folder_name)
                # visualize_pc_with_motion_flow(f1_xyzi[:, :3], transform_f2_xyz[:, :3], f1_moitonflow)

    def process_sequences_from_dbscan(self):
        folder_name = "motionflow_dbscan"
        for sequence in self.process_sequences_list[7:]:
            sequence = '{0:02d}'.format(int(sequence))
            print(f"Process seq {sequence} ....")
            import pdb; pdb.set_trace()
            # get scan paths
            scan_paths = os.path.join(self.data_path, "sequences", str(sequence), "velodyne")
            label_paths = os.path.join(self.data_path, "sequences", str(sequence), "labels")
            pose_file = os.path.join(self.data_path, "sequences", str(sequence), "poses.txt")
            calib_file = os.path.join(self.data_path, "sequences", str(sequence), "calib.txt")         

            calib_file = parse_calibration(calib_file)
            self.seq_poses = parse_poses(pose_file, calib_file)

            # populate the scan names and label names
            # self.seq_scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_paths)) for f in fn if ".bin" in f]
            # self.seq_scan_names.sort()
            self.seq_scan_names = sorted(glob.glob(os.path.join(scan_paths, "*.bin")))

            assert len(self.seq_scan_names) == len(self.seq_poses)

            # for f_id in trange(len(self.seq_scan_names)):
            for f_id in trange(1764, 1765):

                # load the frame_1 data 
                f1_xyzi, f1_inslabel, f1_pose = self.get_frame_data_with_dbscan(f_id)
                f1_moitonflow = np.zeros(shape=(f1_xyzi.shape[0], 3))

                # if there is no moving object in frame_1 or frame_1 is the last frame of this sequence.
                if (f_id == len(self.seq_scan_names) - 1) or (f1_inslabel == -1).all():
                    if FLAG_save_file:
                        self.save_motionflow_vector_to_file(f_id, f1_moitonflow, folder_name)
                    continue
                
                # else load the frame_2 data 
                f2_xyzi, f2_inslabel, f2_pose = self.get_frame_data_with_dbscan(f_id + 1)
                transform_f2_xyz = (np.linalg.inv(f1_pose) @ \
                                    (f2_pose @ np.hstack((f2_xyzi[:, :3], np.ones((f2_xyzi.shape[0], 1)))).T)).T


                f1_ins_center, f2_ins_center, f1_f2_ins_match = {}, {}, {}
                f1_ids, pc_nums = np.unique(f1_inslabel, return_counts=True)
                for ins_id, pc_num in zip(f1_ids, pc_nums):
                    if ins_id == -1 or pc_num > 2000: continue
                    f1_ins_center[ins_id] = np.mean(f1_xyzi[:, :3][f1_inslabel == ins_id], axis=0)
                f2_ids, pc_nums = np.unique(f2_inslabel, return_counts=True)
                for ins_id, pc_num in zip(f2_ids, pc_nums):
                    if ins_id == -1 or pc_num > 2000: continue
                    f2_ins_center[ins_id] = np.mean(f2_xyzi[:, :3][f2_inslabel == ins_id], axis=0)

                if (len(f1_ins_center) == 0) or (len(f2_ins_center) == 0):
                    if FLAG_save_file:
                        self.save_motionflow_vector_to_file(f_id, f1_moitonflow, folder_name)
                    continue

                order_f2_ins_id = sorted(f2_ins_center.keys())
                for f1_ins_id in sorted(f1_ins_center.keys()):
                    min_dis = [np.linalg.norm(f1_ins_center[f1_ins_id] - f2_ins_center[f2_ins_id]) for f2_ins_id in order_f2_ins_id]
                    idx = np.argmin(min_dis)
                    if min_dis[idx] > 1.0: continue
                    f1_f2_ins_match[f1_ins_id] = (order_f2_ins_id[idx], min_dis[idx])

                for ins_id, mathch_tuple in f1_f2_ins_match.items():
                    f1_ins_mask = (f1_inslabel == ins_id)
                    f1_ins_xyzi = f1_xyzi[f1_ins_mask]
                    f2_part_xyz = transform_f2_xyz[f2_inslabel == mathch_tuple[0]]

                    # the point cloud registration function.
                    # self.debug = True

                    trans = self.caculate_the_releative_pose(f1_ins_xyzi[:, :3], f2_part_xyz[:, :3], flags = "two_instance")
                    # ic(trans.transformation[0:3, 3], np.linalg.norm(trans.transformation[0:3, 3]))

                    if (trans.transformation == np.eye(4)).all() or\
                        (np.abs(trans.transformation[2][3]) > 2) or\
                        (np.linalg.norm(trans.transformation[0:3, 3]) > 3) : # 80 km/h ~= 22m/s 10hz 60km/h ~= 16m/s
                        continue

                    # caculate the moiton flow of each object
                    tmp_tranform_xyz = (trans.transformation @ np.hstack((f1_ins_xyzi[:, :3], np.ones((f1_ins_xyzi.shape[0], 1)))).T).T
                    f1_ins_flow_xyz = tmp_tranform_xyz[:, :3] - f1_ins_xyzi[:, :3]
                    f1_moitonflow[f1_ins_mask] = f1_ins_flow_xyz
                if FLAG_save_file:
                    self.save_motionflow_vector_to_file(f_id, f1_moitonflow, folder_name)
                # visualize_pc_with_motion_flow(f1_xyzi[:, :3], transform_f2_xyz[:, :3], f1_moitonflow)
    


if __name__ == "__main__":
    FLAG_save_file = True
    FLAG_debug_ICP = False #True

    proSemKitti = ProcessSemanticKITTI(process_split='test')
    # proSemKitti.judge_process_sequences()
    proSemKitti.process_sequences_from_dbscan()