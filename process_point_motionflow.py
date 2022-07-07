from datetime import datetime
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

    def __init__(self, process_split="train") -> None:

        self.semkitti_cfg = './semantic-kitti.yaml'
        self.color_map_cfg = './instance_colormap.yaml'
        self.data_path = '/share/sgb/semantic_kitti/dataset'
        self.cfg = yaml.safe_load(open(self.semkitti_cfg, 'r'))
        assert process_split in splits
        self.process_split = process_split
        self.process_sequences_list = self.cfg["split"][process_split]

        self.moving_class = [252, 253, 254, 255, 256, 257, 258, 259]
        self.mini_pcnumber_for_one_instance = 50
        self.debug = FLAG_debug_ICP
        self.logs = []

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
            label_path = self.seq_scan_names[frame_id].replace("velodyne", "ground_masks")[:-4] + ".label"  #! 这个文件哪里来的？
            ground_labels = np.fromfile(label_path, dtype=np.uint32)
            non_ground_mask = (ground_labels == 2)  # 0 is the unlabel data, 1: ground, 2: non-ground points
        else:
            non_ground_mask = pc_data[:, 2] > -1.4  # z-axis #!这个值怎么设定的，可视化了一帧，1.4以下还有很多有用的点

        # 2. The mask of the close/near points
        tmp_dis = np.linalg.norm(pc_data[:, 0:2], axis=1)  # np.sqrt(pc_data[:, 0] **2 + pc_data[:, 1] **2)
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
            pcd1 = make_open3d_point_cloud(pc1_xyz, color=[1, 0, 0])
            pcd2 = make_open3d_point_cloud(pc2_xyz, color=[0, 0, 1])
        else:
            pcd1 = make_open3d_point_cloud(pc1_xyz)  #, color=[1,0,0])
            pcd2 = make_open3d_point_cloud(pc2_xyz)  #, color=[0,0,1])
        # define the initialize matrix used for icp
        init_matrix = np.eye(4)
        if flags == "two_instance":
            init_matrix[0:3, 3] = pc2_xyz.mean(axis=0) - pc1_xyz.mean(axis=0)

            iteration = 200  # 1000
        else:
            iteration = 200
        if self.debug:
            start = time.time()

        trans = o3d.pipelines.registration.registration_icp(
            pcd1,
            pcd2,
            0.2,
            init_matrix,  #np.eye(4), # max_correspondence_distance, init
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),  # estimation_method
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iteration))  # criteria

        if trans.fitness < 0.4:
            # TODO: It may be possible to separate two mislabels that are farther apart. DBCSCAN? by find center
            # ic(trans.fitness, "modify the init matrix")
            trans = o3d.pipelines.registration.registration_icp(
                pcd1,
                pcd2,
                0.2,
                np.eye(4),  # max_correspondence_distance, init
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),  # estimation_method
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iteration))  # criteria
        if self.debug:
            end = time.time()
            print(f"point_num:{len(pc1_xyz)} cost_time:{end-start}")
        if self.debug:
            # evaluation = o3d.registration.evaluate_registration(pcd1, pcd2, 0.2, np.eye(4))
            # ic(trans.transformation)
            # R_ref = trans.transformation[0:3,0:3].astype(np.float32)
            # t_ref = trans.transformation[0:3,3:4].astype(np.float32)
            tmp_tranform_xyz = (trans.transformation @ np.hstack((pc1_xyz[:, :3], np.ones((pc1_xyz.shape[0], 1)))).T).T
            pcd3 = make_open3d_point_cloud(tmp_tranform_xyz[:, :3], color=[0, 1, 0])
            ic(trans.transformation)
            ic(trans.fitness, trans.inlier_rmse)
            ic(pc1_xyz.shape, pc2_xyz.shape)
            corrds = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))
            vis_flow_lineset = genereate_correspondence_line_set(pc1_xyz, tmp_tranform_xyz[:, :3])
            o3d.visualization.draw_geometries([pcd1, pcd2, pcd3, vis_flow_lineset])  #corrds,
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

    def process_sequences_from_gt(self, add_ego_motion=False):
        '''
        通过moving_instance_mask生成flow，以当前帧为基准计算过去帧到当前帧的flow
        '''
        count = 0
        if add_ego_motion:
            folder_name = f'motionflow_ego_motion_{FRAME_DIFF}'
        else:
            folder_name = f"motionflow_{FRAME_DIFF}"
        # ic(folder_name)
        for sequence in self.process_sequences_list:
            start = time.time()
            sequence = '{0:02d}'.format(int(sequence))
            print(f"Process seq {sequence} ....")

            # get scan paths
            scan_paths = os.path.join(self.data_path, "sequences", str(sequence), "velodyne")
            label_paths = os.path.join(self.data_path, "sequences", str(sequence), "labels")
            pose_file = os.path.join(self.data_path, "sequences", str(sequence), "poses.txt")
            calib_file = os.path.join(self.data_path, "sequences", str(sequence), "calib.txt")

            calib_file = parse_calibration(calib_file)
            self.seq_poses = parse_poses(pose_file, calib_file)  #tr_inv pose tr

            # populate the scan names and label names
            self.seq_scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_paths)) for f in fn if ".bin" in f]
            self.seq_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_paths)) for f in fn if ".label" in f]
            self.seq_scan_names.sort()
            self.seq_label_names.sort()

            assert len(self.seq_label_names) == len(self.seq_scan_names) and len(self.seq_scan_names) == len(self.seq_poses)

            for f_id in trange(len(self.seq_scan_names)):
                # if f_id < 620: continue
                # load the frame_1 data
                current_xyzi, current_semlabel, current_inslabel, current_pose = self.get_frame_data(f_id)
                currrent_moving_label_mask = (current_semlabel > 250)

                motionflow = np.zeros(shape=(current_xyzi.shape[0], 3))

                if f_id - FRAME_DIFF < 0 or (currrent_moving_label_mask.any() is False):
                    if FLAG_save_file:
                        self.save_motionflow_vector_to_file(f_id, motionflow, folder_name)
                    continue

                last_xyzi, last_semlabel, last_inslabel, last_pose = self.get_frame_data(f_id - FRAME_DIFF)
                last_moving_label_mask = (last_semlabel > 250)

                last_xyzi_transformed=(np.linalg.inv(current_pose) @ \
                                    (last_pose @ np.hstack((last_xyzi[:, :3], np.ones((last_xyzi.shape[0], 1)))).T)).T
                current_xyzi_transformed=(np.linalg.inv(last_pose) @ \
                    (current_pose@ np.hstack((current_xyzi[:,:3],np.ones((current_xyzi.shape[0],1)))).T)).T

                current_moving_instance_ids, current_moving_instance_pcnum = np.unique(current_inslabel[currrent_moving_label_mask], return_counts=True)
                last_moving_instance_ids, last_moving_instance_pcnum = np.unique(last_inslabel[last_moving_label_mask], return_counts=True)

                # moving_instance
                for ins_id, ins_pcnum in zip(current_moving_instance_ids, current_moving_instance_pcnum):
                    if FLAG_mini_pcnumber_for_one_instance and ins_pcnum < self.mini_pcnumber_for_one_instance:
                        log = f'sequence:{sequence} f_id:{f_id} ins_id:{ins_id} ins_pcnum:{ins_pcnum} current_moving_instance_pcnum点太少，没有进入ICP\n'
                        self.logs.append(log)
                        continue

                    current_ins_mask = (current_inslabel == ins_id)
                    current_ins_xyzi = current_xyzi[current_ins_mask]
                    #             if ins_id == 0 and (ins_id in last_moving_instance_ids) and (FLAG_mini_pcnumber_for_one_instance and (last_inslabel == ins_id).sum() > 0.2 * self.mini_pcnumber_for_one_instance):
                    #                 count += 1
                    # print(f'count:{count}')

                    if (ins_id in last_moving_instance_ids) and (FLAG_mini_pcnumber_for_one_instance and (last_inslabel == ins_id).sum() > 0.2 * self.mini_pcnumber_for_one_instance):
                        last_ins_xyzi = last_xyzi_transformed[last_inslabel == ins_id]
                        flags = 'two_instance'

                    else:
                        last_ins_xyzi = last_xyzi[last_semlabel > 250]
                        if ins_id in last_moving_instance_ids:
                            temp_content = 'in last_moving_instance_ids'
                        else:
                            temp_content = 'not in last_moving_instance_ids '
                        last_inslabel_sum = (last_inslabel == ins_id).sum()
                        log = f'sequence:{sequence} f_id:{f_id} ins_id:{ins_id} ins_pcnum:{ins_pcnum} last_inslabel_sum:{last_inslabel_sum}---{temp_content}\n'
                        self.logs.append(log)
                        flags = 'unmatched'

                    trans = self.caculate_the_releative_pose(current_ins_xyzi[:, :3], last_ins_xyzi[:, :3], flags)  #scr:当前帧，tar：过去帧，因为要计算的是当前帧的flow，所以把当前帧转到过去帧
                    tmp_tranform_xyz = (trans.transformation @ np.hstack((current_ins_xyzi[:, :3], np.ones((current_ins_xyzi.shape[0], 1)))).T).T
                    ins_flow_xyz = current_ins_xyzi[:, :3] - tmp_tranform_xyz[:, :3]
                    motionflow[current_ins_mask] = ins_flow_xyz

                # ego_motion
                if add_ego_motion:
                    ego_motion = current_xyzi[:, :3] - current_xyzi_transformed[:, :3]
                    motionflow += ego_motion

                if FLAG_save_file:
                    self.save_motionflow_vector_to_file(f_id, motionflow, folder_name)

            end = time.time()
            time_cost = (end - start) / len(self.seq_scan_names)
            print(f"time cost per seq mean:{time_cost}")

        if FLAG_save_log and len(self.logs) != 0 and FLAG_save_file:
            with open('unmatched_log.txt', 'a') as f:
                f.write(str(datetime.now()) + '\n')
                f.write(''.join(self.logs))

    def process_sequences_from_dbscan(self):
        folder_name = "motionflow_dbscan"
        for sequence in self.process_sequences_list[7:]:
            sequence = '{0:02d}'.format(int(sequence))
            print(f"Process seq {sequence} ....")
            import pdb
            pdb.set_trace()
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

                #不同帧之间的instance匹配，利用两帧instance中心点的距离
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
                    if min_dis[idx] > 1.0: continue  #!这个1是怎么来的？
                    f1_f2_ins_match[f1_ins_id] = (order_f2_ins_id[idx], min_dis[idx])

                for ins_id, mathch_tuple in f1_f2_ins_match.items():
                    f1_ins_mask = (f1_inslabel == ins_id)
                    f1_ins_xyzi = f1_xyzi[f1_ins_mask]
                    f2_part_xyz = transform_f2_xyz[f2_inslabel == mathch_tuple[0]]

                    # the point cloud registration function.
                    # self.debug = True

                    trans = self.caculate_the_releative_pose(f1_ins_xyzi[:, :3], f2_part_xyz[:, :3], flags="two_instance")
                    ic(trans.transformation[0:3, 3], np.linalg.norm(trans.transformation[0:3, 3]))

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

    def process_sequences_from_all_instance(self):

        start = time.time()

        folder_name = f'motionflow_all_instance_{FRAME_DIFF}'
        for sequence in self.process_sequences_list:

            sequence = '{0:02d}'.format(int(sequence))
            print(f"Process seq {sequence} ....")

            scan_paths = os.path.join(self.data_path, "sequences", str(sequence), "velodyne")
            label_paths = os.path.join(self.data_path, "sequences", str(sequence), "labels")
            pose_file = os.path.join(self.data_path, "sequences", str(sequence), "poses.txt")
            calib_file = os.path.join(self.data_path, "sequences", str(sequence), "calib.txt")

            calib_file = parse_calibration(calib_file)
            self.seq_poses = parse_poses(pose_file, calib_file)

            self.seq_scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_paths)) for f in fn if ".bin" in f]
            self.seq_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(label_paths)) for f in fn if ".label" in f]
            self.seq_scan_names.sort()
            self.seq_label_names.sort()

            assert len(self.seq_label_names) == len(self.seq_scan_names) and len(self.seq_scan_names) == len(self.seq_poses)

            for f_id in trange(len(self.seq_scan_names)):
                current_xyzi, current_semlabel, current_inslabel, current_pose = self.get_frame_data(f_id)
                motionflow = np.zeros(shape=(current_xyzi.shape[0], 3))

                if f_id - FRAME_DIFF < 0:
                    if FLAG_save_file:
                        self.save_motionflow_vector_to_file(f_id, motionflow, folder_name)
                    continue

                last_xyzi, last_semlabel, last_inslabel, last_pose = self.get_frame_data(f_id - FRAME_DIFF)
                last_xyzi_transformed = (np.linalg.inv(current_pose) @ (last_pose @ np.hstack((last_xyzi[:, :3], np.ones((last_xyzi.shape[0], 1)))).T)).T

                current_instance_ids, current_instance_pcnum = np.unique(current_inslabel, return_counts=True)
                last_instance_ids, last_instance_pcnum = np.unique(last_inslabel, return_counts=True)

                for ins_id, ins_pcnum in zip(current_instance_ids, current_instance_pcnum):
                    if (ins_id == 0) or (FLAG_mini_pcnumber_for_one_instance and ins_pcnum < self.mini_pcnumber_for_one_instance):
                        continue

                    current_ins_mask = (current_inslabel == ins_id)
                    current_ins_xyzi = current_xyzi[current_ins_mask]

                    if (ins_id in last_instance_ids) and (FLAG_mini_pcnumber_for_one_instance and (last_inslabel == ins_id).sum() > 0.2 * self.mini_pcnumber_for_one_instance):
                        last_ins_xyzi = last_xyzi_transformed[last_inslabel == ins_id]
                        flags = 'two_instance'

                    else:
                        #todo:如果当前帧的instance在last_frame中没有，那么就把当前帧instance的所有点都设置为0，点数小于mini_pcnumber_for_one_instance的情况同理，并且记录一下有多少这种情况，方便后续统计
                        #todo: 这种方法合适吗？
                        if ins_id in last_instance_ids:
                            temp_content = 'in last_instance_ids'
                        else:
                            temp_content = 'not in last_instance_ids'
                        last_inslabel_sum = (last_inslabel == ins_id).sum()
                        log = f'sequence:{sequence} f_id:{f_id} ins_id:{ins_id} ins_pcnum:{ins_pcnum} last_inslabel_sum:{last_inslabel_sum}---{temp_content}\n'
                        self.logs.append(log)
                        # if FLAG_save_file:
                        #     with open('unmached.txt', 'a') as f:
                        #         f.write(f'sequence:{sequence} f_id:{f_id} ins_id:{ins_id} ins_pcnum:{ins_pcnum} last_inslabel_sum:{last_inslabel_sum} {temp_content}\n')
                        flags = 'unmatched'
                        continue

                    trans = self.caculate_the_releative_pose(current_ins_xyzi[:, :3], last_ins_xyzi[:, :3], flags)
                    # ic(trans.fitness)
                    tmp_tranform_xyz = (trans.transformation @ np.hstack((current_ins_xyzi[:, :3], np.ones((current_ins_xyzi.shape[0], 1)))).T).T
                    ins_flow_xyz = current_ins_xyzi[:, :3] - tmp_tranform_xyz[:, :3]
                    # ic(ins_flow_xyz)
                    motionflow[current_ins_mask] = ins_flow_xyz

                if FLAG_save_file:
                    self.save_motionflow_vector_to_file(f_id, motionflow, folder_name)
            end = time.time()
            time_cost = (end - start) / len(self.seq_scan_names)
            print(f'time cost per seq mean:{time_cost}')

        if FLAG_save_log and len(self.logs) != 0 and FLAG_save_file:
            with open('unmatched_log.txt', 'a') as f:
                f.write(str(datetime.now()) + '\n')
                f.write(''.join(self.logs))


if __name__ == "__main__":
    FLAG_save_file = True  # default:True
    FLAG_save_log = False  # default:False
    FRAME_DIFF = 1  # default:1
    FLAG_add_ego_motion = False  # default:False

    FLAG_debug_ICP = False  # default:False
    FLAG_mini_pcnumber_for_one_instance = True  # default:True

    proSemKitti = ProcessSemanticKITTI(process_split='valid')
    proSemKitti.process_sequences_from_gt(add_ego_motion=FLAG_add_ego_motion)
    # proSemKitti.process_sequences_from_dbscan()
    # proSemKitti.process_sequences_from_all_instance()
