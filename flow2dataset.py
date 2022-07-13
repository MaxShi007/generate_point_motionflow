from glob import glob
from flow2residual import Flow2Residual
import numpy as np
import os
from tqdm import tqdm
from icecream import ic
import glob
import yaml


class Flow2Dataset(Flow2Residual):

    def __init__(self, config_file='flow2some.yaml'):
        super().__init__(config_file)
        self.save_dataset = self.config['save_dataset']
        self.save_dataset_root = self.config['save_dataset_root']
        self.save_dataset_dir_name = self.config['save_dataset_dir_name']
        self.non_ground_mask = self.config['non_ground_mask']
        self.non_ground_mask_root = self.config['non_ground_mask_root']
        self.velodyne_dir = self.config['velodyne_dir']
        self.labels_dir = self.config['labels_dir']
        self.flow_dir = self.config['flow_dir']

    def load_velo_data(self, velo_file_id):
        """
        Loads velo data
        """
        velo_data = np.fromfile(self.velo_files[velo_file_id], dtype=np.float32).reshape(-1, 4)
        return velo_data

    def load_flow_data(self, velo_file_id):
        """
        Loads flow data
        """
        path = self.flow_files[velo_file_id]
        flow_data = np.load(path)
        # print(flow_data)
        flow_data = (-flow_data)  # 原本生成的flow应该是以current作为基点，last指向current的flow，但我们是要让网络估计current作为基点，current指向last的flow
        # print(flow_data)
        return flow_data

    def save_nsfp_flow_dataset_file(self, current_point, last_point, flow_gt, velo_file_id, save_path, sequence):

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        file_id = str(velo_file_id).zfill(6)
        file_name = f'{sequence}_{file_id}.npz'
        file_path = os.path.join(save_path, file_name)
        np.savez(file_path, current_point=current_point, last_point=last_point, flow_gt=flow_gt)
        print(file_path)

    def load_nsfp_flow_dataset_file(self, file_path):
        data = np.load(file_path)
        return data['current_point'], data['last_point'], data['flow_gt']

    def generate_dataset(self):
        for sequence in self.process_sequences_list:
            sequence = str(sequence).zfill(2)
            print(f"Processing sequence {sequence}")

            flow_paths = os.path.join(self.data_path, 'sequences', sequence, self.flow_dir)
            label_paths = os.path.join(self.data_path, 'sequences', sequence, self.labels_dir)
            velo_paths = os.path.join(self.data_path, 'sequences', sequence, self.velodyne_dir)
            # save_paths = os.path.join(self.data_path, 'sequences', sequence, self.save_dir_name)

            self.flow_files = self.get_flow_files(flow_paths)
            self.label_files = self.get_label_files(label_paths)
            self.velo_files = self.get_velo_files(velo_paths)

            assert len(self.flow_files) == len(self.label_files) and len(self.flow_files) == len(self.velo_files)

            for velo_file_id in tqdm(range(1, len(self.velo_files))):
                current_velo_data = self.load_velo_data(velo_file_id)[:, :3]
                last_velo_data = self.load_velo_data(velo_file_id - 1)[:, :3]
                flow_gt = self.load_flow_data(velo_file_id)
                # ic(current_velo_data,last_velo_data,flow_gt)
                if self.save_dataset:

                    save_path = os.path.join(self.save_dataset_root, self.save_dataset_dir_name)
                    self.save_nsfp_flow_dataset_file(current_velo_data, last_velo_data, flow_gt, velo_file_id, save_path, sequence)

    def remove_nsfp_ground_point(self):
        dataset = os.path.join(self.save_dataset_root, self.save_dataset_dir_name)
        # print(dataset_path)
        dataset_paths = sorted(glob.glob(os.path.join(dataset, '*.npz')))
        # print(dataset_paths)
        save_path = os.path.join(self.save_dataset_root, self.save_dataset_dir_name + '_non_ground_point')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for dataset_path in tqdm(dataset_paths):
            print(f'remove ground point : {dataset_path}')
            current_point, last_point, flow_gt = self.load_nsfp_flow_dataset_file(dataset_path)
            # print(current_point.shape,last_point.shape,flow_gt.shape)
            if self.non_ground_mask:
                sequence = dataset_path.split('/')[-1].split('_')[0]
                current_label_file_name = dataset_path.split('/')[-1].split('_')[1].replace('.npz', '.label')
                current_non_ground_mask_path = os.path.join(self.non_ground_mask_root, f'{sequence}/ground_labels/{current_label_file_name}')

                last_label_file_name = str(int(os.path.splitext(dataset_path.split('/')[-1].split('_')[1])[0]) - 1).zfill(6) + '.label'
                last_non_ground_mask_path = os.path.join(self.non_ground_mask_root, f'{sequence}/ground_labels/{last_label_file_name}')

                # ic(current_non_ground_mask_path,last_non_ground_mask_path)

                current_non_ground_mask_data = np.fromfile(current_non_ground_mask_path, dtype=np.uint32).reshape(-1)
                last_non_ground_mask_data = np.fromfile(last_non_ground_mask_path, dtype=np.uint32).reshape(-1)
                current_sem_label = current_non_ground_mask_data & 0xFFFF
                last_sem_label = last_non_ground_mask_data & 0xFFFF
                current_non_ground_mask = (current_sem_label != 9)
                last_non_ground_mask = (last_sem_label != 9)
                flow_gt_non_ground_mask = current_non_ground_mask

                # ic(np.unique(current_non_ground_mask,return_counts=True),np.unique(last_non_ground_mask,return_counts=True))

            else:
                current_non_ground_mask = current_point[:, 2] > -1.4
                last_non_ground_mask = last_point[:, 2] > -1.4
                flow_gt_non_ground_mask = current_non_ground_mask

            # assert(current_non_ground_mask.sum()<86000 and last_non_ground_mask.sum()<86000)

            current_point = current_point[current_non_ground_mask]
            last_point = last_point[last_non_ground_mask]
            flow_gt = flow_gt[flow_gt_non_ground_mask]
            # print(current_point.shape,last_point.shape,flow_gt.shape)
            file_name = dataset_path.split('/')[-1]
            save_file_path = os.path.join(save_path, file_name)
            # print(save_file_path)
            #todo保存为nsfp格式
            np.savez(save_file_path, current_point=current_point, last_point=last_point, flow_gt=flow_gt)
            print(save_file_path)
            #todo保存为pointpwc格式


# class FromNonGroundGenerateDataset(Flow2Dataset):
#     def __init__(self,data_path,config,split,save_file):
#         self.data_path=data_path
#         self.config=yaml.safe_load(open(config, "r"))
#         self.process_sequences_list=self.config["split"][split]
#         print("process_sequences_list:", self.process_sequences_list)
#         self.save_file = save_file

#     def from_non_ground_generate_dataset(self, process_flow_dir):
#         for sequence in self.process_sequences_list:
#             sequence = str(sequence).zfill(2)
#             print(f"Processing sequence {sequence}")

#             label_paths = os.path.join(self.data_path, 'sequences', sequence, 'non_ground_labels')
#             velo_paths = os.path.join(self.data_path, 'sequences', sequence, 'non_ground_velodyne')
#             flow_paths = os.path.join(self.data_path, 'sequences', sequence, process_flow_dir)

#             self.flow_files = self.get_flow_files(flow_paths)
#             self.label_files = self.get_label_files(label_paths)
#             self.velo_files = self.get_velo_files(velo_paths)

#             assert len(self.flow_files) == len(self.label_files) and len(self.flow_files) == len(self.velo_files)

if __name__ == "__main__":
    semantic_kitti = Flow2Dataset(config_file='flow2some.yaml')
    semantic_kitti.generate_dataset()
    # semantic_kitti.remove_nsfp_ground_point()
