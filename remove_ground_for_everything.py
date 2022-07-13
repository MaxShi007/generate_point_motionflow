#该脚本为每个sequence移除velo，label的ground，因为pointpwc生成的flow是不包括地面点的，所以在训练4dmos的时候也需要把地面去掉
import numpy as np
import os
from tqdm import tqdm
import yaml


class RemoveGroundForEverything(object):

    def __init__(self, data_root, non_ground_mask_root, semantickitti_cfg, process_split, save_file, save_dir_name) -> None:
        self.data_root = data_root
        self.non_ground_mask_root = non_ground_mask_root
        self.semantickitti_cfg = yaml.safe_load(open(semantickitti_cfg, "r"))
        assert process_split in ["train", "valid", "test"]
        self.process_sequences_list = self.semantickitti_cfg['split'][process_split]
        print("process_sequences_list:", self.process_sequences_list)
        self.save_file = save_file
        self.save_dir_name = save_dir_name
        self.velo_files = []
        self.label_files = []
        self.flow_files = []

    def get_label_files(self, label_paths):
        """
        Gets label files
        """
        files = [os.path.join(root, file) for root, dir, files in os.walk(label_paths) for file in files]
        files.sort()
        return files

    def get_velo_files(self, velo_paths):
        """
        Gets velo files
        """
        files = [os.path.join(root, file) for root, dir, files in os.walk(velo_paths) for file in files]
        files.sort()
        return files

    def get_flow_files(self, flow_paths):
        """
        Gets flow files
        """
        files = [os.path.join(root, file) for root, dir, files in os.walk(flow_paths) for file in files]
        files.sort()
        return files

    def get_mask_files(self, mask_paths):
        """
        Gets mask files
        """
        files = [os.path.join(root, file) for root, dir, files in os.walk(mask_paths) for file in files]
        files.sort()
        return files

    def load_velo(self, velo_file_id):
        """
        Loads velo data
        """
        velo_data = np.fromfile(self.velo_files[velo_file_id], dtype=np.float32).reshape(-1, 4)
        return velo_data

    def load_label(self, label_file_id):
        """
        Loads label data
        """
        label_data = np.fromfile(self.label_files[label_file_id], dtype=np.uint32).reshape(-1)
        return label_data

    def load_flow(self, flow_file_id):
        """
        Loads flow data
        """
        flow_data = np.load(self.flow_files[flow_file_id])
        return flow_data

    def load_mask(self, mask_file_id):
        """
        Loads mask data
        """
        mask_data = np.fromfile(self.non_ground_mask_files[mask_file_id], dtype=np.uint32).reshape(-1)
        sem_mask_data = mask_data & 0xFFFF
        non_ground_mask = (sem_mask_data != 9)
        return non_ground_mask

    def remove_ground_for_velo(self):
        for sequence in self.process_sequences_list:
            sequence = str(sequence).zfill(2)
            print(f"Processing sequence {sequence}")

            save_dir = os.path.join(self.data_root, "sequences", sequence, self.save_dir_name + "velodyne")

            velo_paths = os.path.join(self.data_root, 'sequences', sequence, 'velodyne')
            non_ground_mask_paths = os.path.join(self.non_ground_mask_root, sequence, "ground_labels")

            self.non_ground_mask_files = self.get_mask_files(non_ground_mask_paths)
            self.velo_files = self.get_velo_files(velo_paths)

            assert len(self.non_ground_mask_files) == len(self.velo_files)

            for id in tqdm(range(len(self.velo_files))):
                velo_data = self.load_velo(id)
                non_ground_mask = self.load_mask(id)

                assert len(velo_data) == len(non_ground_mask), "error!!! velo file unmatched with mask file"

                non_ground_velo = velo_data[non_ground_mask]

                if save_file:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    non_ground_velo_file = self.velo_files[id].replace(velo_paths, save_dir)
                    non_ground_velo.tofile(non_ground_velo_file)
                else:
                    print(non_ground_velo)

    def remove_ground_for_label(self):
        for sequence in self.process_sequences_list:
            sequence = str(sequence).zfill(2)
            print(f"Processing sequence {sequence}")

            save_dir = os.path.join(self.data_root, "sequences", sequence, self.save_dir_name + "labels")

            label_paths = os.path.join(self.data_root, 'sequences', sequence, 'labels')
            non_ground_mask_paths = os.path.join(self.non_ground_mask_root, sequence, "ground_labels")

            self.non_ground_mask_files = self.get_mask_files(non_ground_mask_paths)
            self.label_files = self.get_label_files(label_paths)

            assert len(self.non_ground_mask_files) == len(self.label_files)

            for id in tqdm(range(len(self.label_files))):
                label_data = self.load_label(id)
                non_ground_mask = self.load_mask(id)

                assert len(label_data) == len(non_ground_mask), "error!!! label file unmatched with mask file"

                non_ground_label = label_data[non_ground_mask]

                if save_file:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    non_ground_label_file = self.label_files[id].replace(label_paths, save_dir)
                    non_ground_label.tofile(non_ground_label_file)
                else:
                    print(non_ground_label)

    #todo 把flow的地面点也移除了，这样我就有完整的 无地面点数据，随意组合生成sceneflow训练要用的数据集格式
    def remove_ground_for_flow(self, flow_dir_name):
        for sequence in self.process_sequences_list:
            sequence = str(sequence).zfill(2)
            print(f"Processing sequence {sequence}")

            save_dir = os.path.join(self.data_root, "sequences", sequence, self.save_dir_name + flow_dir_name)

            flow_paths = os.path.join(self.data_root, 'sequences', sequence, flow_dir_name)
            non_ground_mask_paths = os.path.join(self.non_ground_mask_root, sequence, "ground_labels")

            self.non_ground_mask_files = self.get_mask_files(non_ground_mask_paths)
            self.flow_files = self.get_flow_files(flow_paths)

            assert len(self.non_ground_mask_files) == len(self.flow_files)

            for id in tqdm(range(len(self.flow_files))):
                flow_data = self.load_flow(id)
                non_ground_mask = self.load_mask(id)

                assert len(flow_data) == len(non_ground_mask), "error!!! flow file unmatched with mask file"

                non_ground_flow = flow_data[non_ground_mask]

                if save_file:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    non_ground_flow_file = self.flow_files[id].replace(flow_paths, save_dir)
                    np.save(non_ground_flow_file, non_ground_flow)
                else:
                    print(non_ground_flow.shape)
                    print(non_ground_flow)


if __name__ == "__main__":
    data_root = "/share/sgb/semantic_kitti/dataset/"
    non_ground_mask_root = "/share/sgb/kitti-ground"
    semantickitti_cfg = "./semantic-kitti.yaml"
    flow_dir_name = "motionflow_egomotion_4DMOS_POSES_1"  # 去掉哪个flow的地面点
    process_split = 'train'  # train  valid  test

    save_file = True
    save_dir_name = "non_ground_"  # 为了适配non_ground_velodyne  non_ground_label，提供一个统一的命名头

    remove_ground = RemoveGroundForEverything(data_root, non_ground_mask_root, semantickitti_cfg, process_split, save_file, save_dir_name=save_dir_name)
    remove_ground.remove_ground_for_velo()
    remove_ground.remove_ground_for_label()
    remove_ground.remove_ground_for_flow(flow_dir_name=flow_dir_name)
