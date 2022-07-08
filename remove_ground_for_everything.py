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
        self.process_sequences_list = self.semantickitti_cfg['split'][process_split]
        print("process_sequences_list:", self.process_sequences_list)
        self.save_file = save_file
        self.save_dir_name = save_dir_name
        self.velo_files = []
        self.label_files = []

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

    def load_mask(self, mask_file_id):
        """
        Loads mask data
        """
        mask_data = np.fromfile(self.non_ground_mask_files[mask_file_id], dtype=np.uint32).reshape(-1)
        non_ground_mask = (mask_data != 9)
        return non_ground_mask

    def remove_ground_for_velo(self):
        for sequence in self.process_sequences_list:
            sequence = str(sequence).zfill(2)
            print(f"Processing sequence {sequence}")

            self.velo_paths = os.path.join(self.data_root, 'sequences', sequence, 'velodyne')
            self.non_ground_mask_paths = os.path.join(self.non_ground_mask_root, sequence, "ground_labels")

            self.non_ground_mask_files = self.get_mask_files(self.non_ground_mask_paths)
            self.velo_files = self.get_velo_files(self.velo_paths)

            assert len(self.non_ground_mask_files) == len(self.velo_files)

            for id in tqdm(range(len(self.velo_files))):
                velo_data = self.load_velo(id)
                non_ground_mask = self.load_mask(id)

                assert len(velo_data) == len(non_ground_mask), "error!!! velo file unmatched with mask file"

                non_ground_velo = velo_data[non_ground_mask]
                #todo 存文件

    def remove_ground_for_label(self):
        pass


if __name__ == "__main__":
    data_root = "/share/sgb/semantic_kitti/dataset/"
    non_ground_mask_root = "/share/sgb/kitti-ground"
    semantickitti_cfg = "./semantic-kitti.yaml"
    process_split = 'train'
    save_file = False
    save_dir_name = "non_ground_"

    remove_ground = RemoveGroundForEverything(data_root, non_ground_mask_root, semantickitti_cfg, process_split, save_file, save_dir_name=save_dir_name)
    remove_ground.remove_ground_for_velo()
