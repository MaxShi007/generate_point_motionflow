import numpy as np
import yaml
import os
from tqdm import tqdm
from icecream import ic
import matplotlib.pyplot as plt


class Flow2Residual:
    """
    Converts flow to residual
    """

    def __init__(self, config_file='flow2some.yaml'):

        self.config = yaml.load(open(config_file), Loader=yaml.FullLoader)

        semantickitti_cfg_path = self.config['semantickitti_cfg']
        self.semantickitti_cfg = yaml.load(open(semantickitti_cfg_path), Loader=yaml.FullLoader)

        self.data_path = self.config['data_path']

        process_split = self.config['process_split']
        assert process_split in ['train', 'valid', 'test']
        self.process_sequences_list = self.semantickitti_cfg['split'][process_split]
        # print(self.process_sequences_list)
        self.save_file = self.config['save_file']
        self.visualization = self.config['visualization']
        self.save_dir_name = self.config['save_dir_name']
        self.visualization_dir_name = self.config['visualization_dir_name']
        self.flow_dir = self.config['flow_dir']

    def get_flow_files(self, flow_paths):
        """
        Gets flow files
        """
        files = [os.path.join(root, file) for root, dir, files in os.walk(flow_paths) for file in files]
        files.sort()
        return files

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

    def load_flow_value(self, flow_file_id, normalize=True):
        """
        Loads flow vertex

        Returns:
            [x,y,z,flow_value]
        """
        flow = np.load(self.flow_files[flow_file_id])
        label = np.fromfile(self.label_files[flow_file_id], dtype=np.uint32).reshape(-1)
        velo = np.fromfile(self.velo_files[flow_file_id], dtype=np.float32).reshape(-1, 4)
        assert flow.shape[0] == label.shape[0] and flow.shape[0] == velo.shape[0]

        # TODO gen flow values
        # x,y,z L2
        flow_value = np.linalg.norm(flow, ord=2, axis=1)

        # # x,y L2
        # flow_value = np.linalg.norm(flow[:, :2], ord=2, axis=1)

        # TODO how to normalize?
        if normalize:
            flow_value = flow_value / np.max(flow_value)
        #
        flow_vertex = np.zeros((flow.shape[0], 4))
        flow_vertex[:, :3] = velo[:, :3]
        flow_vertex[:, 3] = flow_value

        # ic(flow_value.shape)
        # ic(flow_value.min(),flow_value.max())
        # ic(np.unique(flow_value,return_counts=True))
        # ic(flow_vertex)

        return flow_vertex

    def range_projection(self, flow_vertex, range_image):
        """
        Range projection
        """
        height = range_image['height']
        width = range_image['width']
        fov_up = range_image['fov_up']
        fov_down = range_image['fov_down']
        max_range = range_image['max_range']
        min_range = range_image['min_range']

        fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
        fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

        depth = np.linalg.norm(flow_vertex[:, :3], 2, axis=1)
        flow_vertex = flow_vertex[(depth > min_range) & (depth < max_range)]
        depth = depth[(depth > min_range) & (depth < max_range)]

        scan_x = flow_vertex[:, 0]
        scan_y = flow_vertex[:, 1]
        scan_z = flow_vertex[:, 2]
        flow_value = flow_vertex[:, 3]

        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)
        # print(yaw.shape)

        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        proj_x *= width  # in [0.0, W]
        proj_y *= height  # in [0.0, H]

        proj_x = np.floor(proj_x)
        proj_x = np.minimum(width - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
        proj_x_orig = np.copy(proj_x)

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(height - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
        proj_y_orig = np.copy(proj_y)

        order = np.argsort(depth)[::-1]
        flow_value = flow_value[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]
        # print(proj_y,proj_x)
        # print(flow_value)

        proj_flow_value = np.full((height, width), 0, dtype=np.float32)
        proj_flow_value[proj_y, proj_x] = flow_value
        # ic(np.unique(proj_flow_value,return_counts=True))
        # ic(flow_value.max(),flow_value.min())

        return proj_flow_value

    def save_flow_image_file(self, flow_image, flow_file_id, save_paths):
        """
        Saves flow image file
        """
        if not os.path.exists(save_paths):
            os.makedirs(save_paths)

        file_id = str(flow_file_id).zfill(6)
        file_path = os.path.join(save_paths, file_id)
        # ic(file_path)
        np.save(file_path, flow_image)

    def visualize(self, flow_image, flow_file_id, visualiaztion_paths):
        """
        Visualize flow image
        """
        if not os.path.exists(visualiaztion_paths):
            os.makedirs(visualiaztion_paths)

        file_id = str(flow_file_id).zfill(6)
        file_path = os.path.join(visualiaztion_paths, file_id)
        # ic(file_path)

        fig = plt.figure(frameon=False, figsize=(16, 10))
        fig.set_size_inches(20.48, 0.64)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(flow_image, vmin=0, vmax=1)
        plt.savefig(file_path)
        plt.close()

    def generate_residual(self):
        """
        Generates residual from flow
        """
        for sequence in self.process_sequences_list:
            sequence = str(sequence).zfill(2)
            print(f"Processing sequence {sequence}")

            #get flow paths
            flow_paths = os.path.join(self.data_path, 'sequences', sequence, self.flow_dir)
            label_paths = os.path.join(self.data_path, 'sequences', sequence, 'labels')
            velo_paths = os.path.join(self.data_path, 'sequences', sequence, 'velodyne')
            save_paths = os.path.join(self.data_path, 'sequences', sequence, self.save_dir_name)
            visualization_paths = os.path.join(self.data_path, 'sequences', sequence, self.visualization_dir_name)

            #get flow files
            self.flow_files = self.get_flow_files(flow_paths)
            self.label_files = self.get_label_files(label_paths)
            self.velo_files = self.get_velo_files(velo_paths)

            assert len(self.flow_files) == len(self.label_files) and len(self.flow_files) == len(self.velo_files)

            for flow_file_id in tqdm(range(len(self.flow_files))):
                flow_vertex = self.load_flow_value(flow_file_id, normalize=False)
                flow_image = self.range_projection(flow_vertex, self.config['range_image'])

                if self.save_file:
                    self.save_flow_image_file(flow_image, flow_file_id, save_paths)

                if self.visualization:
                    self.visualize(flow_image, flow_file_id, visualization_paths)


if __name__ == "__main__":

    residual = Flow2Residual(config_file='flow2some.yaml')
    residual.generate_residual()