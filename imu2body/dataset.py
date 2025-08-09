# Copyright (c) Facebook, Inc. and its affiliates.
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

import numpy as np
import pickle
import torch
import trimesh
import pandas as pd
import json
import bisect
from fairmotion.utils import constants
from IPython import embed
from imu2body.preprocess import load_data as load_amass_data
from imu2body.preprocess_gimo import load_data as load_gimo_data
# from preprocess_bvh import *
from imu2body.functions import invert_T, xyz_to_transform_matrices, transform_matrices_to_xyz
from torchvision import transforms as tv_transforms
import torch.utils.data as data
import constants.motion_data as motion_constants 
from torch.utils.data import Dataset, DataLoader
from interaction.contact import *
from sklearn.neighbors import BallTree
import open3d as o3d
from tqdm import tqdm

class MotionData(Dataset):
    def __init__(self, dataset_path="", device="cuda", data=None, base_dir="", mean=None, std=None, debug=False):
        """
        Args:
            bvh_path (string): Path to the bvh files.
            seq_len (int): The max len of the sequence for interpolation.
        """

        self.debug = debug 
        if isinstance(dataset_path, list):
            print("IMU2Body: got a list of pkl files")

            self.data = {}

            for dataset_file in dataset_path:
                print(f"loading {dataset_file}")				
                with open(dataset_file, "rb") as file:
                    current_dict = pickle.load(file)
                if not self.data:
                    for key in current_dict.keys():
                        self.data[key] = []
                for key in self.data.keys():
                    self.data[key].append(current_dict[key])				
                del current_dict

            for key in self.data.keys():
                self.data[key] = np.concatenate(self.data[key], axis=0)

        elif 'pkl' in dataset_path:
            print(f"loading {dataset_path}")
            with open(dataset_path, "rb") as file:
                self.data = pickle.load(file)
        elif 'npz' in dataset_path:
            self.data = load_amass_data([dataset_path], base_dir="../data/amass/")
            
        # load dimension info
        self.load_data_dict()

        # set x_mean and x_std for pos scaling
        global_p = self.data['global_p']
        x_mean = np.mean(global_p.reshape([global_p.shape[0], global_p.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)
        x_std = np.std(global_p.reshape([global_p.shape[0], global_p.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)
        self.data['x_mean'] = x_mean
        self.data['x_std'] = x_std
        
        # normalize 
        if mean is None or std is None:
            self.mean = np.mean(self.data['input_seq'], axis=(0,1))
            self.std = np.std(self.data['input_seq'], axis=(0,1))
        else:
            self.mean = mean
            self.std = std

        self.device = device
        
    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        idx_ = None
        if self.debug:
            idx_ = 0
        else:
            idx_ = idx

        norm_input_seq = (self.data['input_seq'][idx_] - self.mean) / (
            self.std + constants.EPSILON
        )	# Seq_normalization

        sample = {}
        sample['input_seq'] = norm_input_seq.astype(dtype=np.float32)
        sample['mid_seq'] = self.data['mid_seq'][idx_].astype(dtype=np.float32)
        sample['tgt_seq'] = self.data['tgt_seq'][idx_].astype(dtype=np.float32)

        sample['global_p'] = self.data['global_p'][idx_].astype(dtype=np.float32)
        sample['root'] = self.data['root'][idx_].astype(dtype=np.float32)

        # this is for testing and visualization		
        sample['local_rot'] = self.data['local_rot'][idx_].astype(dtype=np.float32)
        sample['head_start'] = self.data['head_start'][idx_].astype(dtype=np.float32)
        sample['contact_label'] = self.data['contact_label'][idx_].astype(dtype=np.float32)

        return sample

    def get_x_mean_and_std(self):
        return self.data['x_mean'], self.data['x_std']
    
    def get_seq_length(self):
        return motion_constants.preprocess_window

    def load_data_dict(self):
        self.num_frames, seq_len, input_seq_dim = self.data['input_seq'].shape
        assert seq_len == motion_constants.preprocess_window, "seq length should be same as window size in preprocessing! check preprocess.py"
        
        mid_seq_dim = self.data['mid_seq'].shape[2]
        output_seq_dim = self.data['tgt_seq'].shape[2]

        self.dim_dict = {}
        self.dim_dict['input_dim'] = input_seq_dim
        self.dim_dict['mid_dim'] = mid_seq_dim
        self.dim_dict['output_dim'] = output_seq_dim

    def get_data_dict(self):
        return self.dim_dict	

class CustomMotionData(Dataset):
    def __init__(self, motion_clip_path, custom_config, mean, std, device="cuda", debug=False):
        """
        Args:
            bvh_path (string): Path to the bvh files.
            seq_len (int): The max len of the sequence for interpolation.
        """

        base_dir = "../data/amass/"
        if 'npz' not in motion_clip_path:
            base_dir = ""

        self.data, _ = load_amass_data(base_dir=base_dir, file_list=[motion_clip_path], custom_config=custom_config)		

        # load dimension info
        self.load_data_dict()

        self.debug = debug 

        self.mean = mean
        self.std = std 
        
        self.device = device
        
        self.config = custom_config

        # contact
        self.contact = {}
        self.contact[0] = 0.0


    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        idx_ = None
        if self.debug:
            idx_ = 0
        else:
            idx_ = idx

        # apply contact 
        frame_idx_from_start = self.config['offset'] * idx 
        frame_key, contact_height = get_height_offset_current_frame(self.contact, frame_idx_from_start)

        self.data['input_seq'][idx_][:,0] -= contact_height
        norm_input_seq = (self.data['input_seq'][idx_] - self.mean) / (
            self.std + constants.EPSILON
        )

        sample = {}
        sample['input_seq'] = norm_input_seq.astype(dtype=np.float32)
        sample['mid_seq'] = self.data['mid_seq'][idx_].astype(dtype=np.float32)
        sample['tgt_seq'] = self.data['tgt_seq'][idx_].astype(dtype=np.float32)

        sample['global_p'] = self.data['global_p'][idx_].astype(dtype=np.float32)

        sample['local_rot'] = self.data['local_rot'][idx_].astype(dtype=np.float32)
        sample['head_start'] = self.data['head_start'][idx_].astype(dtype=np.float32)

        return sample
    
    def get_x_mean_and_std(self):
        return self.data['x_mean'], self.data['x_std']
    
    def get_seq_length(self):
        return motion_constants.preprocess_window

    def load_data_dict(self):
        self.num_frames, seq_len, input_seq_dim = self.data['input_seq'].shape
        assert seq_len == motion_constants.preprocess_window, "seq length should be same as window size in preprocessing! check preprocess.py"
        
        mid_seq_dim = self.data['mid_seq'].shape[2]
        output_seq_dim = self.data['tgt_seq'].shape[2]

        self.dim_dict = {}
        self.dim_dict['input_dim'] = input_seq_dim
        self.dim_dict['mid_dim'] = mid_seq_dim
        self.dim_dict['output_dim'] = output_seq_dim

    def get_data_dict(self):
        return self.dim_dict	


class RealMotionData(Dataset):
    def __init__(self, input_dict, mean, std, custom_config=None, device="cuda", debug=False):
        """
        Args:
            bvh_path (string): Path to the bvh files.
            seq_len (int): The max len of the sequence for interpolation.
        """

        # load dimension info
        self.data = input_dict
        self.load_data_dict()

        self.debug = debug 

        self.mean = mean
        self.std = std 
        
        self.device = device
        if custom_config is not None:
            self.config = custom_config 

        # contact
        self.contact = {}
        self.contact[0] = 0.0


    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        idx_ = None
        if self.debug:
            idx_ = 0
        else:
            idx_ = idx

        # apply contact 
        frame_idx_from_start = self.config['offset'] * idx 
        frame_key, contact_height = get_height_offset_current_frame(self.contact, frame_idx_from_start)


        self.data['input_seq'][idx_][:,0] -= contact_height

        norm_input_seq = (self.data['input_seq'][idx_] - self.mean) / (
            self.std + constants.EPSILON
        )

        # real data does not have gt!
        sample = {}
        sample['input_seq'] = norm_input_seq.astype(dtype=np.float32)
        sample['head_start'] = self.data['head_start'][idx_].astype(dtype=np.float32)

        return sample
    
    
    def get_seq_length(self):
        return motion_constants.preprocess_window

    def load_data_dict(self):
        self.num_frames, seq_len, input_seq_dim = self.data['input_seq'].shape
        assert seq_len == motion_constants.preprocess_window, "seq length should be same as window size in preprocessing!"
        

        self.dim_dict = {}
        self.dim_dict['input_dim'] = input_seq_dim
        self.dim_dict['mid_dim'] = 12
        self.dim_dict['output_dim'] = 135

    def get_data_dict(self):
        return self.dim_dict	

def get_loader_gimo(
    data_root=None,
    batch_size=16,
    training=False,
    drop_last=True
):
    """Returns data loader for custom dataset.
    Args:
        dataset_path: path to pickled numpy dataset
        device: Device in which data is loaded -- 'cpu' or 'cuda'
        batch_size: mini-batch size.
    Returns:
        data_loader: data loader.
    """
 
    dataset = GIMODataset(data_root, training)

    data_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=training, num_workers=16, drop_last=drop_last
    )
    return data_loader

def get_loader_training(
    data_root=None,
    batch_size=16,
    training=False,
    drop_last=True
):
    """Returns data loader for custom dataset.
    Args:
        dataset_path: path to pickled numpy dataset
        device: Device in which data is loaded -- 'cpu' or 'cuda'
        batch_size: mini-batch size.
    Returns:
        data_loader: data loader.
    """

    mode = 'train' if training == True else 'test'

    if training:
        dataset = TrainingDataset(data_root, mode)
        data_loader = DataLoader(
            # BUG num_workers
            dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=drop_last
        )
    else:
        data_loader = DataLoader(
            # BUG num_workers
            dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=drop_last
        )
    return data_loader

def get_loader_validation(
    data_root=None,
    batch_size=16,
    dataset=None,
):
    """Returns data loader for custom dataset.
    Args:
        dataset_path: path to pickled numpy dataset
        device: Device in which data is loaded -- 'cpu' or 'cuda'
        batch_size: mini-batch size.
    Returns:
        data_loader: data loader.
    """
    training = False
    mode = 'test'

    dataset = TrainingDataset(data_root, mode, test_only=True, test_dataset=dataset)

    data_loader = DataLoader(
        # BUG num_workers
        dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False
    )
    return data_loader


def get_loader(
    dataset_path,
    batch_size=100,
    device="cuda",
    mean=None,
    std=None,
    shuffle=False,
    drop_last=True
):
    """Returns data loader for custom dataset.
    Args:
        dataset_path: path to pickled numpy dataset
        device: Device in which data is loaded -- 'cpu' or 'cuda'
        batch_size: mini-batch size.
    Returns:
        data_loader: data loader.
    """
    dataset = MotionData(dataset_path=dataset_path, device=device, mean=mean, std=std)

    data_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8,  drop_last=drop_last
    )
    return data_loader

def get_custom_loader(
    motion_clip_path,
    custom_config,
    mean,
    std,
    device="cuda"
):
    """Returns data loader for custom motion clip
    Args:
        dataset_path: path to pickled numpy dataset
        device: Device in which data is loaded -- 'cpu' or 'cuda'
        batch_size: mini-batch size.
    Returns:
        data_loader: data loader.
    """
    dataset = CustomMotionData(motion_clip_path=motion_clip_path, device=device, custom_config=custom_config, mean=mean, std=std)

    data_loader = DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False
    )
    return data_loader

def get_realdata_loader(
    input_dict,
    custom_config,
    mean,
    std,
    device="cuda"
):
    """Returns data loader for custom motion clip
    Args:
        dataset_path: path to pickled numpy dataset
        device: Device in which data is loaded -- 'cpu' or 'cuda'
        batch_size: mini-batch size.
    Returns:
        data_loader: data loader.
    """
    dataset = RealMotionData(input_dict=input_dict, custom_config=custom_config, device=device, mean=mean, std=std)

    # data loader
    data_loader = DataLoader(
        dataset=dataset, batch_size=1, shuffle=False, num_workers=8,  drop_last=False
    )
    return data_loader
  

class GIMODataset(data.Dataset):
    def __init__(self, dataroot, train=False):
        self.dataroot = dataroot
        self.train = train
        
        # NOTE: Hard coded
        self.input_seq_len = 40
        self.output_seq_len = 40
        self.fps = 30
        self.sample_points = 20000
        self.sigma = 0.1
        self.img_size = 224

        self.dataset_info = pd.read_csv(os.path.join(self.dataroot, 'dataset.csv'))
        self.parse_data_info()
        self.load_scene()
        self.load_imu()

        self.random_ori_list = [-180, -90, 0, 90]
        self.transform = tv_transforms.Compose([
            tv_transforms.Resize(self.img_size),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])
        self.load_data_dict()
        
        global_p = self.imu_data['global_p']
        x_mean = np.mean(global_p.reshape([global_p.shape[0], global_p.shape[1], -1]), axis=(0, 1), keepdims=True)
        x_std = np.std(global_p.reshape([global_p.shape[0], global_p.shape[1], -1]), axis=(0, 1), keepdims=True)
        self.x_mean = x_mean
        self.x_std = x_std
        
        # normalize 
        self.mean = np.mean(self.imu_data['input_seq'], axis=(0,1))
        self.std = np.std(self.imu_data['input_seq'], axis=(0,1))
 
    def __getitem__(self, index):
        #======================= imu parameters =======================
        input_seq = torch.from_numpy(self.imu_data['input_seq'][index]).float()
        mid_seq = torch.from_numpy(self.imu_data['mid_seq'][index]).float()
        tgt_seq = torch.from_numpy(self.imu_data['tgt_seq'][index]).float()
        global_p = torch.from_numpy(self.imu_data['global_p'][index]).float()
        contact_label = torch.from_numpy(self.imu_data['contact_label'][index]).float()
        start_frame, end_frame = int(self.imu_data['start_end'][index][0]), int(self.imu_data['start_end'][index][1])
        local_rot = torch.from_numpy(self.imu_data['local_rot'][index]).float()
        head_start = torch.from_numpy(self.imu_data['head_start'][index]).float()
        scene, seq, transform_path = self.find_scene_seq(index)
        #======================= imu parameters =======================
        
        img_list = os.listdir(os.path.join(self.dataroot, scene, seq, 'PV'))
        img_list.sort()

        imgs = []
        poses_input_idx = []

        random_ori = np.random.choice(self.random_ori_list)
        
        transform_info = json.load(open(os.path.join(self.dataroot, scene, seq, transform_path), 'r'))
        scale = transform_info['scale'] 
        transform_norm = np.loadtxt(os.path.join(self.dataroot, scene, 'scene_obj', 'transform_norm.txt')).reshape((4, 4))
        transform_norm[:3, 3] /= scale
         
        # for f in range(self.input_seq_len):
        #     pose_idx = start_frame + int(f * 30 / self.fps)
        #     poses_input_idx.append(pose_idx)
            
        #     #=============================prepare_image===========================
        #     img_data = Image.open(os.path.join(self.dataroot, scene, seq, 'PV', img_list[pose_idx])).convert('RGB') # Read input image
        #     img_data = self.transform(img_data)
        #     imgs.append(img_data)
        # 	#=============================prepare_image===========================
        # imgs = torch.stack(imgs, dim=0)

        #=============================Scene Point cloud=============================
        scene_points = self.scene_list['{}_{}'.format(scene, seq)]
        scene_points = scene_points[np.random.choice(range(len(scene_points)), self.sample_points)]
        scene_points = scene_points / scale
        scene_points = (transform_norm[:3, :3] @ scene_points.T + transform_norm[:3, 3:]).T #
        if self.train:
            scene_points += np.random.normal(loc=0, scale=self.sigma, size=scene_points.shape)
        #=============================Scene Point cloud=============================
        
        #=============================IMU Parameters=============================
        input_ = {}
        
        # norm_input_seq = (input_seq.float() - self.mean) / (
        # 	self.std + constants.EPSILON
        # )	# Seq_normalization
        input_['input_seq'] = input_seq.float()
        input_['mid_seq'] = mid_seq.float()
        input_['tgt_seq'] = tgt_seq.float()
        input_['global_p'] = global_p.float()
        input_['root'] = global_p[..., 0, :].float()
        input_['contact_label'] = contact_label.float()
        input_['local_rot'] = local_rot.float()
        input_['head_start'] = head_start.float()
        #=============================IMU Parameters=============================
        # Scene Points
        input_['scene_points'] =  torch.from_numpy(scene_points).float()
        # Input Images
        input_['imgs'] = []     
  
        return input_

    def __len__(self):
        return self.imu_data['input_seq'].shape[0]

    def load_imu(self):
        self.imu_data = {}
        self.imu_seq_info = []  # 新增列表，记录每个seq的元数据和索引范围
        for i, seq in enumerate(self.dataset_info['sequence_path']):
            if self.dataset_info['training'][i] != self.train: 
                continue # ignore the test/validation
            scene = self.dataset_info['scene'][i]
            start_frame = self.dataset_info['start_frame'][i]   
            end_frame = self.dataset_info['end_frame'][i]
            transform = self.dataset_info['transformation'][i]
            with open(os.path.join(self.dataroot, scene, seq, "IMU", f'imu_{start_frame}_{end_frame}.pkl'), 'rb') as f:
                imu_param = pickle.load(f)
                for k, v in imu_param.items():
                    if k not in self.imu_data.keys():
                        self.imu_data[k] = []
                    self.imu_data[k] += [v]
                    seq_info = v.shape[0]
                     
                self.imu_seq_info.append({
                    'scene': scene,
                    'seq': seq,
                    'transform': transform,
                    'length': seq_info 
                    })
        for k, v in self.imu_data.items():
            self.imu_data[k] = np.concat(v, axis=0)
        
        # 计算每个seq在合并后的数据中的索引范围
        current_idx = 0
        for seq_info in self.imu_seq_info:
            seq_length = seq_info['length']
            seq_info['start'] = current_idx
            seq_info['end'] = current_idx + seq_length
            current_idx += seq_length
            
        print('IMU information load done')

    def find_scene_seq(self, idx):
        starts = [seq_info["start"] for seq_info in self.imu_seq_info]
        pos = bisect.bisect_right(starts, idx) - 1
        
        if 0 <= pos < len(self.imu_seq_info):
            seq_info = self.imu_seq_info[pos]
            if seq_info["start"] <= idx < seq_info["end"]:
                return seq_info["scene"], seq_info["seq"], seq_info["transform"]
        
        return None, None
                 
    def load_data_dict(self):
        data_sample = self.__getitem__(0)
        seq_len, input_seq_dim = data_sample['input_seq'].shape
        assert seq_len == motion_constants.preprocess_window, "seq length should be same as window size in preprocessing! check preprocess.py"

        mid_seq_dim = data_sample['mid_seq'].shape[1]
        output_seq_dim = data_sample['tgt_seq'].shape[1]

        self.dim_dict = {}
        self.dim_dict['input_dim'] = input_seq_dim
        self.dim_dict['mid_dim'] = mid_seq_dim
        self.dim_dict['output_dim'] = output_seq_dim
        
    def get_data_dict(self):
        return self.dim_dict	
                
    def load_scene(self):
        self.scene_list = {}
        for i, seq in enumerate(self.dataset_info['sequence_path']):
            if self.dataset_info['training'][i] != self.train: 
                continue # ignore the test/validation
            scene = self.dataset_info['scene'][i]
            #start_frame = self.dataset_info['start_frame'][i]
            scene_ply = trimesh.load(os.path.join(self.dataroot, scene, 'scene_obj', 'scene_downsampled.ply'))
            # print(scene_ply.vertices.shape)
            scene_points = scene_ply.vertices
            self.scene_list['{}_{}'.format(scene, seq)] = scene_points
        print('Scene load done')
        
def fast_load_obj_vertices(path):
    vertices = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):  # 只读取顶点
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices)

class TrainingDataset(data.Dataset):
    def __init__(self, dataroot, mode='train', imu_path='./preprocess_train_vr_new',
                test_only=False, test_dataset='gimo'):    # BUG preprocess_train_vr_old
        self.dataroot = dataroot
        self.gimo_dataroot = os.path.join(self.dataroot, 'GIMO') # for GIMO dataset
        self.egobody_dataroot = os.path.join(self.dataroot, 'Egobody_dataset') # for Egobody dataset
        self.mode = mode
        self.training = True if self.mode == 'train' else False
        self.test_only = test_only
        if not self.test_only:
            self.imu_path = os.path.join(imu_path, f'{mode}_vr.pkl')
        else:
            self.imu_path = os.path.join(imu_path, f'test_{test_dataset}_vr.pkl')


        # NOTE: Hard coded
        self.input_seq_len = 40
        self.output_seq_len = 40
        self.fps = 30
        self.scene_downsample_points = 60000
        self.cxt_num_points = 1024          # NOTE ablation study required
        self.sigma = 0.1
        self.img_size = 224
        self.radius = 1.5                   # NOTE

        self.dataset_info_gimo = pd.read_csv(os.path.join(self.gimo_dataroot, 'dataset.csv'))
        self.dataset_info_egobody = pd.read_csv(os.path.join(self.egobody_dataroot, 'data_info_release.csv'))
        self.dataset_splitinfo_egobody = pd.read_csv(os.path.join(self.egobody_dataroot, 'data_split.csv'))
        self.transform_scale_dict = {}  # 新增：缓存transform_
        self.transform_norm_dict = {}  # 新增：缓存transform_norm
        self.preprocessed_scene_dict = {}

        
        # self.parse_data_info()
        self.load_imu()
        self.load_scene()
        self._precompute_sampled_scene_points()

        self.random_ori_list = [-180, -90, 0, 90]
        self.transform = tv_transforms.Compose([
            tv_transforms.Resize(self.img_size),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])
        self.load_data_dict()
        
        global_p = self.imu_data['global_p']
        self.x_mean = np.mean(global_p.reshape([global_p.shape[0], global_p.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)
        self.x_std = np.std(global_p.reshape([global_p.shape[0], global_p.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)
        
        # normalize 
        self.mean = np.mean(self.imu_data['input_seq'], axis=(0,1))
        self.std = np.std(self.imu_data['input_seq'], axis=(0,1))

    def downsample_point_cloud(self, points: np.ndarray, target_num: int) -> np.ndarray:
        num_points = len(points)

        if num_points == 0:
            # 全部点为空，直接返回 zeros
            return np.zeros((target_num, 3), dtype=np.float32)

        if num_points >= target_num:
            # 随机下采样
            indices = np.random.choice(num_points, target_num, replace=False)
            return points[indices]
        else:
            # BUG 不足，重复补齐
            # print(f"Points {num_points} less than {target_num}")
            repeat_count = target_num // num_points + 1
            padded = np.tile(points, (repeat_count, 1))
            return padded[:target_num]

    def extract_points_in_radius(self, point_cloud, centers, radius, max_points_per_center=None):
        assert centers.ndim == 2 and centers.shape[1] == 3, "中心点序列必须是 (N, 3) 形状"
        
        if max_points_per_center is None:
            max_points_per_center = min(5000, len(point_cloud) // max(1, len(centers)))
        
        tree = BallTree(point_cloud, leaf_size=15, metric='euclidean')
        
        all_cropped = []
        
        for center in centers[::20]:                # NOTE define scene points sample length to 10
            indices = tree.query_radius([center], r=radius)[0]
            
            if len(indices) == 0:
                cropped = np.tile(center, (max_points_per_center, 1))  # 创建虚拟点
            else:
                cropped = point_cloud[indices]
                
                if len(cropped) > max_points_per_center:
                    step = len(cropped) / max_points_per_center
                    indices = (np.arange(max_points_per_center) * step).astype(int)
                    cropped = cropped[indices]
                
                elif len(cropped) < max_points_per_center:
                    repeat_count = max_points_per_center // len(cropped) + 1
                    cropped = np.tile(cropped, (repeat_count, 1))
                    cropped = cropped[:max_points_per_center]
            
            all_cropped.append(cropped)
        
        concatenated = np.vstack(all_cropped)
        _, unique_idx = np.unique(concatenated, axis=0, return_index=True)
        concatenated = concatenated[unique_idx]

        final_points = self.downsample_point_cloud(concatenated, self.cxt_num_points)
        
        return final_points

    def extract_points_in_bbox(self, point_cloud, centers, radius):
        assert centers.ndim == 2 and centers.shape[1] == 3, "中心点序列必须是 (N, 3) 形状"
        all_cropped = []
        for center in centers[::20]:
            lower = center - radius
            upper = center + radius
            mask = np.all((point_cloud >= lower) & (point_cloud <= upper), axis=1)
            cropped = point_cloud[mask]
            if len(cropped) > 0:
                all_cropped.append(cropped)
        if len(all_cropped) == 0:
            # 没有采到点，直接用中心点补齐
            concatenated = np.tile(centers[0], (self.cxt_num_points, 1))
        else:
            concatenated = np.vstack(all_cropped)
            # 去重
            concatenated = np.unique(concatenated, axis=0)
        # 下采样
        final_points = self.downsample_point_cloud(concatenated, self.cxt_num_points)
        return final_points

    def __getitem__(self, index):
        input_seq = torch.from_numpy(self.imu_data['input_seq'][index]).float()
        mid_seq = torch.from_numpy(self.imu_data['mid_seq'][index]).float()
        tgt_seq = torch.from_numpy(self.imu_data['tgt_seq'][index]).float()
        global_p = torch.from_numpy(self.imu_data['global_p'][index]).float()
        contact_label = torch.from_numpy(self.imu_data['contact_label'][index]).float()
        local_rot = torch.from_numpy(self.imu_data['local_rot'][index]).float()
        head_start = self.imu_data['head_start'][index]
        head_start_invert = self.imu_data['head_start_invert'][index]
        info = self.imu_data['info'][index]

        # root_pose = self.imu_data['global_p'][index][:, 0]

        # root_pose_invert = (head_start @ xyz_to_transform_matrices(root_pose))[0, :, :3, 3]

        start_frame, end_frame = int(info['start_end'][0]),  int(info['start_end'][1])

        imgs = []

        # if info['dataset'] == 'gimo': # For GIMO dataset
        #     scene = info['scene']
        #     seq = info['seq']
        #     transform_path = info['transform']
        #     imgs = []
            # transform_info_file = os.path.join(self.gimo_dataroot, seq, transform_path)
            # scale = self.transform_scale_dict[transform_info_file] 
            # transform_norm_file = os.path.join(self.gimo_dataroot, scene, 'scene_obj', 'transform_norm.txt')
            # transform_norm = self.transform_norm_dict[transform_norm_file]
            # transform_norm[:3, 3] /= scale
            # scene_points = self.scene_list[seq.replace('/', '_')] # prepare scene pointcloud
            # scene_points *= 1 / scale
            # scene_points = (transform_norm[:3, :3] @ scene_points.T + transform_norm[:3, 3:]).T
            # sampled_scene_points = self.extract_points_in_bbox(scene_points, root_pose_invert, radius=self.radius)
            # sampled_scene_points = (head_start_invert @ xyz_to_transform_matrices(sampled_scene_points))[0, :, :3, 3]
        # elif info['dataset'] == 'egobody': # For Egobody dataset
        #     scene = info['scene']
        #     seq = info['seq']
        #     imgs = []
            # scene_points = self.scene_list['{}_{}'.format(scene, seq)] # prepare scene pointcloud            
            # sampled_scene_points = self.extract_points_in_bbox(scene_points, root_pose_invert, radius=self.radius)
            # sampled_scene_points = (head_start_invert @ xyz_to_transform_matrices(sampled_scene_points))[0, :, :3, 3]
        

        sampled_scene_points = torch.from_numpy(self.sampled_scene_points_list[index].astype(np.float32))


        input_ = {}
        
        input_['input_seq'] = input_seq.float()
        input_['mid_seq'] = mid_seq.float()
        input_['tgt_seq'] = tgt_seq.float()
        input_['global_p'] = global_p.float()
        input_['root'] = global_p[..., 0, :].float()
        input_['contact_label'] = contact_label.float()
        input_['local_rot'] = local_rot.float()
        input_['head_start'] = torch.from_numpy(head_start).float()
        # Scene Points
        input_['scene_points'] = sampled_scene_points
        # input_['scene_points'] = torch.zeros((1024, 3)).float()
        # Input Images
        input_['imgs'] = imgs
        input_['dataset_name'] = info['dataset']

        if self.test_only:
            input_['total_length'] = self.imu_data['total_length'][index]
            input_['info'] = info
            input_['head_start'] = self.imu_data['head_start'][index]
  
        return input_

    def __len__(self):
        return self.imu_data['input_seq'].shape[0]

    def parse_data_info(self):
        self.sequences_path_list = []
        self.scenes_path_list = []
        self.trans_path_list = []
        self.poses_path_list = []
        self.start_end_list = []
        for i, seq in enumerate(self.dataset_info_gimo['sequence_path']): # for GIMO dataset
            if self.dataset_info_gimo['training'][i] != self.training:
                continue
            start_frame = self.dataset_info_gimo['start_frame'][i]
            scene = self.dataset_info_gimo['scene'][i]
            transform = self.dataset_info_gimo['transformation'][i]
            
            self.poses_path_list.append(start_frame)
            self.sequences_path_list.append(seq)
            self.scenes_path_list.append(scene)
            self.trans_path_list.append(transform)
            self.start_end_list.append([self.dataset_info_gimo['start_frame'][i], self.dataset_info_gimo['end_frame'][i]])
            
        for i, seq in enumerate(self.dataset_info_egobody['recording_name']): # for egobody dataset
            if (self.dataset_splitinfo_egobody[self.mode] == seq).any() != self.training: 
                continue
            start_frame = self.dataset_info_egobody['start_frame'][i]
            scene = self.dataset_info_egobody['scene_name'][i]
            # no transform info in Egobody dataset
            self.poses_path_list.append(start_frame)
            self.sequences_path_list.append(seq)
            self.scenes_path_list.append(scene)
            self.start_end_list.append([self.dataset_info_egobody['start_frame'][i], self.dataset_info_egobody['end_frame'][i]])
        
    def load_imu(self):
        with open(self.imu_path, 'rb') as f:
            self.imu_data = pickle.load(f)
        # if 'scene_points' not in self.imu_data:
        #     self.imu_data['scene_points'] = []
        print('IMU information load done')
                 
    def load_data_dict(self):
        data_sample = self.__getitem__(0)
        seq_len, input_seq_dim = data_sample['input_seq'].shape
        assert seq_len == motion_constants.preprocess_window, "seq length should be same as window size in preprocessing! check preprocess.py"

        mid_seq_dim = data_sample['mid_seq'].shape[1]
        output_seq_dim = data_sample['tgt_seq'].shape[1]

        self.dim_dict = {}
        self.dim_dict['input_dim'] = input_seq_dim
        self.dim_dict['mid_dim'] = mid_seq_dim
        self.dim_dict['output_dim'] = output_seq_dim
        
    def get_data_dict(self):
        return self.dim_dict	
                
    def load_scene(self):
        self.scene_list = {}
        for i, seq in enumerate(tqdm(self.dataset_info_gimo['sequence_path'], desc="Loading GIMO scene")):  # for GIMO
            scene = self.dataset_info_gimo['scene'][i]
            scene_key = f"{scene}_{seq}"
            scene_dir = os.path.join(self.gimo_dataroot, scene, 'scene_obj')
            full_ply_path = os.path.join(scene_dir, 'scene_downsampled.ply')  # 原始场景点云
            downsample_ply_path = os.path.join(scene_dir, f'scene_downsampled_{self.scene_downsample_points}.ply')

            if os.path.exists(downsample_ply_path):
                # 如果已经存在 downsampled 文件，直接读取
                pcd = o3d.io.read_point_cloud(downsample_ply_path)
                scene_points = np.asarray(pcd.points, dtype=np.float32)
            else:
                # 否则读取原始场景点云并进行 downsample
                pcd = o3d.io.read_point_cloud(full_ply_path)
                scene_points = np.asarray(pcd.points, dtype=np.float32)

                if len(scene_points) >= self.scene_downsample_points:
                    indices = np.random.choice(len(scene_points), self.scene_downsample_points, replace=False)
                else:
                    pad = self.scene_downsample_points - len(scene_points)
                    pad_points = np.tile(scene_points[-1:], (pad, 1))
                    scene_points = np.concatenate([scene_points, pad_points], axis=0)
                    indices = np.arange(self.scene_downsample_points)

                scene_points = scene_points[indices]

                # 保存 downsample 后的点云
                downsampled_pcd = o3d.geometry.PointCloud()
                downsampled_pcd.points = o3d.utility.Vector3dVector(scene_points)
                o3d.io.write_point_cloud(downsample_ply_path, downsampled_pcd)

            self.scene_list[scene_key] = scene_points

        
        for i, seq in enumerate(tqdm(self.dataset_info_egobody['recording_name'], desc="Loading EgoBody scene")):
            scene = self.dataset_info_egobody['scene_name'][i]
            scene_key = f"{scene}_{seq}"
            scene_dir = os.path.join(self.egobody_dataroot, 'scene_mesh', scene)
            ply_path = os.path.join(scene_dir, f'{scene}.obj')
            
            # Downsampled point cloud save path
            downsample_ply_path = os.path.join(scene_dir, f'{scene}_downsampled_{self.scene_downsample_points}.ply')

            if os.path.exists(downsample_ply_path):
                # Load downsampled point cloud from .ply
                downsampled_mesh = o3d.io.read_point_cloud(downsample_ply_path)
                scene_points = np.asarray(downsampled_mesh.points, dtype=np.float32)
            else:
                # Read full mesh and get vertices
                mesh = o3d.io.read_triangle_mesh(ply_path)
                scene_points = np.asarray(mesh.vertices, dtype=np.float32)

                if len(scene_points) >= self.scene_downsample_points:
                    indices = np.random.choice(len(scene_points), self.scene_downsample_points, replace=False)
                else:
                    pad = self.scene_downsample_points - len(scene_points)
                    pad_points = np.tile(scene_points[-1:], (pad, 1))
                    scene_points = np.concatenate([scene_points, pad_points], axis=0)
                    indices = np.arange(self.scene_downsample_points)

                scene_points = scene_points[indices]

                # Save as point cloud .ply
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(scene_points)
                o3d.io.write_point_cloud(downsample_ply_path, pcd)

            self.scene_list[scene_key] = scene_points
        print('Scene load done')

        for i, info in enumerate(tqdm(self.imu_data['info'], desc="Loading transformation")):
            if info['dataset'] == 'gimo':
                seq = info['seq']
                scene = info['scene']
                transform_path = info['transform']

                transform_info_file = os.path.join(self.gimo_dataroot, seq, transform_path)
                if transform_info_file not in self.transform_scale_dict:
                    transform_info = json.load(open(transform_info_file, 'r'))
                    scale = transform_info['scale']
                    self.transform_scale_dict[transform_info_file] = scale
                transform_norm_file = os.path.join(self.gimo_dataroot, scene, 'scene_obj', 'transform_norm.txt')
                if transform_norm_file not in self.transform_norm_dict:
                    transform_norm = np.loadtxt(transform_norm_file).reshape((4, 4)).astype(np.float32)
                    self.transform_norm_dict[transform_norm_file] = transform_norm
        
        if 'head_start_invert' not in self.imu_data:
            print("head_start_invert 不存在，正在计算...")
            head_start = self.imu_data['head_start']  # shape: [N, ...]
            head_start_invert = []
            for i in range(len(head_start)):
                # 这里假设 head_start[i] 是 4x4 或 3x4 的变换矩阵
                head_start_invert.append(invert_T(head_start[i]))
            self.imu_data['head_start_invert'] = np.array(head_start_invert)
            print("head_start_invert 计算完成！")
        else:
            print("head_start_invert 已存在，无需计算。")


        # for i, info in enumerate(tqdm(self.imu_data['info'], desc="Loading transformation")):
        #     head_start_invert = self.imu_data['head_start_invert'][i]
        #     seq = info['seq']
        #     scene = info['scene']
        #     transform_path = info['transform']
        #     head_start = self.imu_data['head_start'][i]
        #     root_pose = self.imu_data['global_p'][i][:, 0]
        #     root_pose_invert = (head_start @ xyz_to_transform_matrices(root_pose))[0, :, :3, 3]
        #     if info['dataset'] == 'gimo':
        #         transform_info_file = os.path.join(self.gimo_dataroot, seq, transform_path)
        #         scale = self.transform_scale_dict[transform_info_file] 
        #         transform_norm_file = os.path.join(self.gimo_dataroot, scene, 'scene_obj', 'transform_norm.txt')
        #         transform_norm = self.transform_norm_dict[transform_norm_file]
        #         transform_norm[:3, 3] /= scale
                
        #         scene_points = self.scene_list[seq.replace('/', '_')] # prepare scene pointcloud
        #         scene_points *= 1 / scale
        #         scene_points = (transform_norm[:3, :3] @ scene_points.T + transform_norm[:3, 3:]).T

        #         sampled_scene_points = self.extract_points_in_bbox(scene_points, root_pose_invert, radius=self.radius)
                
        #         sampled_scene_points = (head_start_invert @ xyz_to_transform_matrices(sampled_scene_points))[0, :, :3, 3]

        #     elif info['dataset'] == 'egobody':
        #         scene_points = self.scene_list['{}_{}'.format(scene, seq)] # prepare scene pointcloud
        #         sampled_scene_points = self.extract_points_in_bbox(scene_points, root_pose_invert, radius=self.radius) # Extract points in a radius of 1.0m
                        
        #         sampled_scene_points = (head_start_invert @ xyz_to_transform_matrices(sampled_scene_points))[0, :, :3, 3]
        #     else:
        #         raise NotImplementedError
            
        #     self.imu_data['scene_points'].append(sampled_scene_points)


    def _precompute_sampled_scene_points(self, use_disk_cache: bool = True):
        """
        Precompute sampled scene points for every sample to avoid heavy work in __getitem__.
        Each item costs roughly 1024*3*4 bytes ≈ 12KB, so memory is usually fine.
        For huge datasets, enable on-disk cache.
        """
        os.makedirs(os.path.join(self.dataroot, "cache"), exist_ok=True)
        cache_path = os.path.join(
            self.dataroot, "cache",
            f"{self.mode}_sampled_scene_points_{self.cxt_num_points}_{self.radius}.npy"
        )

        if use_disk_cache and os.path.exists(cache_path):
            self.sampled_scene_points_list = np.load(cache_path, mmap_mode=None, allow_pickle=True)
            print(f"[Cache] Loaded precomputed scene points from {cache_path}")
            return

        N = len(self.imu_data['info'])
        out = [None] * N

        print("Precomputing sampled scene points...")
        for i, info in enumerate(tqdm(self.imu_data['info'], desc="Precompute scene points")):
            dataset = info['dataset']
            # Per-sample transforms
            head_start = self.imu_data['head_start'][i]
            head_start_invert = self.imu_data['head_start_invert'][i]
            root_pose = self.imu_data['global_p'][i][:, 0]  # (T, 3)

            # Convert root world points into scene frame used for bbox/radius crop
            # (same as your __getitem__):
            root_pose_invert = (head_start @ xyz_to_transform_matrices(root_pose))[0, :, :3, 3]

            if dataset == 'gimo':
                seq = info['seq']
                scene = info['scene']
                transform_path = info['transform']

                # Load transforms from dict caches (already prepared in load_scene)
                transform_info_file = os.path.join(self.gimo_dataroot, seq, transform_path)
                scale = self.transform_scale_dict[transform_info_file]
                transform_norm_file = os.path.join(self.gimo_dataroot, scene, 'scene_obj', 'transform_norm.txt')
                transform_norm = self.transform_norm_dict[transform_norm_file].copy()
                transform_norm[:3, 3] /= scale

                scene_key = seq.replace('/', '_')
                scene_points = self.scene_list[scene_key].astype(np.float32)  # base scene pcd
                scene_points *= 1.0 / scale
                scene_points = (transform_norm[:3, :3] @ scene_points.T + transform_norm[:3, 3:]).T

                # Crop -> then map back with head_start_invert (same as your __getitem__)
                cropped = self.extract_points_in_bbox(scene_points, root_pose_invert, radius=self.radius)
                
                cropped = (head_start_invert @ xyz_to_transform_matrices(cropped))[0, :, :3, 3]
                out[i] = cropped.astype(np.float32)

            elif dataset == 'egobody':
                scene = info['scene']
                seq = info['seq']
                scene_key = f"{scene}_{seq}"
                scene_points = self.scene_list[scene_key].astype(np.float32)

                cropped = self.extract_points_in_bbox(scene_points, root_pose_invert, radius=self.radius)

                cropped = (head_start_invert @ xyz_to_transform_matrices(cropped))[0, :, :3, 3]
                out[i] = cropped.astype(np.float32)
            else:
                # Fallback: zero points (shouldn't happen)
                out[i] = np.zeros((self.cxt_num_points, 3), dtype=np.float32)

        self.sampled_scene_points_list = np.stack(out).astype(np.float32)
        print("Precompute done.")

        if use_disk_cache:
            np.save(cache_path, self.sampled_scene_points_list, allow_pickle=True)
            print(f"[Cache] Saved precomputed scene points to {cache_path}")


def vis_points(train_dataset, i):
    data_sample = train_dataset[i]
    print(data_sample['dataset_name'])
    #input_xyz = data_sample['input_seq'][..., 10:10+3].reshape(-1, 3)
    input_xyz = data_sample['global_p'].reshape(-1, 3)
    #trimesh.Trimesh(input_xyz).export('/home/zhanggangjian/nanjie/head.obj')
    input_xyz = np.array(input_xyz)  # Ensure it's an ndarray
    print(input_xyz.shape)
    scene_points = np.array(data_sample['scene_points'])

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(input_xyz)
    pcd1.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (input_xyz.shape[0], 1)))  # 红色 (RGB: 1, 0, 0)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(scene_points)
    pcd2.colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (scene_points.shape[0], 1)))  # 绿色 (RGB: 0, 1, 0)

    # 合并两个点云
    combined_pcd = pcd1 + pcd2

    # 导出为带颜色的 .ply 文件
    o3d.io.write_point_cloud(f'./vis/combine_pcd_{i}.ply', combined_pcd)

if __name__=="__main__":
    data_root = '../MocapEvery_Files_Download'
    mode = 'train'
    train_dataset = TrainingDataset(data_root, mode) # ['train', 'test', 'val']
    i = -200

    import ipdb; ipdb.set_trace()
    vis_points(train_dataset, i)

    
