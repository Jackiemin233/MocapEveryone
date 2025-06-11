# Copyright (c) Facebook, Inc. and its affiliates.
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

import numpy as np
import pickle
import torch
from fairmotion.utils import constants
from IPython import embed
from imu2body.preprocess import load_data as load_amass_data
from imu2body.preprocess_gimo import load_data as load_gimo_data
# from preprocess_bvh import *
from pytorch3d import transforms
import constants.motion_data as motion_constants 
from torch.utils.data import Dataset, DataLoader
from interaction.contact import *

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
		)

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
		dataset=dataset, batch_size=batch_size, shuffle=training, num_workers=8, drop_last=drop_last
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
		dataset=dataset, batch_size=1, shuffle=False, num_workers=8,  drop_last=False
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
  
import torch.utils.data as data
import torch
from torchvision import transforms
import numpy as np
import random
import os
import json
import pickle
from PIL import Image
from tqdm import tqdm
import pandas as pd
import trimesh
from scipy.spatial.transform import Rotation
import bisect

class GIMODataset(data.Dataset):
    def __init__(self, dataroot, train=False):
        self.dataroot = dataroot
        self.train = train
        
        # NOTE: Hard coded
        self.input_seq_len = 40
        self.output_seq_len = 40
        self.fps = 30
        self.sample_points = 200000
        self.sigma = 0.1

        self.dataset_info = pd.read_csv(os.path.join(self.dataroot, 'dataset.csv'))
        self.parse_data_info()
        self.load_scene()
        self.load_imu()

        self.random_ori_list = [-180, -90, 0, 90]
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])
        self.load_data_dict()
        
        global_p = self.imu_data['global_p']
        x_mean = np.mean(global_p.reshape([global_p.shape[0], global_p.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)
        x_std = np.std(global_p.reshape([global_p.shape[0], global_p.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)
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
        scene, seq, transform_path= self.find_scene_seq(index)
        #======================= imu parameters =======================
        
        img_list = os.listdir(os.path.join(self.dataroot, scene, seq, 'PV'))
        img_list.sort()

        imgs = []
        poses_input = []
        poses_input_idx = []

        random_ori = np.random.choice(self.random_ori_list)  # np.random.uniform(-self.config.random_angle, self.config.random_angle)
        random_rotation = Rotation.from_euler('xyz', [0, random_ori, 0], degrees=True).as_matrix()
        transform_info = json.load(open(os.path.join(self.dataroot, scene, seq, transform_path), 'r'))
        scale = transform_info['scale']
        trans_pose2scene = np.array(transform_info['transformation'])
        trans_pose2scene[:3, 3] /= scale
        transform_norm = np.loadtxt(os.path.join(self.dataroot, scene, 'scene_obj', 'transform_norm.txt')).reshape((4, 4))
        transform_norm[:3, 3] /= scale
        transform_pose = transform_norm @ trans_pose2scene
         
        for f in range(self.input_seq_len):
            pose_idx = start_frame + int(f * 30 / self.fps)
            poses_input_idx.append(pose_idx)
            
            #=============================prepare_image===========================
            img_data = Image.open(os.path.join(self.dataroot, scene, seq, 'PV', img_list[pose_idx])).convert('RGB') # Read input image
            img_data = self.transform(img_data)
            imgs.append(img_data)
			#=============================prepare_image===========================
                        
            pose_data = pickle.load(open(os.path.join(self.dataroot, scene, seq, 'smplx_local',
                                                      '{}.pkl'.format(pose_idx)), 'rb'))
            ori = pose_data['orient'].detach().cpu().numpy()
            trans = pose_data['trans'].detach().cpu().numpy().reshape((3, 1))
            R = Rotation.from_rotvec(ori).as_matrix()

            R_s = transform_pose[:3, :3] @ R
            ori_s = Rotation.from_matrix(R_s).as_rotvec()
            trans_s = (transform_pose[:3, :3] @ trans + transform_pose[:3, 3:]).reshape(3)

            if self.train:
                ori_s = Rotation.from_matrix(random_rotation @ R_s).as_rotvec()
                trans_s = (random_rotation @ trans_s.reshape((3, 1))).reshape(3)

            poses_input.append(
                torch.cat([torch.from_numpy(ori_s.copy()).float(), torch.from_numpy(trans_s.copy()).float(),
                           pose_data['latent']]))

        imgs = torch.stack(imgs, dim=0)
        poses_input = torch.stack(poses_input, dim=0).detach()

		#=============================Scene Point cloud=============================
        scene_points = self.scene_list['{}_{}'.format(scene, seq)]
        scene_points = scene_points[np.random.choice(range(len(scene_points)), self.sample_points)]
        scene_points *= 1 / scale
        scene_points = (transform_norm[:3, :3] @ scene_points.T + transform_norm[:3, 3:]).T
        if self.train:
            scene_points = (random_rotation @ scene_points.T).T
            scene_points += np.random.normal(loc=0, scale=self.sigma, size=scene_points.shape)
        #=============================Scene Point cloud=============================
        
		#=============================IMU Parameters=============================
        input_ = {}
        input_['input_seq'] = input_seq.float()
        input_['mid_seq'] = mid_seq.float()
        input_['tgt_seq'] = tgt_seq.float()
        input_['global_p'] = global_p.float()
        input_['root'] = global_p[..., 0, :].float()
        input_['contact_label'] = contact_label.float()
        #=============================IMU Parameters=============================
    	# VPoser Latent
        input_['poses_input'] = poses_input 
		# SMPL_vertices
        #input_['smplx_vertices'] = smplx_vertices

		# Scene Points
        input_['scene_points'] =  torch.from_numpy(scene_points).float()
        # Input Images
        input_['imgs'] = imgs
        #==============================              
  
        return input_

    def __len__(self):
        return self.imu_data['input_seq'].shape[0]

    def parse_data_info(self):
        self.sequences_path_list = []
        self.scenes_path_list = []
        self.trans_path_list = []
        self.poses_path_list = []
        self.start_end_list = []
        for i, seq in enumerate(self.dataset_info['sequence_path']):
            if self.dataset_info['training'][i] != self.train:
                continue
            start_frame = self.dataset_info['start_frame'][i]
            end_frame = self.dataset_info['end_frame'][i]
            scene = self.dataset_info['scene'][i]
            transform = self.dataset_info['transformation'][i]
            
            self.poses_path_list.append(start_frame)
            self.sequences_path_list.append(seq)
            self.scenes_path_list.append(scene)
            self.trans_path_list.append(transform)
            self.start_end_list.append([self.dataset_info['start_frame'][i], self.dataset_info['end_frame'][i]])
            
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
					'length': seq_info  # 后续计算索引
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
            
if __name__=="__main__":

	data_root = '/hpc2hdd/home/gzhang292/nanjie/project6/orion/group/GIMO'
	training = False
	train_dataset = GIMODataset(data_root, training)
	data_sample = train_dataset[0]

	print('')