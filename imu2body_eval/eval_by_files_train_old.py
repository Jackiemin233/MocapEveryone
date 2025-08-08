from copy import deepcopy
from matplotlib.pyplot import axes
import torch
import sys, os
# sys.path.insert(0, os.path.dirname(__file__))
# sys.path.append("..")
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

import argparse
import logging
import numpy as np
import os
import pickle

from IPython import embed
from pytorch3d import transforms
# from bvh import *
from fairmotion.core import motion as motion_classes
from fairmotion.ops import conversions, math as fairmotion_math
from fairmotion.data import bvh
import sys
from datetime import datetime
from copy import deepcopy
from imu2body_eval.functions import *
# from visualizer import bvh_single_visualizer
import imu2body.egobody as egobody
import imu2body.gimo as gimo
import imu2body_eval.amass_smplh as amass_smplh
from fairmotion.utils import utils
from tqdm import tqdm
from copy import deepcopy
import constants.imu as imu_constants
import constants.motion_data as motion_constants
import imu2body_eval.imu as imu
# from interaction.contact import *
from IPython import embed 
from tqdm import tqdm
# for totalcapture data
from datasets.tc_data import * 
from datasets.hps_data import *
import pandas as pd 

logging.basicConfig(
	format="[%(asctime)s] %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
	level=logging.INFO,
)

bm_path = "../data/smpl_models/smplh/male/model.npz"
CUR_BM_TYPE = "smplx"

def load_data_from_gimo_eval(base_dir, file_list, idx, start_end=None):
	assert isinstance(file_list, list), "Always a list of filenames should be given. If custom, should be given as [filename] format."	
	assert len(file_list) > 0, "There should be more than one file in the file list"

	filepath_list = [os.path.join(base_dir, file) for file in file_list]
	num_cpus = min(24, len(file_list))

	pkl_files = [f for f in filepath_list if f.endswith('.pkl')]

	# read skel and files	
	body_model = amass_smplh.load_body_model(bm_path=motion_constants.SMPLH_BM_PATH)
	skel_with_offset = amass_smplh.create_skeleton_from_amass_bodymodel(bm=body_model)	
	skel = skel_with_offset[0]
	motion_list = [gimo.create_motion_from_gimo_data(pkl_files, bm=body_model, skel_with_offset=deepcopy(skel_with_offset))]

	logging.info(f"Done converting GIMO into fairmotion Motion class")
 
	# start end tuple 
	start_frame, end_frame = start_end

	# read list
	local_T = [] 
	global_T = []

	# imu signal list
	imu_rot = []
	imu_acc = []

	# contact labels
	c_lr = []
 
	# start/end list
	start_end_list = []
	
	ee_joint_names = imu_constants.imu_joint_names + motion_constants.FOOT_JOINTS
	ee_joint_idx = [skel.get_index_joint(jn) for jn in ee_joint_names]

	imu_joint_names = imu_constants.imu_joint_names
	imu_joint_idx = [skel.get_index_joint(jn) for jn in imu_joint_names]

	# constants
	window = motion_constants.preprocess_window
	offset = motion_constants.preprocess_window
	height_indice = 1 if motion_constants.UP_AXIS == "y" else 2

	for motion in tqdm(motion_list):
		if motion is None or motion.num_frames() < window:
			continue
		motion_local_T = motion.to_matrix()
		motion_global_T = motion.to_matrix(local=False)
		motion_imu_rot, motion_imu_acc = imu.imu_from_global_T(motion_global_T, imu_joint_idx)

		# set contact/height offset 
		height_offset = 0.0
		contact_frame = 0
		contact = {}
		contact[contact_frame] = height_offset 

		# split into sliding windows
		i = start_frame
		while True:
			if i+window > end_frame:
				break
			else:
				local_T_window = motion_local_T[i: i+window]
				global_T_window = motion_global_T[i: i+window]
				imu_rot_window = motion_imu_rot[i: i+window]
				imu_acc_window = motion_imu_acc[i: i+window]

			# no height adjust in eval
			local_T_window_height_adjust = deepcopy(local_T_window)
			global_T_window_height_adjust = deepcopy(global_T_window)

			# record
			local_T.append(local_T_window_height_adjust)
			global_T.append(global_T_window_height_adjust)
			imu_rot.append(imu_rot_window)
			imu_acc.append(imu_acc_window)
			start_end_list.append(np.array([i, i+window]))

			i += offset

	local_T = np.asarray(local_T).astype(dtype=np.float32) # [# n of window, window size, J, 4, 4]
	global_T = np.asarray(global_T).astype(dtype=np.float32)
	imu_rot = np.asarray(imu_rot).astype(dtype=np.float32) 
	imu_acc = np.asarray(imu_acc).astype(dtype=np.float32)
	start_end = np.asarray(start_end_list).astype(dtype=np.float32)

	head_idx = skel.get_index_joint("Head")

	upvec_axis = np.array([0,0,0]).astype(dtype=np.float32)
	upvec_axis[1] = 1.0 # upvec is y even in amass

	# y = np.array([0,1,0]).astype(dtype=np.float32)
	head_upvec = np.einsum('ijkl,l->ijk', global_T[..., head_idx,:3,:3], upvec_axis) # fixed bug! 
	head_height = global_T[...,head_idx,height_indice,3][..., np.newaxis]

	# by head 
	head_start_T = global_T[:,0:1,head_idx:head_idx+1,...] # [# window, 1, 1, 4, 4]
	batch, seq_len, num_joints, _, _ = local_T.shape
	head_invert = invert_T(head_start_T)
	local_T[...,0:1,:,:] = head_invert @ local_T[...,0:1,:,:] # only adjust root

	# loop to save ram space..
	normalized_global_T = np.zeros(shape=global_T.shape)
	for i in range(seq_len):
		g_t = head_invert @ global_T[:,i:i+1,...]
		normalized_global_T[:,i:i+1,...] = g_t

	del global_T

	# imu & head input
	head_invert_rot = head_invert[...,:3,:3] 
	normalized_imu_rot = head_invert_rot @ imu_rot  # [Window #, seq, 2, 3, 3]
	normalized_imu_acc = np.einsum('ijklm,ijkm->ijkl', head_invert_rot, imu_acc) # [Window #, seq, 2, 3]
	normalized_imu_concat = T_to_6d_and_pos(conversions.Rp2T(normalized_imu_rot, normalized_imu_acc)) # [Window #, seq, 2, 9]
	normalized_imu_concat = normalized_imu_concat.reshape(batch, seq_len, -1)

	normalized_head = T_to_6d_and_pos(normalized_global_T[...,head_idx, :, :])
	head_imu_input = np.concatenate((head_height, head_upvec, normalized_head, normalized_imu_concat), axis=-1) 

	# mid (output of 1st network, input of 2nd network)
	ee_pos = normalized_global_T[...,ee_joint_idx, :3, 3]	
	reshaped_ee_pos = np.transpose(ee_pos, (1, 2, 0, 3))
	ee_pos_v = reshaped_ee_pos.reshape(batch, seq_len, -1)

	local_rotation_6d = T_to_6d_rot(local_T)
	local_rotation_6d = local_rotation_6d.reshape(batch, seq_len, -1)

	output = np.concatenate((normalized_global_T[...,0,:3,3], local_rotation_6d), axis=-1) # [# of windows, seq_len, 6J+3]	
	
	# return global pos for FK loss calc
	global_p = normalized_global_T[...,:3,3]
 
	total, seq_len, _  = output.shape
	result_dict = {}
	result_dict['input_seq'] = torch.Tensor(head_imu_input).float() 
	result_dict['mid_seq'] = torch.Tensor(ee_pos_v).float()
	result_dict['tgt_seq'] = torch.Tensor(output).float() 
	result_dict['global_p'] = torch.Tensor(global_p).float()
	result_dict['root'] = torch.Tensor(global_p[..., 0, :]).float() 
	result_dict['local_rot'] = torch.Tensor(local_T[...,:3,:3]).float()
	result_dict['head_start'] = torch.Tensor(head_start_T) 
	result_dict['start_end'] = start_end
	result_dict['total_length'] = motion_local_T.shape[0]

	# save
	os.makedirs(args.save_path, exist_ok = True)
	with open(os.path.join(os.path.join(args.save_path), f"{idx}_eval.pkl"), "wb") as file:
		pickle.dump(result_dict, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_data_from_amass(base_dir, file_list, save_path, debug=False):
	assert isinstance(file_list, list), "Always a list of filenames should be given. If custom, should be given as [filename] format."	
	assert len(file_list) > 0, "There should be more than one file in the file list"

	filepath_list = [os.path.join(base_dir, file) for file in file_list]
	num_cpus = min(24, len(file_list)) if not debug else 1

	npz_files = [f for f in filepath_list if f.endswith('.npz')]
	bvh_files = [f for f in filepath_list if f.endswith('.bvh')]
	pkl_files = [f for f in filepath_list if f.endswith('.pkl')] # this is for hps data

	# this only works when the list is npz (amass data)
	# read skel
	body_model = amass_smplh.load_body_model(bm_path=motion_constants.SMPLH_BM_PATH)
	skel_with_offset = amass_smplh.create_skeleton_from_amass_bodymodel(bm=body_model)	
	skel = skel_with_offset[0]

	motion_list = utils.run_parallel(amass_smplh.create_motion_from_amass_data, npz_files, num_cpus=num_cpus, bm=body_model, skel_with_offset=deepcopy(skel_with_offset))
	
	if len(bvh_files) > 0:
		motion_list_bvh = utils.run_parallel(amass_smplh.bvh_to_amass_motion, bvh_files, num_cpus=num_cpus, amass_skel=deepcopy(skel))
		motion_list += motion_list_bvh
		npz_files += bvh_files
	
	if len(pkl_files) > 0:
		motion_list = utils.run_parallel(load_motion_and_scene, pkl_files, num_cpus=num_cpus, only_read_motion=True)
		npz_files += pkl_files

	dataset_folder_list = [s.split('/')[3] for s in npz_files]
	logging.info(f"Done converting into fairmotion Motion class")
	
	ee_joint_names = imu_constants.imu_joint_names + motion_constants.FOOT_JOINTS
	ee_joint_idx = [skel.get_index_joint(jn) for jn in ee_joint_names]

	imu_joint_names = imu_constants.imu_joint_names
	imu_joint_idx = [skel.get_index_joint(jn) for jn in imu_joint_names]

	# constants
	window = motion_constants.preprocess_window
	offset = motion_constants.preprocess_window
	height_indice = 1 if motion_constants.UP_AXIS == "y" else 2
 
	for idx, motion in enumerate(tqdm(motion_list)):
		# read list
		local_T = [] 
		global_T = []

		# imu signal list
		imu_rot = []
		imu_acc = []

		# contact labels
		c_lr = []

		if motion is None or motion.num_frames() < window:
			continue
		motion_local_T = motion.to_matrix()
		motion_global_T = motion.to_matrix(local=False)

		# for totalcapture: replace to real imu signals 
		if 'TotalCapture' in npz_files[idx]:
			orig_pose_filename = npz_files[idx].split("/")
			orig_pose_filename = '_'.join(orig_pose_filename[-2:])
			tc_filename = orig_pose_filename.replace("_poses.npz", ".pkl")
			motion_imu_rot, motion_imu_acc = get_imu_from_tc(tc_filename)
			
			# there is 1-2 frame difference in TC amass and TC DIP
			diff = abs(motion.num_frames() - motion_imu_rot.shape[0])
			# print(f"diff: {diff} name: {orig_pose_filename}")
			if diff > 10: #  s5_freestyle3 
				continue
		else:
			motion_imu_rot, motion_imu_acc = imu.imu_from_global_T(motion_global_T, imu_joint_idx)

		# set contact/height offset 
		height_offset = 0.0
		contact_frame = 0
		contact = {}
		contact[contact_frame] = height_offset 

		# split into sliding windows
		start_frame_label = []

		i = 0
		while True:
			if i >= motion_local_T.shape[0]:
				break 
			if i+window >= motion_local_T.shape[0]:
				i = motion_local_T.shape[0] - window
			else:
				local_T_window = motion_local_T[i: i+window]
				global_T_window = motion_global_T[i: i+window]
				imu_rot_window = motion_imu_rot[i: i+window]
				imu_acc_window = motion_imu_acc[i: i+window]

			# no height adjust in eval
			local_T_window_height_adjust = deepcopy(local_T_window)
			global_T_window_height_adjust = deepcopy(global_T_window)

			# record
			local_T.append(local_T_window_height_adjust)
			global_T.append(global_T_window_height_adjust)
			imu_rot.append(imu_rot_window)
			imu_acc.append(imu_acc_window)

			# record start frame idx
			start_frame = i
			start_frame_label.append(start_frame)
			i += offset

		# do it per motion
		local_T = np.asarray(local_T).astype(dtype=np.float32) # [# of window, window size, J, 4, 4]
		global_T = np.asarray(global_T).astype(dtype=np.float32)
		imu_rot = np.asarray(imu_rot).astype(dtype=np.float32) 
		imu_acc = np.asarray(imu_acc).astype(dtype=np.float32)

		# preprocess and add sensor offset
		head_idx = skel.get_index_joint("Head")

		upvec_axis = np.array([0,0,0]).astype(dtype=np.float32)
		upvec_axis[1] = 1.0 # upvec is y even in amass

		head_upvec = np.einsum('ijkl,l->ijk', global_T[..., head_idx,:3,:3], upvec_axis) # fixed bug! 
		head_height = global_T[...,head_idx,height_indice,3][..., np.newaxis]

		# by head 
		head_start_T = global_T[:,0:1,head_idx:head_idx+1,...] # [# window, 1, 1, 4, 4]
		batch, seq_len, num_joints, _, _ = local_T.shape
		head_invert = invert_T(head_start_T)
		local_T[...,0:1,:,:] = head_invert @ local_T[...,0:1,:,:] # only adjust root

		# loop to save ram space..
		normalized_global_T = np.zeros(shape=global_T.shape)
		for i in range(seq_len):
			g_t = head_invert @ global_T[:,i:i+1,...]
			normalized_global_T[:,i:i+1,...] = g_t

		del global_T
  
		# imu & head input
		head_invert_rot = head_invert[...,:3,:3] 
		normalized_imu_rot = head_invert_rot @ imu_rot  # [Window #, seq, 2, 3, 3]
		normalized_imu_acc = np.einsum('ijklm,ijkm->ijkl', head_invert_rot, imu_acc) # [Window #, seq, 2, 3]
		normalized_imu_concat = T_to_6d_and_pos(conversions.Rp2T(normalized_imu_rot, normalized_imu_acc)) # [Window #, seq, 2, 9]
		normalized_imu_concat = normalized_imu_concat.reshape(batch, seq_len, -1)

		normalized_head = T_to_6d_and_pos(normalized_global_T[...,head_idx, :, :])
		head_imu_input = np.concatenate((head_height, head_upvec, normalized_head, normalized_imu_concat), axis=-1) 

		# mid (output of 1st network, input of 2nd network)
		ee_pos = normalized_global_T[...,ee_joint_idx, :3, 3]	
		reshaped_ee_pos = np.transpose(ee_pos, (1, 2, 0, 3))
		ee_pos_v = reshaped_ee_pos.reshape(batch, seq_len, -1)

		if debug:
			return normalized_imu_rot, normalized_imu_acc, ee_pos_v, local_T, normalized_global_T, head_start_T # invert_T(head_invert)

		# change into output sequence form by concatenating root pos 3d and (root-included) joint rotations (6d)
		local_rotation_6d = T_to_6d_rot(local_T)
		local_rotation_6d = local_rotation_6d.reshape(batch, seq_len, -1)

		output = np.concatenate((normalized_global_T[...,0,:3,3], local_rotation_6d), axis=-1) # [# of windows, seq_len, 6J+3]	

		# return global pos for FK loss calc
		global_p = normalized_global_T[...,:3,3]

		total, seq_len, _  = output.shape
		result_dict = {}
		result_dict['input_seq'] = torch.Tensor(head_imu_input).float() 
		result_dict['mid_seq'] = torch.Tensor(ee_pos_v).float()
		result_dict['tgt_seq'] = torch.Tensor(output).float() 
		result_dict['global_p'] = torch.Tensor(global_p).float()
		result_dict['root'] = torch.Tensor(global_p[..., 0, :]).float() 
		result_dict['local_rot'] = torch.Tensor(local_T[...,:3,:3]).float()
		result_dict['head_start'] = torch.Tensor(head_start_T) 
		result_dict['contact_label'] = torch.Tensor(c_lr).float()
		result_dict['start_frame'] = start_frame_label
		result_dict['total_length'] = motion_local_T.shape[0]

		filename = npz_files[idx]
		result_dict['filename'] = filename 

		# save
		dataset_folder = dataset_folder_list[idx]
		save_path_per_file = os.path.join(save_path, dataset_folder)
		utils.create_dir_if_absent(save_path_per_file)

		with open(os.path.join(save_path_per_file, f"{idx}.pkl"), "wb") as file:
			pickle.dump(result_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
   
def load_data_from_training(base_dir, file, setting = 'vr', debug=False, normalization = False):
	motion_list = []
	data_set_info = []
 
	seq = file
	if seq['dataset'] == 'gimo':
		filepath_list = [os.path.join(base_dir, 'GIMO', seq['fname'], 'smplx_local', file) for file in seq['file']]
		transform_info = json.load(open(os.path.join(base_dir, 'GIMO', seq['fname'], seq['transform']), 'r')) # pose to scene transformation
		transform_norm = np.loadtxt(os.path.join(base_dir, 'GIMO', seq['fname'], '../', 'scene_obj', 'transform_norm.txt')).reshape((4, 4))
		start, end = seq['start_end'][0], seq['start_end'][1]
		# scene (pose) normalization transformation
		pkl_files = [f for f in filepath_list if f.endswith('.pkl')][start:end]

	elif seq['dataset'] == 'egobody':
		pkl_files = [os.path.join(base_dir, 'Egobody_dataset', f"smplx_camera_wearer_{seq['mode']}", seq['fname'], seq['body_index'], 'results', file, '000.pkl') for file in seq['file']]
		transform_info = None
		transform_norm = None

	# read skel and files	
	if CUR_BM_TYPE == "smplx":
		if seq['dataset'] == 'gimo':
			body_model = amass_smplh.load_body_model(bm_path=motion_constants.SMPLH_BM_PATH)
			skel_with_offset = gimo.create_skeleton_from_amass_bodymodel(bm=body_model)	
			skel = skel_with_offset[0]
			motion_list.append(gimo.create_motion_from_gimo_data(pkl_files, 
																bm=body_model, 
																transform_info = transform_info, 
																transform_norm = transform_norm, 
																skel_with_offset=deepcopy(skel_with_offset)))
			data_set_info.append(seq)
		elif seq['dataset'] == 'egobody':
			body_model = amass_smplh.load_body_model(bm_path=motion_constants.SMPLH_BM_PATH)
			skel_with_offset = egobody.create_skeleton_from_amass_bodymodel(bm=body_model)	
			skel = skel_with_offset[0]
			motion_list.append(egobody.create_motion_from_egobody_data(pkl_files, 
																bm=body_model, 
																transform_info = transform_info, 
																transform_norm = transform_norm, 
																skel_with_offset=deepcopy(skel_with_offset)))
			data_set_info.append(seq)
	
	else:
		raise NotImplementedError("Only smplx are supported!")

	logging.info(f"Done converting {seq['dataset']} into fairmotion Motion class")

	# read list
	local_T = [] 
	global_T = []

	# imu signal list
	imu_rot = []
	imu_acc = []
 
	# slice_info list
	info_list = []
	
	if setting == 'vr':
		ee_joint_names = imu_constants.imu_joint_names + motion_constants.FOOT_JOINTS
		ee_joint_idx = [skel.get_index_joint(jn) for jn in ee_joint_names]
	else:
		ee_joint_names = motion_constants.FOOT_JOINTS
		ee_joint_idx = [skel.get_index_joint(jn) for jn in ee_joint_names]

	# constants
	window = motion_constants.preprocess_window
	offset = motion_constants.preprocess_window
	height_indice = 1 if motion_constants.UP_AXIS == "y" else 2

	for motion, info in tqdm(zip(motion_list, data_set_info)):
		if motion is None or motion.num_frames() < window:
			continue
		motion_local_T = motion.to_matrix()
		motion_global_T = motion.to_matrix(local=False)
		motion_imu_rot, motion_imu_acc = imu.imu_from_global_T(motion_global_T, imu_joint_idx)

		# set contact/height offset 
		height_offset = 0.0
		contact_frame = 0
		contact = {}
		contact[contact_frame] = height_offset 

		# split into sliding windows
		start_frame, end_frame = 0, info['start_end'][1] - info['start_end'][0]
		i = start_frame
		while True:
			if i >= end_frame:
				break
			if i+window >= motion_local_T.shape[0]:
				i = motion_local_T.shape[0] - window
			else:
				local_T_window = motion_local_T[i: i+window]
				global_T_window = motion_global_T[i: i+window]
				imu_rot_window = motion_imu_rot[i: i+window]
				imu_acc_window = motion_imu_acc[i: i+window]

			# apply height offset: TODO check sign
			local_T_window_height_adjust = deepcopy(local_T_window)
			global_T_window_height_adjust = deepcopy(global_T_window)

			# record
			local_T.append(local_T_window_height_adjust)
			global_T.append(global_T_window_height_adjust)
			imu_rot.append(imu_rot_window)
			imu_acc.append(imu_acc_window)
			info_list.append({
				'start_end': np.array([i, i+window]),
				'seq': info['fname'],
				'scene': info['scene'],
				'dataset': info['dataset'],
				'transform': info['transform']
			})
			i += offset

	local_T = np.asarray(local_T).astype(dtype=np.float32) # [# n of window, window size, J, 4, 4]
	global_T = np.asarray(global_T).astype(dtype=np.float32)
	imu_rot = np.asarray(imu_rot).astype(dtype=np.float32) 
	imu_acc = np.asarray(imu_acc).astype(dtype=np.float32)

	head_idx = skel.get_index_joint("Head")
 
	if info['dataset'] == 'gimo':
		upvec_axis = np.array([0,0,0]).astype(dtype=np.float32)
		upvec_axis[1] = 1.0
	elif info['dataset'] == 'egobody':
		upvec_axis = np.array([0,0,0]).astype(dtype=np.float32)
		upvec_axis[1] = 1.0
	
	head_upvec = np.einsum('ijkl,l->ijk', global_T[..., head_idx,:3,:3], upvec_axis) # fixed bug! 
	head_height = global_T[..., head_idx, height_indice, 3][..., np.newaxis]

	# by head 
	head_start_T = global_T[:,0:1,head_idx:head_idx+1,...] # [# window, 1, 1, 4, 4]
	batch, seq_len, num_joints, _, _ = local_T.shape
	head_invert = invert_T(head_start_T)
	
	if normalization == True:
		#loop to save ram space..
		local_T[...,0:1,:,:] = head_invert @ local_T[...,0:1,:,:] # only adjust root
		normalized_global_T = np.zeros(shape=global_T.shape)
		for i in range(seq_len):
			g_t = head_invert @ global_T[:,i:i+1,...]
			normalized_global_T[:,i:i+1,...] = g_t
		del global_T
	else: 
		normalized_global_T = global_T

	# imu & head input
	if normalization == True:
		head_invert_rot = head_invert[...,:3,:3] 
		normalized_imu_rot = head_invert_rot @ imu_rot  # [Window #, seq, 2, 3, 3]
		normalized_imu_acc = np.einsum('ijklm,ijkm->ijkl', head_invert_rot, imu_acc) # [Window #, seq, 2, 3]
		normalized_imu_concat = T_to_6d_and_pos(conversions.Rp2T(normalized_imu_rot, normalized_imu_acc)) # [Window #, seq, 2, 9]
		normalized_imu_concat = normalized_imu_concat.reshape(batch, seq_len, -1)

	else:
		normalized_imu_rot = imu_rot  # [Window #, seq, 2, 3, 3]
		normalized_imu_acc = imu_acc  # [Window #, seq, 2, 3]
		normalized_imu_concat = T_to_6d_and_pos(conversions.Rp2T(normalized_imu_rot, normalized_imu_acc)) # [Window #, seq, 2, 9]
		normalized_imu_concat = normalized_imu_concat.reshape(batch, seq_len, -1)

	if setting == 'vr':
		normalized_lhand = T_to_6d_and_pos(normalized_global_T[..., skel.get_index_joint("LeftHand"),  :, :])
		normalized_rhand = T_to_6d_and_pos(normalized_global_T[..., skel.get_index_joint("RightHand"), :, :])
		normalized_head = T_to_6d_and_pos(normalized_global_T[..., head_idx, :, :]) # Head position + left and right hand
		head_imu_input = np.concatenate((head_height, head_upvec, normalized_head, normalized_lhand, normalized_rhand), axis=-1) 
	else: 
		normalized_head = T_to_6d_and_pos(normalized_global_T[..., head_idx, :, :]) # Head position 
		head_imu_input = np.concatenate((head_height, head_upvec, normalized_head, normalized_imu_concat), axis=-1) 
 
	# mid (output of 1st network, input of 2nd network)
	ee_pos = normalized_global_T[...,ee_joint_idx, :3, 3]	
	reshaped_ee_pos = np.transpose(ee_pos, (1, 2, 0, 3))
	ee_pos_v = reshaped_ee_pos.reshape(batch, seq_len, -1)

	if debug:
		return normalized_imu_rot, normalized_imu_acc, ee_pos_v, local_T, normalized_global_T, head_start_T 

	local_rotation_6d = T_to_6d_rot(local_T)
	local_rotation_6d = local_rotation_6d.reshape(batch, seq_len, -1)

	output = np.concatenate((normalized_global_T[...,0,:3,3], local_rotation_6d), axis=-1) # [# of windows, seq_len, 6J+3]	
	
	# return global pos for FK loss calc
	global_p = normalized_global_T[...,:3,3]

	total, seq_len, _  = output.shape
	result_dict = {}
	result_dict['input_seq'] = torch.Tensor(head_imu_input).float() 
	result_dict['mid_seq'] = torch.Tensor(ee_pos_v).float()
	result_dict['tgt_seq'] = torch.Tensor(output).float() 
	result_dict['global_p'] = torch.Tensor(global_p).float()
	result_dict['root'] = torch.Tensor(global_p[..., 0, :]).float() 
	result_dict['local_rot'] = torch.Tensor(local_T[...,:3,:3]).float()
	result_dict['head_start'] = torch.Tensor(head_start_T) 
	result_dict['info'] = info_list
	result_dict['total_length'] = motion_local_T.shape[0]

	# save
	with open(os.path.join(os.path.join(args.save_path), f"{seq['dataset']}_test_old", f"{seq['fname'].replace('/', '-')}.pkl"), "wb") as file:
		pickle.dump(result_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
   
def load_filelist(args):
	test_txt_filename = ""
	if args.data_type == "amass_vr":
		test_txt_filename = "amass_vr_fnames.txt"
	if args.data_type == "tc":
		test_txt_filename = "tc_fnames.txt"
	if args.data_type == "hps":
		test_txt_filename = "hps_fnames.txt"
	if args.data_type == "GIMO":
		test_txt_filename = pd.read_csv(os.path.join(args.base_dir, 'dataset.csv'))	
		df = test_txt_filename
	if args.data_type == 'train':
		base_dir = args.base_dir
		gimo_path = os.path.join(base_dir, 'GIMO')
		egobody_path = os.path.join(base_dir, 'Egobody_dataset')
	
		gimo_data_info = pd.read_csv(os.path.join(gimo_path, 'dataset.csv'))	
		egobody_data_info = pd.read_csv(os.path.join(egobody_path, 'data_info_release.csv'))
		egobody_data_split_info = pd.read_csv(os.path.join(egobody_path, 'data_split.csv'))
		
		# Aligned - GIMO
		fnames_list = (gimo_data_info['scene'].astype(str) + '/' + gimo_data_info['sequence_path'].astype(str)).tolist()
		start_end_list = list(zip(gimo_data_info['start_frame'].astype(int), gimo_data_info['end_frame'].astype(int)))
		scene_list = (gimo_data_info['scene'].astype(str)).tolist()
		transform_info = gimo_data_info['transformation']
		training = gimo_data_info['training']
		
		file_lists = []
		for fnames, start_end, transform, scene, training in tqdm(zip(fnames_list, start_end_list, transform_info, scene_list, training)): # GIMO Dataset
			if training == 1:
				continue
			seqlists = {}
			seqlists['fname'] = fnames
			seqlists['start_end'] = start_end
			seqlists['scene'] = scene
			seqlists['transform'] = transform
			seqlists['file'] = [f for f in os.listdir(os.path.join(gimo_path, fnames, 'smplx_local')) if f.endswith('.pkl')]
			seqlists['file'].sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
			seqlists['mode'] = 'test'
			seqlists['dataset'] = 'gimo'
			file_lists.append(seqlists) #NOTE: comment this line to debug Egobody
	
		fnames_list = (egobody_data_info['recording_name'].astype(str)).tolist()
		start_end_list = list(zip(egobody_data_info['start_frame'].astype(int), egobody_data_info['end_frame'].astype(int)))
		scene_list = (egobody_data_info['scene_name'].astype(str)).tolist()
	
		for fnames, start_end, scene in tqdm(zip(fnames_list, start_end_list, scene_list)):
			for col in egobody_data_split_info.columns:
				if bool((egobody_data_split_info[col] == fnames).any()):
					mode = col 
					break
			if mode != 'test': #Only test
				continue
			seqlists = {}
			seqlists['fname'] = fnames
			seqlists['scene'] = scene
			seqlists['start_end_ori'] = start_end
			seqlists['start_end'] = (0, start_end[1]- start_end[0])
			seqlists['transform'] = None
			seqlists['body_index'] = os.listdir(os.path.join(egobody_path, f'smplx_camera_wearer_{mode}', fnames))[0]
			seqlists['file'] = [f for f in os.listdir(os.path.join(egobody_path, f'smplx_camera_wearer_{mode}', fnames, seqlists['body_index'], 'results'))]
			seqlists['file'].sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
			seqlists['mode'] = 'test'
			seqlists['dataset'] = 'egobody'
			file_lists.append(seqlists)

	if args.data_type == "amass_vr":   # TC AMASS HPS
		file = open(os.path.join(args.data_config_path, test_txt_filename), 'r')
		utils.create_dir_if_absent(args.save_path)
		os.system(f'cp {os.path.join(args.data_config_path, test_txt_filename)} {os.path.join(args.save_path, "test_fnames.txt")}')
		filename_list = file.read().split("\n")
		# copy this to args.save_path 
		load_data_from_amass(base_dir=args.base_dir, file_list=filename_list, save_path=args.save_path)
	elif args.data_type == "GIMO":
		fnames_list = (df['scene'].astype(str) + '/' + df['sequence_path'].astype(str)).tolist()
		is_training = (df['training']).tolist()
		start_end_list = list(zip(df['start_frame'].astype(int), df['end_frame'].astype(int)))

		idx = 0
		for fnames, start_end, training in tqdm(zip(fnames_list, start_end_list, is_training)): 
			if training == 1:
				idx += 1
				continue
			bvh_list = [f for f in os.listdir(os.path.join(args.base_dir, fnames, 'smplx_local')) if f.endswith('.pkl')]
			bvh_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
			load_data_from_gimo_eval(base_dir = os.path.join(args.base_dir, fnames, 'smplx_local'), file_list=bvh_list, start_end = start_end, idx = idx) #base_dir, file_list, debug=False, start_end=None
			idx += 1
	elif args.data_type == 'train':
		os.makedirs(os.path.join(args.save_path, 'gimo_test_old'), exist_ok=True)
		os.makedirs(os.path.join(args.save_path, 'egobody_test_old'), exist_ok=True)
		for file in file_lists:
			load_data_from_training(base_dir = args.base_dir, file = file, normalization = True, setting = args.setting)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--base-dir",
		type=str,
		required=True
	)

	parser.add_argument(
		"--data-config-path",
		type=str,
		required=True
	)

	parser.add_argument(
		"--save-path",
		type=str,
		required=True
	)
	parser.add_argument(
		"--data-type",
		default='GIMO',
		type=str,
		required=True
	)
	parser.add_argument(
		"--setting",
		type=str,
		default ='vr',
		choices = ['vr', 'hmc']
	)

	args = parser.parse_args()
	load_filelist(args=args)