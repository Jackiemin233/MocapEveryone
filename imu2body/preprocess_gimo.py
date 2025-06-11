from copy import deepcopy
from matplotlib.pyplot import axes
import torch
import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)

import argparse
import logging
import numpy as np
import os
import pickle

from IPython import embed
from fairmotion.core import motion as motion_classes
from fairmotion.ops import conversions, math as fairmotion_math
from fairmotion.data import bvh
import sys
from datetime import datetime
from copy import deepcopy
from imu2body.functions import *
# import imu2body.amass as amass
import imu2body.gimo as gimo
import imu2body_eval.amass_smplh as amass_smplh
from fairmotion.utils import utils
from tqdm import tqdm
from copy import deepcopy
import constants.imu as imu_constants
import constants.motion_data as motion_constants
import imu2body.imu as imu
from interaction.contact import *
import pandas as pd

logging.basicConfig(
	format="[%(asctime)s] %(message)s",
	datefmt="%Y-%m-%d %H:%M:%S",
	level=logging.INFO,
)

smplx_bm_path = "../data/smpl_models/smplx/SMPLX_NEUTRAL.npz"
smplh_bm_path = "../data/smpl_models/smplh/male/model.npz"

CUR_BM_TYPE = "smplx"

def load_data_from_gimo(base_dir, file_list, debug=False, start_end=None):
	assert isinstance(file_list, list), "Always a list of filenames should be given. If custom, should be given as [filename] format."	
	assert len(file_list) > 0, "There should be more than one file in the file list"

	filepath_list = [os.path.join(base_dir, file) for file in file_list]
	num_cpus = min(24, len(file_list)) if not debug else 1

	pkl_files = [f for f in filepath_list if f.endswith('.pkl')]

	# read skel and files	
	if CUR_BM_TYPE == "smplx":
		bm_path = smplx_bm_path
		body_model = gimo.load_body_model(bm_path=bm_path)
		skel_with_offset = gimo.create_skeleton_from_amass_bodymodel(bm=body_model)	
		skel = skel_with_offset[0]
		motion_list = [gimo.create_motion_from_gimo_data(pkl_files, bm=body_model, skel_with_offset=deepcopy(skel_with_offset))]
		#motion_list = utils.run_parallel(gimo.create_motion_from_gimo_data, pkl_files, num_cpus=num_cpus, bm=body_model, skel_with_offset=deepcopy(skel_with_offset))

	else:
		raise NotImplementedError("Only smplx are supported!")

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
	offset = motion_constants.preprocess_offset
	height_indice = 1 if motion_constants.UP_AXIS == "y" else 2
 
	is_custom_run = False

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

			# apply height offset: TODO check sign
			local_T_window_height_adjust = deepcopy(local_T_window)
			global_T_window_height_adjust = deepcopy(global_T_window)

			if not is_custom_run:
				_, cur_height_offset = get_height_offset_current_frame(contact_dict=contact, cur_frame=i)
				if abs(cur_height_offset) > 0:
					# print(f"motion idx:{len(global_T)} i: {i} contact key: {_} height offset: {cur_height_offset}")
					local_T_window_height_adjust[:,0,height_indice,3] -= cur_height_offset
					global_T_window_height_adjust[...,height_indice,3] -= cur_height_offset

			# record
			local_T.append(local_T_window_height_adjust)
			global_T.append(global_T_window_height_adjust)
			imu_rot.append(imu_rot_window)
			imu_acc.append(imu_acc_window)
			start_end_list.append(np.array([i, i+window]))

			i += offset

			if not is_custom_run:
				# update floor for next window
				result_dict = update_height_offset(global_T=global_T_window, prev_offset=height_offset, frame_start=i, return_contact_labels=True)

				updated_height_offset = result_dict['height']
				updated_contact_frame = result_dict['frame']
				contact_labels  = result_dict['contact_label']

				if updated_contact_frame > contact_frame:
					contact_frame = updated_contact_frame
					height_offset = updated_height_offset
					contact[contact_frame] = height_offset

				c_lr.append(contact_labels)

	local_T = np.asarray(local_T).astype(dtype=np.float32) # [# n of window, window size, J, 4, 4]
	global_T = np.asarray(global_T).astype(dtype=np.float32)
	imu_rot = np.asarray(imu_rot).astype(dtype=np.float32) 
	imu_acc = np.asarray(imu_acc).astype(dtype=np.float32)
	start_end = np.asarray(start_end_list).astype(dtype=np.float32)

	c_lr = np.asarray(c_lr).astype(dtype=np.float32)
	c_lr = c_lr.transpose(0,2,1)

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

	if debug:
		return normalized_imu_rot, normalized_imu_acc, ee_pos_v, local_T, normalized_global_T, head_start_T 

	local_rotation_6d = T_to_6d_rot(local_T)
	local_rotation_6d = local_rotation_6d.reshape(batch, seq_len, -1)

	output = np.concatenate((normalized_global_T[...,0,:3,3], local_rotation_6d), axis=-1) # [# of windows, seq_len, 6J+3]	
	
	# return global pos for FK loss calc
	global_p = normalized_global_T[...,:3,3]

	return head_imu_input, ee_pos_v, output, global_p, local_T[...,:3,:3], head_start_T, c_lr, start_end


def load_data_with_args(fnames, bvh_list, start_end ,args):
	logging.info(f"Start processing {fnames} data with {len(bvh_list)} files...")
	data, total_len = load_data(bvh_list, base_dir=os.path.join(args.base_dir, fnames, 'smplx_local'), start_end = start_end)
	logging.info(f"Processed {fnames} data with {total_len} sequences")
	os.makedirs(os.path.join(args.base_dir, fnames, 'IMU'), exist_ok=True)
	with open(os.path.join(args.base_dir, fnames, 'IMU', f"imu_{start_end[0]}_{start_end[1]}.pkl"), "wb") as f_write:
		pickle.dump(data, f_write, protocol=pickle.HIGHEST_PROTOCOL)
	logging.info(f"Saved {fnames} data with {total_len} sequences in {os.path.join(args.base_dir, fnames, 'IMU', f'imu_{start_end[0]}_{start_end[1]}.pkl')}")

def load_data(file_list, base_dir="", start_end = None):

	head_imu_input, ee_pos, output, global_p, local_rot, head_start, c_lr, start_end = load_data_from_gimo(base_dir, file_list, start_end = start_end)

	# set necessary information to dictionary
	total, seq_len, _  = output.shape
	input_ = {}
	input_['input_seq'] = head_imu_input 
	input_['mid_seq'] = ee_pos 
	input_['tgt_seq'] = output 
	input_['global_p'] = global_p
	input_['root'] = global_p[..., 0, :] 
	input_['local_rot'] = local_rot
	input_['head_start'] = head_start 
	input_['contact_label'] = c_lr
	input_['start_end'] = start_end

	return input_, total

def parse_filenames_and_load(args):

	if not os.path.exists(args.preprocess_path):
		os.mkdir(args.preprocess_path)
	
	base_dir = args.base_dir
	df = pd.read_csv(os.path.join(base_dir, 'dataset.csv'))	
	# Aligned
	fnames_list = (df['scene'].astype(str) + '/' + df['sequence_path'].astype(str)).tolist()
	start_end_list = list(zip(df['start_frame'].astype(int), df['end_frame'].astype(int)))
 
	for fnames, start_end in tqdm(zip(fnames_list, start_end_list)):
		bvh_list = [f for f in os.listdir(os.path.join(base_dir, fnames, 'smplx_local')) if f.endswith('.pkl')]
		bvh_list.sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
		load_data_with_args(fnames=fnames, bvh_list=bvh_list, start_end = start_end, args=args)
		
	dest = os.path.join(args.preprocess_path, "data_config")
	utils.create_dir_if_absent(dest)
	config_copy_command = f"cp -r {args.data_config_path} {args.preprocess_path}"
	os.system(config_copy_command)

if __name__ == "__main__":
	import torch.multiprocessing as mp
	mp.set_start_method('spawn')
	# add argparse
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
		"--preprocess-path",
		type=str,
		default=True
	)
	
	args = parser.parse_args()
	
	# for generating preprocessed pkl files
	parse_filenames_and_load(args)

	# for debugging
	# result = load_data_from_amass("../data/amass", ["CMU/01/01_01_stageii.npz"])
