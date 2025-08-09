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
import json

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
import imu2body.egobody as egobody
import imu2body_eval.amass_smplh as amass_smplh
from fairmotion.utils import utils
from tqdm import tqdm
from copy import deepcopy
import constants.imu as imu_constants
import constants.motion_data as motion_constants
import imu2body.imu as imu
from interaction.contact import *
import pandas as pd
from functools import partial
from multiprocessing import Pool, cpu_count

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

smplx_bm_path = "../data/smpl_models/smplx/SMPLX_NEUTRAL.npz"
smplh_bm_path = "../data/smpl_models/smplh/male/model.npz"

CUR_BM_TYPE = "smplx"





def process_sequence(seq, base_dir, cur_bm_type='smplx'):
    if seq['dataset'] == 'gimo':
        filepath_list = [os.path.join(base_dir, 'GIMO', seq['fname'], 'smplx_local', file) for file in seq['file']]
        transform_info = json.load(open(os.path.join(base_dir, 'GIMO', seq['fname'], seq['transform']), 'r')) # pose to scene transformation
        transform_norm = np.loadtxt(os.path.join(base_dir, 'GIMO', seq['fname'], '../', 'scene_obj', 'transform_norm.txt')).reshape((4, 4))
        pkl_files = [f for f in filepath_list if f.endswith('.pkl')]
    elif seq['dataset'] == 'egobody':
        pkl_files = [os.path.join(base_dir, 'Egobody_dataset', f"smplx_camera_wearer_{seq['mode']}", seq['fname'], seq['body_index'], 'results', file, '000.pkl') for file in seq['file']]
        transform_info = None
        transform_norm = None

    # read skel and files	
    if CUR_BM_TYPE == "smplx":
        if seq['dataset'] == 'gimo':
            bm_path = smplx_bm_path
            body_model = gimo.load_body_model(bm_path=bm_path)
            skel_with_offset = gimo.create_skeleton_from_amass_bodymodel(bm=body_model)	
            skel = skel_with_offset[0]
            motion = gimo.create_motion_from_gimo_data(pkl_files, 
                                                        bm=body_model, 
                                                        transform_info = transform_info, 
                                                        transform_norm = transform_norm, 
                                                        skel_with_offset=deepcopy(skel_with_offset))
        elif seq['dataset'] == 'egobody':
            bm_path = smplx_bm_path
            body_model = egobody.load_body_model(bm_path=bm_path)
            skel_with_offset = egobody.create_skeleton_from_amass_bodymodel(bm=body_model)	
            skel = skel_with_offset[0]
            motion = egobody.create_motion_from_egobody_data(pkl_files, 
                                                            bm=body_model, 
                                                            transform_info = transform_info, 
                                                            transform_norm = transform_norm, 
                                                            skel_with_offset=deepcopy(skel_with_offset))
    
    else:
        raise NotImplementedError("Only smplx are supported!")
    
    return {
        'motion': motion,
        'seq': seq,
        'skel': skel
    }




def load_data_from_training(base_dir, file_list, setting = 'vr', debug=False, normalization = False):
    logging.info(f"Start loading {len(file_list)} sequences with multiprocessing...")

    try:
        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(pool.imap(partial(process_sequence, base_dir=base_dir), file_list), total=len(file_list)))
    except KeyboardInterrupt:
        print("User cancelled. Terminating pool...")
        pool.terminate()
        pool.join()
        raise
    except Exception as e:
        print(f"Error: {e}. Terminating pool...")
        pool.terminate()
        pool.join()
        raise
    finally:
        if pool:
            pool.close()
            pool.join()


    motion_list = []
    data_set_info = []
    for res in results:
        if res['motion'] is not None:
            motion_list.append(res['motion'])
            data_set_info.append(res['seq'])
            skel = res['skel']  # Use skel from the last one


    logging.info(f"Done converting GIMO and Egobody dataset into fairmotion Motion class")

    # read list
    local_T = [] 
    global_T = []

    # imu signal list
    imu_rot = []
    imu_acc = []

    # contact labels
    c_lr = []
 
    # slice_info list
    info_list = []
    if setting == 'vr':
        ee_joint_names = motion_constants.FOOT_JOINTS
        ee_joint_idx = [skel.get_index_joint(jn) for jn in ee_joint_names]
    else:
        ee_joint_names = imu_constants.imu_joint_names + motion_constants.FOOT_JOINTS
        ee_joint_idx = [skel.get_index_joint(jn) for jn in ee_joint_names]

    imu_joint_names = imu_constants.imu_joint_names
    imu_joint_idx = [skel.get_index_joint(jn) for jn in imu_joint_names]

    # constants
    window = motion_constants.preprocess_window
    offset = motion_constants.preprocess_offset
 
    is_custom_run = False
    for motion, info in tqdm(zip(motion_list, data_set_info)):
        height_indice = 1 if info['dataset'] == 'gimo' else 2 # y for GIMO and z for egobody
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
        start_frame, end_frame = info['start_end'][0], info['start_end'][1] #[242, 528]
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
                    local_T_window_height_adjust[:, 0, height_indice, 3] -= cur_height_offset
                    global_T_window_height_adjust[..., height_indice, 3] -= cur_height_offset

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

    c_lr = np.asarray(c_lr).astype(dtype=np.float32)
    c_lr = c_lr.transpose(0,2,1)

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

    else: # IMU ROT AND ACC IS ONLY USED IN HMC SETTING/ For VR WE USE 3D Pos DIRECTLY
        normalized_imu_rot = imu_rot  # [Window #, seq, 2, 3, 3]
        normalized_imu_acc = imu_acc # [Window #, seq, 2, 3]
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
    # [1, 3, 6+3]
 
    # mid (output of 1st network, input of 2nd network)
    ee_pos = normalized_global_T[..., ee_joint_idx, :3, 3]	
    reshaped_ee_pos = np.transpose(ee_pos, (1, 2, 0, 3))
    ee_pos_v = reshaped_ee_pos.reshape(batch, seq_len, -1)

    if debug:
        return normalized_imu_rot, normalized_imu_acc, ee_pos_v, local_T, normalized_global_T, head_start_T 

    local_rotation_6d = T_to_6d_rot(local_T)
    local_rotation_6d = local_rotation_6d.reshape(batch, seq_len, -1)

    output = np.concatenate((normalized_global_T[...,0,:3,3], local_rotation_6d), axis=-1) # [# of windows, seq_len, 6J+3]	
    
    # return global pos for FK loss calc
    global_p = normalized_global_T[...,:3,3]

    tot_length = motion_local_T.shape[0]

    return head_imu_input, ee_pos_v, output, global_p, local_T[...,:3,:3], head_start_T, c_lr, info_list, tot_length

def load_data_with_args_train(file_list, args, mode = 'train', save_name=None):
    data, total_len = load_data(file_list, base_dir=args.base_dir, setting=args.setting, mode = mode)            
    if save_name is not None:
        write_path = os.path.join(args.preprocess_path, f'{save_name}_vr.pkl')
    else:
        write_path = os.path.join(args.preprocess_path, f'{mode}_vr.pkl')
    with open(write_path, "wb") as f_write:
        pickle.dump(data, f_write, protocol=pickle.HIGHEST_PROTOCOL)
    logging.info(f"Saved {mode} data with {total_len} sequences in {write_path}")

def load_data(file_list_total, base_dir = "", setting = 'vr', mode = 'train'):
    file_list = [f for f in file_list_total if f['mode'] == mode]

    head_imu_input, ee_pos, output, global_p, local_rot, head_start, c_lr, info, tot_length= load_data_from_training(base_dir, file_list, setting = setting , normalization=True)

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
    input_['info'] = info
    input_['total_length'] = tot_length

    return input_, total

def parse_filenames_and_load(args):
    if not os.path.exists(args.preprocess_path):
        os.mkdir(args.preprocess_path)
    base_dir = args.base_dir
    gimo_path = os.path.join(base_dir, 'GIMO')
    egobody_path = os.path.join(base_dir, 'Egobody_dataset')
 
    gimo_data_info = pd.read_csv(os.path.join(gimo_path, 'dataset.csv'))	
    egobody_data_info = pd.read_csv(os.path.join(egobody_path, 'data_info_release.csv'))
    egobody_data_split_info = pd.read_csv(os.path.join(egobody_path, 'data_split.csv'))
    
    # ===================== GIMO =====================
    # Aligned - GIMO
    fnames_list = (gimo_data_info['scene'].astype(str) + '/' + gimo_data_info['sequence_path'].astype(str)).tolist()
    start_end_list = list(zip(gimo_data_info['start_frame'].astype(int), gimo_data_info['end_frame'].astype(int)))
    scene_list = (gimo_data_info['scene'].astype(str)).tolist()
    transform_info = gimo_data_info['transformation']
    training = gimo_data_info['training']
    
    file_lists = []
    file_list_gimo = []
    file_list_egobody = []
    for fnames, start_end, transform, scene, training in tqdm(zip(fnames_list, start_end_list, transform_info, scene_list, training), desc='load GIMO seqlists'): # GIMO Dataset
        seqlists = {}
        seqlists['fname'] = fnames
        seqlists['start_end'] = start_end
        seqlists['scene'] = scene
        seqlists['transform'] = transform
        seqlists['file'] = [f for f in os.listdir(os.path.join(gimo_path, fnames, 'smplx_local')) if f.endswith('.pkl')]
        seqlists['file'].sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        seqlists['mode'] = 'train' if training == 1 else 'test'     # NOTE
        seqlists['dataset'] = 'gimo'
        file_lists.append(seqlists) #NOTE: comment this line to debug Egobody
        file_list_gimo.append(seqlists)
    

    # ===================== EgoBody =====================
    fnames_list = (egobody_data_info['recording_name'].astype(str)).tolist()
    start_end_list = list(zip(egobody_data_info['start_frame'].astype(int), egobody_data_info['end_frame'].astype(int)))
    scene_list = (egobody_data_info['scene_name'].astype(str)).tolist()
 
    for fnames, start_end, scene in tqdm(zip(fnames_list, start_end_list, scene_list), desc='load egobody seqlists'):
        for col in egobody_data_split_info.columns:
            if bool((egobody_data_split_info[col] == fnames).any()):
                mode = col
                break
        if mode == 'val' or mode == '': # NOTE we dont process VAL mode in this script
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
        seqlists['mode'] = mode
        seqlists['dataset'] = 'egobody'
        file_lists.append(seqlists)
        file_list_egobody.append(seqlists)
  
    load_data_with_args_train(file_list = file_lists, args=args, mode = 'train')
 
    load_data_with_args_train(file_list = file_lists, args=args, mode = 'test')

    load_data_with_args_train(file_list = file_list_gimo, args=args, mode = 'test', save_name='test_gimo')
    load_data_with_args_train(file_list = file_list_egobody, args=args, mode = 'test', save_name='test_egobody')

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
    parser.add_argument(
        "--setting",
        type=str,
        default ='vr',
        choices = ['vr', 'hmc']
    )
    args = parser.parse_args()
    
    # for generating preprocessed pkl files
    parse_filenames_and_load(args)