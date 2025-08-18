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
import multiprocessing as mp
from functools import partial

from IPython import embed
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
import open3d as o3d
import trimesh

logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

bm_path = "../data/smpl_models/smplh/male/model.npz"
smplx_bm_path = "../data/smpl_models/smplx/SMPLX_NEUTRAL.npz"
smplh_bm_path = "../data/smpl_models/smplh/male/model.npz"

CUR_BM_TYPE = "smplx"

def vis_points(scene_list, global_p, i):
    input_xyz = global_p[i].reshape(-1, 3)
    input_xyz = np.array(input_xyz)
    scene_points = scene_list[i]

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(input_xyz)
    pcd1.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (input_xyz.shape[0], 1)))  # 红色 (RGB: 1, 0, 0)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(scene_points)
    pcd2.colors = o3d.utility.Vector3dVector(np.tile([0, 1, 0], (scene_points.shape[0], 1)))  # 绿色 (RGB: 0, 1, 0)

    # 合并两个点云
    combined_pcd = pcd1 + pcd2

    # 导出为带颜色的 .ply 文件
    print(f'Writing vis to ./vis/test_pcd_{i}.ply')
    o3d.io.write_point_cloud(f'./vis/test_pcd_{i}.ply', combined_pcd)

def load_scene(dataset_info_gimo, dataset_info_egobody, gimo_dataroot, egobody_dataroot, 
               scene_downsample_points=60000):
    scene_list = {}
    for i, seq in enumerate(tqdm(dataset_info_gimo['sequence_path'], desc="Loading GIMO scene")):  # for GIMO
        scene = dataset_info_gimo['scene'][i]
        scene_key = f"{scene}_{seq}"
        scene_dir = os.path.join(gimo_dataroot, scene, 'scene_obj')
        full_ply_path = os.path.join(scene_dir, 'scene_downsampled.ply')  # 原始场景点云
        downsample_ply_path = os.path.join(scene_dir, f'scene_downsampled_{scene_downsample_points}.ply')

        if os.path.exists(downsample_ply_path):
            # 如果已经存在 downsampled 文件，直接读取
            pcd = o3d.io.read_point_cloud(downsample_ply_path)
            scene_points = np.asarray(pcd.points, dtype=np.float32)
        else:
            raise NotImplementedError

        scene_list[scene_key] = scene_points

    
    for i, seq in enumerate(tqdm(dataset_info_egobody['recording_name'], desc="Loading EgoBody scene")):
        scene = dataset_info_egobody['scene_name'][i]
        scene_key = f"{scene}_{seq}"
        scene_dir = os.path.join(egobody_dataroot, 'scene_mesh', scene)
        ply_path = os.path.join(scene_dir, f'{scene}.obj')
        
        # Downsampled point cloud save path
        downsample_ply_path = os.path.join(scene_dir, f'{scene}_downsampled_{scene_downsample_points}.ply')

        if os.path.exists(downsample_ply_path):
            # Load downsampled point cloud from .ply
            downsampled_mesh = o3d.io.read_point_cloud(downsample_ply_path)
            scene_points = np.asarray(downsampled_mesh.points, dtype=np.float32)
        else:
            raise NotImplementedError
        
        scene_list[scene_key] = scene_points
    print('Scene load done')

    return scene_list




def load_data_from_training(base_dir, file, scene_list, setting = 'vr', gimo_dataroot=None, egobody_dataroot=None, 
                            debug=False, normalization = False, cxt_num_points=1024, save_path=None):
    motion_list = []
    data_set_info = []
 
    seq = file
    if seq['dataset'] == 'gimo':
        filepath_list = [os.path.join(base_dir, 'GIMO', seq['fname'], 'smplx_local', file) for file in seq['file']]
        transform_info = json.load(open(os.path.join(base_dir, 'GIMO', seq['fname'], seq['transform']), 'r')) # pose to scene transformation
        transform_norm = np.loadtxt(os.path.join(base_dir, 'GIMO', seq['fname'], '../', 'scene_obj', 'transform_norm.txt')).reshape((4, 4))
        # start, end = seq['start_end'][0], seq['start_end'][1]
        # scene (pose) normalization transformation
        pkl_files = [f for f in filepath_list if f.endswith('.pkl')][seq['start_end'][0] : seq['start_end'][1]]

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
            motion_list.append(gimo.create_motion_from_gimo_data(pkl_files, 
                                                                bm=body_model, 
                                                                transform_info = transform_info, 
                                                                transform_norm = transform_norm, 
                                                                skel_with_offset=deepcopy(skel_with_offset)))
            data_set_info.append(seq)
        elif seq['dataset'] == 'egobody':
            bm_path = smplx_bm_path
            body_model = egobody.load_body_model(bm_path=bm_path)
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
        ee_joint_names = motion_constants.FOOT_JOINTS
        ee_joint_idx = [skel.get_index_joint(jn) for jn in ee_joint_names]
    else:
        ee_joint_names = imu_constants.imu_joint_names + motion_constants.FOOT_JOINTS
        ee_joint_idx = [skel.get_index_joint(jn) for jn in ee_joint_names]

    # constants
    window = motion_constants.preprocess_window
    offset = motion_constants.preprocess_window
    height_indice = 1

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
        start_frame, end_frame = info['start_end'][0], info['start_end'][1]
        i = start_frame
        while True:
            if i >= end_frame:
                break
            if i + window >= end_frame:
                i = end_frame - window
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
    ee_pos = normalized_global_T[..., ee_joint_idx, :3, 3]	
    reshaped_ee_pos = np.transpose(ee_pos, (1, 2, 0, 3))
    ee_pos_v = reshaped_ee_pos.reshape(batch, seq_len, -1)

    if debug:
        return normalized_imu_rot, normalized_imu_acc, ee_pos_v, local_T, normalized_global_T, head_start_T 

    local_rotation_6d = T_to_6d_rot(local_T)
    local_rotation_6d = local_rotation_6d.reshape(batch, seq_len, -1)

    output = np.concatenate((normalized_global_T[...,0,:3,3], local_rotation_6d), axis=-1) # [# of windows, seq_len, 6J+3]	
    
    # return global pos for FK loss calc
    global_p = normalized_global_T[...,:3,3].astype(np.float32)

    total, seq_len, _  = output.shape
    result_dict = {}
    result_dict['input_seq'] = torch.Tensor(head_imu_input).float() 
    result_dict['mid_seq'] = torch.Tensor(ee_pos_v).float()
    result_dict['tgt_seq'] = torch.Tensor(output).float() 
    result_dict['global_p'] = torch.Tensor(global_p).float()
    result_dict['root'] = torch.Tensor(global_p[..., 0, :]).float() 
    result_dict['local_rot'] = torch.Tensor(local_T[...,:3,:3]).float()
    result_dict['head_start'] = torch.Tensor(head_start_T)
    result_dict['head_start_invert'] = torch.Tensor(head_invert)
    result_dict['info'] = info_list
    result_dict['total_length'] = motion_local_T.shape[0]


    N = len(info_list)
    out = [None] * N

    for i, info in enumerate(info_list):
        s = info['seq']
        scene = info['scene']
        transform_path = info['transform']
        head_start = head_start_T[i]
        root_pose = global_p[i][:, 0]
        head_start_invert = head_invert[i]
        root_pose_w = (head_start @ xyz_to_transform_matrices(root_pose))[0, :, :3, 3]

        if info['dataset'] == 'gimo':

            transform_info_file = os.path.join(gimo_dataroot, s, transform_path)
            
            transform_norm_file = os.path.join(gimo_dataroot, scene, 'scene_obj', 'transform_norm.txt')
            transform_norm = np.loadtxt(transform_norm_file).reshape((4, 4)).astype(np.float32)

            # Load transforms from dict caches (already prepared in load_scene)
            transform_info_file = os.path.join(gimo_dataroot, s, transform_path)
            transform_info = json.load(open(transform_info_file, 'r'))
            scale = transform_info['scale']
            transform_norm_file = os.path.join(gimo_dataroot, scene, 'scene_obj', 'transform_norm.txt')
            transform_norm = np.loadtxt(transform_norm_file).reshape((4, 4)).astype(np.float32)
            transform_norm[:3, 3] /= scale

            scene_key = s.replace('/', '_')
            scene_points = scene_list[scene_key].astype(np.float32)  # base scene pcd
            scene_points *= 1.0 / scale
            scene_points = (transform_norm[:3, :3] @ scene_points.T + transform_norm[:3, 3:]).T

            # Crop -> then map back with head_start_invert (same as your __getitem__)
            cropped = extract_points_in_bbox(scene_points, root_pose_w, radius=1.5, cxt_num_points=cxt_num_points)
            
            cropped = (head_start_invert @ xyz_to_transform_matrices(cropped))[0, :, :3, 3]
            out[i] = cropped.astype(np.float32)

        elif info['dataset'] == 'egobody':
            scene = info['scene']
            s = info['seq']
            scene_key = f"{scene}_{s}"
            scene_points = scene_list[scene_key].astype(np.float32)

            cam2world_dir = os.path.join(egobody_dataroot, 'calibrations', s, 'cal_trans/kinect12_to_world')  
            with open(os.path.join(cam2world_dir, scene + '.json'), 'r') as f:
                trans = np.array(json.load(f)['trans'])
            trans = np.linalg.inv(trans)

            scene_points_w = trimesh.transform_points(scene_points, trans).astype(np.float32)            

            cropped = extract_points_in_bbox(scene_points_w, root_pose_w, radius=1.5, cxt_num_points=cxt_num_points)

            cropped = (head_start_invert @ xyz_to_transform_matrices(cropped))[0, :, :3, 3]
            out[i] = cropped.astype(np.float32)
        else:
            # Fallback: zero points (shouldn't happen)
            out[i] = np.zeros((cxt_num_points, 3), dtype=np.float32)
    
    result_dict['scene_points'] = torch.from_numpy(np.stack(out)).float()

    # vis
    vis_points(out, global_p, 0)
    

    # save
    test_save_path = os.path.join(os.path.join(save_path), f"{seq['dataset']}_test", f"{seq['fname'].replace('/', '-')}.pkl")
    print(f"Saving files to {test_save_path}")
    with open(test_save_path, "wb") as file:
        pickle.dump(result_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
   
def load_filelist(args):
    test_txt_filename = ""
    assert args.data_type == 'train', "Unsupported data type"
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
            seqlists['start_end_origin'] = start_end
            seqlists['start_end'] = (0, start_end[1]-start_end[0])
            seqlists['scene'] = scene
            seqlists['transform'] = transform
            seqlists['file'] = [f for f in os.listdir(os.path.join(gimo_path, fnames, 'smplx_local')) if f.endswith('.pkl')]
            seqlists['file'].sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
            seqlists['mode'] = 'test'
            seqlists['dataset'] = 'gimo'
            file_lists.append(seqlists) #NOTE: comment this line to debug Egobody

        # ===================== EgoBody =====================
        fnames_list = (egobody_data_info['recording_name'].astype(str)).tolist()
        start_end_list = list(zip(egobody_data_info['start_frame'].astype(int), egobody_data_info['end_frame'].astype(int)))
        scene_list = (egobody_data_info['scene_name'].astype(str)).tolist()
    
        # for fnames, start_end, scene in tqdm(zip(fnames_list, start_end_list, scene_list)):
        #     for col in egobody_data_split_info.columns:
        #         if bool((egobody_data_split_info[col] == fnames).any()):
        #             mode = col 
        #             break
        #     if mode != 'test': #Only test
        #         continue
        #     seqlists = {}
        #     seqlists['fname'] = fnames
        #     seqlists['scene'] = scene
        #     seqlists['start_end_origin'] = start_end
        #     seqlists['start_end'] = (0, start_end[1]-start_end[0]+1)
        #     # print(start_end)
        #     seqlists['transform'] = None
        #     seqlists['body_index'] = os.listdir(os.path.join(egobody_path, f'smplx_camera_wearer_{mode}', fnames))[0]
        #     seqlists['file'] = [f for f in os.listdir(os.path.join(egobody_path, f'smplx_camera_wearer_{mode}', fnames, seqlists['body_index'], 'results'))]
        #     seqlists['file'].sort(key=lambda x: int(''.join(filter(str.isdigit, x))))
        #     # print(seqlists['file'][0], seqlists['file'][-1])
        #     assert str(start_end[0]) in seqlists['file'][0] and str(start_end[1]) in seqlists['file'][-1]
        #     seqlists['mode'] = 'test'
        #     seqlists['dataset'] = 'egobody'
        #     file_lists.append(seqlists)
    
    ds_scene_list = load_scene(gimo_data_info, egobody_data_info, gimo_path, egobody_path)

    if args.data_type == 'train':
        os.makedirs(os.path.join(args.save_path, 'gimo_test'), exist_ok=True)
        os.makedirs(os.path.join(args.save_path, 'egobody_test'), exist_ok=True)

        for file in tqdm(file_lists, desc="Writing preprocessed file lists"):
            load_data_from_training(base_dir = args.base_dir, file = file, scene_list=ds_scene_list, 
                                    gimo_dataroot=gimo_path, egobody_dataroot=egobody_path,
                                    normalization = True, setting = args.setting, save_path=args.save_path)
        
def get_optimal_process_count():
    """获取最优的进程数量"""
    cpu_count = mp.cpu_count()
    # 建议使用CPU核心数的75%，避免系统过载
    optimal_count = max(1, int(cpu_count * 0.75))
    return optimal_count

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