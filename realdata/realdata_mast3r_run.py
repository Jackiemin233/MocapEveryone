# Package import
import sys
import os
sys.path.insert(0, os.path.dirname(__file__) + '/..')

import torch
import argparse
import numpy as np
from glob import glob
from pycocotools import mask as masktool

from lib.pipeline import video2frames, detect_segment_track
from lib.camera import run_mast3r_metric_slam
from lib.utils.imutils import copy_images 


def main(args):
    # Step1: Run Mast3r-slam
    file = args.input
    root = os.path.dirname(file)
    seq = os.path.basename(file).split('.')[0]

    seq_folder = f'results/{seq}'
    img_folder = f'{seq_folder}/images'
    hps_folder = f'{seq_folder}/hps'
    os.makedirs(seq_folder, exist_ok=True)
    os.makedirs(img_folder, exist_ok=True)

    # Step1.1: extract frames
    print('Extracting frames ...')
    if file.endswith('.mov') or file.endswith('.mp4') or file.endswith('.MP4'):
        nframes = video2frames(file, img_folder)
    else:
        copy_images(file, img_folder)
        
    # Step1.2: Obtain camera intrinsics
    cam_int = None #[intr[0,0], intr[1,1], intr[0,2], intr[1,2]]
    
    print('Masked Metric SLAM ...')
    traj, pc_whole, pc, kf_idx = run_mast3r_metric_slam(img_folder, None, cam_int, seq)
    
    # Step2: Run SMPL Diffusion
    
    # Step2.1 Load diffusion model
    
    # Step2.2 SMPL initialization
    
    # Step2.3 SMPL Denoising
    
    # Step3: Visualization and evluation


if __name__ == "__main__": 
    import torch.multiprocessing as mp
    mp.set_start_method("spawn")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='./data/realdata/1029/video/ego_video.MP4', help='path to your EMDB Test Samples')
    parser.add_argument("--visualize_mask", action='store_true', default=True, help='save deva vos for visualization')
    parser.add_argument('--max_humans', type=int, default=20, help='maximum number of humans to reconstruct')
    parser.add_argument('--output_dir', type=str, default='results', help='the output save directory')
    parser.add_argument('--bin_size', type=int, default=-1, help='rasterization bin_size; set to [64,128,...] to increase speed')
    parser.add_argument('--floor_scale', type=int, default=3, help='size of the floor')
    args = parser.parse_args()
    main(args)


