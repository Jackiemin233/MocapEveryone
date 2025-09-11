import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
import argparse
from copy import deepcopy
import logging
import glob
import numpy as np
import random
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import torch
import torch.nn as nn
import torch.optim as optim
import smplx

from IPython import embed 
import yaml
from dataset import get_loader_training, get_loader_validation
from functions import *
from pytorch3d import transforms
import dadaptation
from fairmotion.data import bvh
import amass
import imu2body_eval.amass_smplh as amass_smplh
from tqdm import tqdm
from fairmotion.ops import conversions
from imu2body.visualize_testset import RenderData 
from imu2body.loss import *
import imu2body.model
import constants.motion_data as motion_constants
from eval.metrics import * 

from tensorboardX import SummaryWriter

from accelerate import Accelerator

'''
    Start Command:
    python run_gimo.py --test_name=Baseline --mode=train --config=example_config
 
    For training logs:
    tensorboard --logdir=/home/zhanggangjian/nanjie/project6/MocapEvery/imu2body/output/lstm --host=127.0.0.1

    For test:
    python3 run_gimo.py --test_name=lstm_40 --mode=test
 
     NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch run_gimo.py --test_name=LSTM_Temporal --mode=train --config=example_config
  
    python run_gimo.py --test_name=lstm --mode=train --config=example_config
    
    python run_gimo_train.py --test_name=VR_Baseline --mode=train --config=config_temporal_wilor
    
    python run_gimo_train.py --test_name=Debug_VR_wavelet --mode=train --config=config_temporal_wilor
'''
 
logging.basicConfig(
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

change_mode_epoch = 40
change_loss_epoch = 80

def set_seeds():
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


smplx_bm_path = "../data/smpl_models/smplx/SMPLX_NEUTRAL.npz"
smplh_bm_path = "../data/smpl_models/smplh/male/model.npz"

CUR_BM_TYPE = "smplx"

def get_warmup_weight(step: int,
                      start: float,
                      end: float,
                      warmup_steps: int,
                      start_step: int = 0) -> float:
    """
    线性从 start → end 的权重调度:
      step < start_step         -> 返回 start
      start_step~start_step+warmup_steps 线性上升
      之后固定为 end
    """
    if step <= start_step:
        return start
    t = min(1.0, (step - start_step) / max(1, warmup_steps))
    return start + t * (end - start)


class IMU2BodyNetwork(object):
    def __init__(self, args):
        set_seeds()

        # init 
        directory = "./output/" + args.test_name + "/"
        if not os.path.exists(directory):
            os.mkdir(directory)

        self.directory = directory
        self.mode = args.mode
        self.setting = args.setting
        
        # open config file
        config_dir = "./config/"+args.config + ".yaml" if self.mode == "train" else self.directory + "config.yaml"
        self.config = yaml.safe_load(open(config_dir, 'r').read())

        if self.mode == "train":
            os.system('cp {} {}'.format('./config/'+args.config+'.yaml', directory+'config.yaml'))

        self.data_path = self.config['data']['preprocess']

        logging.info(f"Starting in {self.mode} mode...")

        self.set_info()
        self.load_data_gimo()  
        
        self.build_network_gimo()
        self.build_optimizer()

        self.smplx_model = smplx.create("../data/smpl_models", model_type='smplx',
                    gender='neutral', use_pca=False, use_face_contour=True, flat_hand_mean=True)


        self.accelerator = Accelerator()
        print('accelerate is preparing')
  
        self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)

        for k, v in self.dataloader.items():
            self.dataloader[k] = self.accelerator.prepare(v)

        print('accelerate is Ready')
        
        # BUG For test
        # self.eval_files_gimo = glob.glob(os.path.join(self.data_path, 'gimo_test', "*.pkl"))
        # self.eval_files_egobody = glob.glob(os.path.join(self.data_path, 'egobody_test', "*.pkl"))
        self.eval_files_gimo = glob.glob(os.path.join(self.data_path, 'gimo_test_vis', "*.pkl"))
        self.eval_files_egobody = glob.glob(os.path.join(self.data_path, 'egobody_test_vis', "*.pkl"))
        self.eval_metric = ['mpjre', 'mpjpe', 'mpjve', 'pred_jitter', 'root_mpjpe', 'rootpe', 'upperpe', 'lowerpe', 'gt_jitter']

    def set_info(self, pretrain=False):
        is_train = True if self.mode == "train" else False
        self.pretrain = pretrain if is_train else True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")
        
        # set information from config and args
        self.num_epoch_eval	= self.config['eval']['num_epoch_eval']
        self.save_frequency = self.config['train']['save_frequency']
        self.use_uncertainty = self.config['model'].get('uncertainty_model', False)
        self.use_swt_ada = self.config['model'].get('use_swt_ada', False)
        self.use_freq_time_decom_loss = self.config['model'].get('use_freq_time_decom_loss', False)
        self.set_skel_info() # load skeleton info (this is needed for train and test)
        
        self.log_dir = os.path.join(self.directory, "log/")
        self.model_dir = os.path.join(self.directory, "model/")
        self.code_dir = os.path.join(self.directory, "code/")

        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        if not os.path.exists(self.code_dir):
            os.mkdir(self.code_dir)

        cur_dir = os.path.dirname(os.path.abspath(__file__))

        ## copy some code to log dir as a backup
        if self.mode == "train":
            logging.info("COPYING IMPORTANT FILES")
            copy_files = ['preprocess_train_mp.py', 'dataset.py', 'gimo.py', 'functions.py', 'model_base_swt_new.py', 'model_base_swt_ada.py',
                        'egobody.py', 'model.py', 'model_base.py', 'pointnet2.py', 'run_gimo_train.py', 'model_base_swt.py', 'loss.py', 
                        'model_base_swt_ada_cxt.py',]
            for file in copy_files:
                os.system(f'cp -r {cur_dir}/{file} {self.code_dir}')

        self.train_epoch = 0
        
        self.loss_func = self.get_loss

    def set_skel_info(self):
        
        if CUR_BM_TYPE == "smplx":
            body_model = amass.load_body_model(motion_constants.BM_PATH) 
            fairmotion_skel, _ = amass.create_skeleton_from_amass_bodymodel(bm=body_model)

        elif CUR_BM_TYPE == "smplh":
            body_model = amass_smplh.load_body_model(motion_constants.SMPLH_BM_PATH) 
            fairmotion_skel, _ = amass_smplh.create_skeleton_from_amass_bodymodel(bm=body_model)
        else:
            raise NotImplementedError("Only smplx and smplh are supported!")
        
        self.skel = fairmotion_skel
        self.skel_offset = fairmotion_skel.get_joint_offset_list()
        self.skel_parent = fairmotion_skel.get_parent_index_list()
        self.ee_idx = fairmotion_skel.get_index_joint(motion_constants.EE_JOINTS)
        self.foot_idx = fairmotion_skel.get_index_joint(motion_constants.FOOT_JOINTS) + fairmotion_skel.get_index_joint(motion_constants.toe_joints)
        self.hand_idx = fairmotion_skel.get_index_joint(motion_constants.HAND_JOINTS)
        # this is to solve overfitting on foot
        self.leg_idx = fairmotion_skel.get_index_joint(motion_constants.LEG_JOINTS)
        # BUG
        # if self.setting == 'vr':
        #     self.mid_ee_idx = self.hand_idx + fairmotion_skel.get_index_joint(motion_constants.FOOT_JOINTS)
        # else:
        #     self.mid_ee_idx = fairmotion_skel.get_index_joint(motion_constants.FOOT_JOINTS)
        if self.setting == 'vr':
            self.mid_ee_idx = fairmotion_skel.get_index_joint(motion_constants.FOOT_JOINTS)
        else:
            self.mid_ee_idx = self.hand_idx + fairmotion_skel.get_index_joint(motion_constants.FOOT_JOINTS)
        self.skel_offset = torch.from_numpy(self.skel_offset[np.newaxis, np.newaxis, ...]).to(self.device).float() 		# expand skel offset into tensor
        if self.mode == "train":
            self.skel_offset = self.skel_offset.repeat(self.config['train']['batch_size'], motion_constants.preprocess_window, 1, 1)
    
    def load_data_gimo(self):
        is_train = False

        fnames = [self.mode]
        if self.mode == "train":
            fnames.append("validation")
            is_train = True

        self.dataloader = {}

        data_root = self.config['data']['preprocess'] # NOTE: Hard code
        if is_train is False:
            data = np.load(self.directory + "mean_and_std.npz")
            self.mean = data['mean']
            self.std = data['std']

            x_data = np.load(self.directory + "x_mean_and_std.npz")
            self.x_mean = x_data['mean']
            self.x_std = x_data['std']
  
        for fname in fnames:
            if fname == "train":
                is_train = True
                batch_size = self.config['train']['batch_size']
                self.dataloader[fname] = get_loader_training(data_root=data_root, \
                                                        batch_size=batch_size, \
                                                        training=is_train,
                                                        num_workers=self.config['train']['num_workers'])
                self.mean = self.dataloader['train'].dataset.mean
                self.std = self.dataloader['train'].dataset.std 
    
                self.x_mean = self.dataloader['train'].dataset.x_mean
                self.x_std = self.dataloader['train'].dataset.x_std
    
                np.savez(self.directory+"mean_and_std", mean=self.mean, std=self.std)

                np.savez(self.directory+"x_mean_and_std", mean=self.x_mean, std=self.x_std)
    
            else:           # validation
                is_train = False
                batch_size = self.config['train']['batch_size']
                self.dataloader[fname] = get_loader_training(data_root=data_root, \
                                                        batch_size=batch_size, \
                                                        training=is_train,
                                                        num_workers=self.config['train']['num_workers'])
                # self.dataloader[f'{fname}_gimo'] = get_loader_validation(data_root=data_root, \
                #                                                     batch_size=batch_size, \
                #                                                     dataset='gimo',
                #                                                     num_workers=self.config['train']['num_workers'])
                # self.dataloader[f'{fname}_egobody'] = get_loader_validation(data_root=data_root, \
                #                                                     batch_size=batch_size, \
                #                                                     dataset='egobody',
                #                                                     num_workers=self.config['train']['num_workers'])
                
    
        # convert to tensor for future calculations
        self.mean = torch.from_numpy(self.mean).to(self.device, dtype=torch.float32)
        self.std = torch.from_numpy(self.std).to(self.device, dtype=torch.float32)
        self.x_mean = torch.from_numpy(self.x_mean).to(self.device, dtype=torch.float32)
        self.x_std = torch.from_numpy(self.x_std).to(self.device, dtype=torch.float32).view(1, 1, motion_constants.NUM_JOINTS, 3)
 
    def build_network_gimo(self):

        logging.info(f"Loading model...")

        data_dict = self.dataloader[self.mode].dataset.get_data_dict()
        model_dict = self.config['model']

        self.model = imu2body.model.load_model(data_config=data_dict, model_config=model_dict)
        self.model.train()
        self.model.zero_grad()
        
        self.criterion = nn.L1Loss()
        # self.scale3d = ScaleSeparated3D(
        #     J=22,          # 例如 24 或 22
        #     lp_kernel=15, lp_cutoff=0.25, causal=False,
        #     w_lf=1.0, w_hf=0.5, w_vel=0.2
        # ).to(self.device)

        self.scale3d = RobustScaleSeparated3D(
            J=22, pelvis_idx=0,  # 例如 0
            lp_kernel=11, lp_cutoff=0.30, causal=False,
            w_lf=1.0, w_hf=0.4, w_vel=0
        ).to(self.device)


        self.stft3d = STFTBand3D(
            n_fft=16, hop=4, win=16,
            low_max_bin=3, high_min_bin=5,
            w_low=0.5, w_high=0.5
        ).to(self.device)

        self.contact_criterion = nn.BCEWithLogitsLoss()

        if self.pretrain:
            self.model.load_state_dict(torch.load(os.path.join(self.model_dir, 'model.pkl')))
            logging.info("pretrained model loaded")

    def build_optimizer(self):
        logging.info("Preparing optimizer...")

        self.optimizer = dadaptation.DAdaptAdam(self.model.parameters(), lr=1.0, decouple=True, weight_decay=1.0) # use AdamW
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=self.config['train']['lr_decay'])

        if self.pretrain:
            self.optimizer.load_state_dict(torch.load(os.path.join(self.model_dir + 'optimizer.pkl')))
            logging.info("optimizer loaded")

    def run(self, mode="test"):
        logging.info(f"Testing model with mode: {mode} ...")

        self.teacher_forcing_ratio = 0
        self.model.eval()

        render_data = []

        # for losses (in test and validation mode)
        epoch_loss = 0
        steps_per_epoch = len(self.dataloader[mode])

        # data recording (in test mode)
        select_idx = 0
        if mode == "test":
            batch = self.config['test']['batch_size']
            select_idx = random.randint(0, batch-1)
            print(f"selected index: {select_idx}")

        for iterations, sampled_batch in enumerate(tqdm(self.dataloader[mode])):
            with torch.no_grad():
                input_seq = sampled_batch['input_seq'].to(self.device)
                input_img = sampled_batch['imgs']#.to(self.device)
                input_pc = sampled_batch['scene_points']#.to(self.device)
    
                input_seq = (input_seq - self.mean.float()) / self.std.float()

                output_tuple = self.model(input_seq, input_img=input_img, input_pc = input_pc) # ee, contact, output [256, 40, 31]

                results = self.loss_func(output_tuple=output_tuple, gt_tuple=sampled_batch, \
                                        get_results=(mode == "test"), \
                                        get_loss=True)

                if results is not None:
                    output_root, output_joint_rot = results
                    tgt_root = sampled_batch['global_p'][...,0,:]
                    tgt_rotations = sampled_batch['local_rot']

                    rd = RenderData(gt_root=tgt_root[select_idx].detach().cpu().numpy(), \
                    gt_rot=tgt_rotations[select_idx].detach().cpu().numpy(), \
                    output_root=output_root[select_idx].detach().cpu().numpy(),\
                    output_rot=output_joint_rot[select_idx].detach().cpu().numpy())

                    start_T = sampled_batch['head_start'].detach().cpu().numpy()
                    rd.convert_to_matrix(start_T=start_T[select_idx])

                    # reshape
                    other_est = {}
                    other_est['ee'] = output_tuple[0][select_idx].detach().cpu().numpy()

                    render_data.append(rd)

            if mode in ["test", "validation"]:
                epoch_loss += self.loss_total.item()

        if mode in ["test", "validation"]:			
            epoch_loss /= steps_per_epoch
            logging.info(
                f"Test mode: {mode} | "
                f"{mode} loss: {epoch_loss}"
            )

        if mode == "test":
            write_result_pkl(render_data_list=render_data, save_dir=os.path.join(self.directory, f"testset_{select_idx}/"))

    def train(self):
        self.writer = SummaryWriter(self.log_dir)
        logging.info("Training model...")
        torch.autograd.set_detect_anomaly(True)
        
        self.w_uncert = 0.
        self.eval()
        self.run(mode="validation") 
        self.loss_total_min = 100000
        self.train_epoch = 0
        for epoch in range(self.config['train']['num_epoch']):
            self.train_epoch = epoch
            epoch_loss = 0
            self.model.train()
            self.teacher_forcing_ratio = 0.0 if (self.pretrain or epoch >= change_mode_epoch) else float((change_mode_epoch-epoch)/change_mode_epoch)

            logging.info(
            f"Running epoch {epoch} | "
            f"teacher_forcing_ratio={self.teacher_forcing_ratio}"
            )

            steps_per_epoch = len(self.dataloader['train'])
            for iterations, sampled_batch in enumerate(tqdm(self.dataloader['train'])):
                input_seq = sampled_batch['input_seq'].to(self.device) #[256, 40, 31]
                input_img = sampled_batch['imgs']#.to(self.device)
                input_pc = sampled_batch['scene_points']#.to(self.device)

                # env = self.encode_scene(input_pc) if self.scene_encoder else None
                
                input_seq = (input_seq - self.mean.float()) / self.std.float()
                # add noise
                input_seq = input_seq + 0.01 * torch.randn(input_seq.shape).to(self.device)
                output_tuple = self.model(input_seq, input_img = input_img, input_pc = input_pc) # hand (mid), foot, final_output (body)

                results = self.loss_func(output_tuple=output_tuple, gt_tuple=sampled_batch, get_results=False, get_loss=True)
                
                self.optimize()

                if iterations % 5 == 0:
                    self.update(epoch, steps_per_epoch, iterations)
                epoch_loss += self.loss_total.item()
            
            epoch_loss /= steps_per_epoch
            self.run(mode="validation") 
            if epoch % self.num_epoch_eval == 0:
                self.eval()
            self.save(epoch_loss, epoch)

    def get_loss(self, output_tuple, gt_tuple, get_results=False, get_loss=True, is_eval=False):
        if self.use_uncertainty:
            mid_ee, contact_output, output_seq, sampled_output, pred_logvar = output_tuple
        else:
            mid_ee, contact_output, output_seq = output_tuple

        batch, seq_len, _ = output_seq.shape

        mid_seq = gt_tuple['mid_seq'].to(self.device)
        tgt_seq = gt_tuple['tgt_seq'].to(self.device) # [batch, seq_len, dim]
        global_pos = gt_tuple['global_p'].to(self.device)
        root = gt_tuple['root'].to(self.device)
        gt_contact_label = gt_tuple['contact_label'].to(self.device)

        # normalize root (provide answer root pos by teacher forcing ration)
        output_root = output_seq[...,:3]
        output_joint_rot = output_seq[...,3:]
        output_joint_rot = output_joint_rot.reshape(batch, seq_len, -1, 6)
        target_joint_rot = tgt_seq[...,3:].reshape(batch, seq_len, -1, 6)

        idx_teacher = []
        idx_else = []
        for i in range(seq_len):
            if random.random() < self.teacher_forcing_ratio:
                idx_teacher.append(i)
            else:
                idx_else.append(i)

        root_view = root.reshape(batch, seq_len, 3)
        root_mix = output_root.clone()
        output_joint_rot_root_mix = output_joint_rot.clone()

        if len(idx_teacher) > 0:
            root_mix[:,idx_teacher,:] = root_view[:,idx_teacher,:]		
            output_joint_rot_root_mix[:,idx_teacher, 0, ...] = target_joint_rot[:, idx_teacher, 0, ...]

        output_joint_rot_root_mix = transforms.rotation_6d_to_matrix(output_joint_rot_root_mix)

        if not get_loss:
            return (output_root, transforms.rotation_6d_to_matrix(output_joint_rot)) if get_results else None

        # compare pos & ee 
        if self.skel_offset.shape[0] != batch:
            output_pos_mat = rot_matrix_fk_tensor(output_joint_rot_root_mix, root_mix, self.skel_offset[0:batch], self.skel_parent)
        else:
            output_pos_mat = rot_matrix_fk_tensor(output_joint_rot_root_mix, root_mix, self.skel_offset, self.skel_parent)

        if is_eval:
            result_dict = {}
            result_dict['pred_pos'] = output_pos_mat.clone()
            result_dict['pred_rot'] = output_joint_rot.clone()
            result_dict['gt_pos'] = global_pos.clone()
            result_dict['gt_rot'] = target_joint_rot.clone()
            return result_dict

        root_diff = torch.abs(output_seq[...,:3] - tgt_seq[...,:3]) / self.x_std[...,0,:]

        # add root rot
        root_rot_diff = self.criterion(output_seq[...,3:9], tgt_seq[...,3:9])
        self.root_mean_loss = torch.mean(root_diff) + root_rot_diff * 0.5
        self.rotation_mse_loss = self.criterion(output_seq[...,3:], tgt_seq[...,3:])					

        # pos related loss
        pos_diff = torch.abs(global_pos - output_pos_mat) / self.x_std
        ee_diff = pos_diff[...,self.ee_idx+self.leg_idx,:]
        foot_diff = pos_diff[..., self.foot_idx,:] # output of the final layer
        
        self.pos_mean_loss = torch.mean(pos_diff)
        self.ee_mean_loss = torch.mean(ee_diff)
        self.foot_pos_loss = torch.mean(foot_diff)

        # TODO: balance loss weight; check decompostition loss
        if self.use_freq_time_decom_loss:
            # ===== 3D 分带损（替代 pos_mean_loss + foot_vel_loss）=====
            # self.loss_scale3d, aux_scale = self.scale3d(output_pos_mat, global_pos)
            # 构造与原来一致的 std（如果你已经有 self.x_std，形状 [B,1,J,3] 或 [1,1,J,3]）
            x_std_for_loss = self.x_std  # 保持与你原pos_diff一致
            # 关节权重（可选）：脚/末端权重更大，例如脚=2.0，其余=1.0
            joint_w = torch.ones((22,), device=self.device)
            joint_w[self.foot_idx] = 2.0

            # warmup（比如 0→1 在线性 10 epoch 内）
            alpha = get_warmup_weight(self.train_epoch, start=0.0, end=1.0, \
                                         warmup_steps=self.config['train']['num_epoch'], start_step=change_loss_epoch)

            self.loss_scale3d, aux_scale = self.scale3d(
                output_pos_mat, global_pos,
                x_std=x_std_for_loss, joint_weight=joint_w, warmup_alpha=alpha
            )

            # ===== STFT 分带损（小权重辅助）=====
            self.loss_spec3d, aux_spec = self.stft3d(output_pos_mat, global_pos)

        # 标准 Gaussian NLL
        if self.use_uncertainty:
            uncertainty_nll_loss = 0.5 * torch.exp(-pred_logvar) * (tgt_seq - output_seq)**2 + 0.5 * pred_logvar
            self.uncertainty_loss = uncertainty_nll_loss.mean()
            
            # EnvPoser 写法
            # eps = 1e-8
            # delta = torch.exp(0.5 * pred_logvar)
            # scaled_err = (tgt_seq - output_seq) / (delta + eps)
            # data_term = torch.linalg.vector_norm(scaled_err, ord=2, dim=-1)  # L2 (非平方)
            # reg_term  = torch.log(torch.linalg.vector_norm(delta, ord=2, dim=-1) + eps)
            # self.uncertainty_loss = (data_term + reg_term).mean()
        
        self.foot_vel_loss = None
        if self.train_epoch > change_mode_epoch:
            vel = global_pos[...,1:,:,:] - global_pos[...,:-1,:,:]
            output_vel = output_pos_mat[...,1:,:,:] - output_pos_mat[...,:-1,:,:]
            vel_diff = torch.abs(output_vel - vel)
            vel_diff = torch.abs(output_vel - vel) / self.x_std
            self.foot_vel_loss = torch.mean(vel_diff[...,self.foot_idx, :])

        # mid ee 
        mid_ee_reshape = mid_ee.reshape(batch, seq_len, -1, 3)
        mid_seq_reshape = mid_seq.reshape(batch, seq_len, -1, 3)
        mid_pos_diff = torch.abs(mid_ee_reshape - mid_seq_reshape) / self.x_std[...,self.mid_ee_idx,:]
        self.mid_mean_loss = torch.mean(mid_pos_diff)

        mid_est_diff = torch.abs(output_pos_mat[...,self.mid_ee_idx,:] - mid_ee_reshape) / self.x_std[...,self.mid_ee_idx,:]
        self.est_loss = torch.mean(mid_est_diff)

        # contact classifier loss
        self.contact_loss = self.contact_criterion(contact_output, gt_contact_label)

        if self.use_freq_time_decom_loss and self.train_epoch > change_loss_epoch:
            self.loss_total = \
                            self.config['train']['loss_foot_weight'] * self.foot_pos_loss + \
                            self.config['train']['loss_ee_weight'] * self.ee_mean_loss + \
                            self.config['train']['loss_mid_weight'] * self.mid_mean_loss + \
                            self.config['train']['loss_est_weight'] * self.est_loss + \
                            1.2 * self.config['train']['loss_quat_weight'] * self.rotation_mse_loss + \
                            1.2 * self.config['train']['loss_root_weight'] * self.root_mean_loss + \
                            self.config['train']['loss_pos_weight'] * self.loss_scale3d + \
                            self.config['train']['loss_spec3d_weight'] * self.loss_spec3d + \
                            self.config['train']['loss_contact_weight'] * self.contact_loss
                            # 0.6 * self.config['train']['loss_quat_weight'] * self.rotation_mse_loss + \
                            # 0.6 * self.config['train']['loss_root_weight'] * self.root_mean_loss + \
        else:
            self.loss_total = self.config['train']['loss_pos_weight'] * self.pos_mean_loss + \
                            self.config['train']['loss_foot_weight'] * self.foot_pos_loss + \
                            self.config['train']['loss_ee_weight'] * self.ee_mean_loss + \
                            self.config['train']['loss_mid_weight'] * self.mid_mean_loss + \
                            1.5 * self.config['train']['loss_quat_weight'] * self.rotation_mse_loss + \
                            self.config['train']['loss_root_weight'] * self.root_mean_loss + \
                            self.config['train']['loss_est_weight'] * self.est_loss + \
                            self.config['train']['loss_contact_weight'] * self.contact_loss
                            # self.config['train']['loss_quat_weight'] * self.rotation_mse_loss + \
        
        if self.use_uncertainty:
            self.w_uncert = get_warmup_weight(self.train_epoch, start=0.0, end=self.config['train']['loss_uncertainty_weight'], \
                                         warmup_steps=self.config['train']['num_epoch'], start_step=self.config['train']['warmup_start_epoch'])
            self.loss_total += self.w_uncert * self.uncertainty_loss
        
        if self.use_swt_ada:
            self.wavelet_arch_loss = self.model.last_wavelet_arch_loss or torch.zeros((), device=self.loss_total.device)
            self.loss_total += self.config['train']['loss_arch_weight'] * self.wavelet_arch_loss

        if self.foot_vel_loss is not None:
            self.loss_total += self.config['train']['loss_vel_weight'] * self.foot_vel_loss

        return (output_root, transforms.rotation_6d_to_matrix(output_joint_rot)) if get_results else None

    def get_loss_eval(self, output_tuple, gt_tuple, get_results=False, get_loss=True, is_eval=False):
        if self.use_uncertainty:
            mid_ee, contact_output, output_seq, sampled_output, pred_logvar = output_tuple
            # mid_ee, contact_output, _, output_seq, pred_logvar = output_tuple           # Use random sampled results
        else:
            mid_ee, contact_output, output_seq = output_tuple

        batch, seq_len, _ = output_seq.shape

        mid_seq = gt_tuple['mid_seq'].to(self.device)
        tgt_seq = gt_tuple['tgt_seq'].to(self.device) # [batch, seq_len, dim]
        global_pos = gt_tuple['global_p'].to(self.device)
        root = gt_tuple['root'].to(self.device)
        #gt_contact_label = gt_tuple['contact_label'].to(self.device)

        output_root = output_seq[...,:3]
        output_joint_rot = output_seq[...,3:]
        output_joint_rot = output_joint_rot.reshape(batch, seq_len, -1, 6)
        target_joint_rot = tgt_seq[...,3:].reshape(batch, seq_len, -1, 6)

        output_joint_rotmat = transforms.rotation_6d_to_matrix(output_joint_rot)

        if not get_loss:
            return (output_root, transforms.rotation_6d_to_matrix(output_joint_rot)) if get_results else None

        # compare pos & ee 
        if self.skel_offset.shape[0] != batch:
            output_pos_mat = rot_matrix_fk_tensor(output_joint_rotmat, output_root, self.skel_offset[0:batch], self.skel_parent)
        else:
            output_pos_mat = rot_matrix_fk_tensor(output_joint_rotmat, output_root, self.skel_offset, self.skel_parent)

        if is_eval:
            result_dict = {}
            result_dict['pred_pos'] = output_pos_mat.clone().detach()
            result_dict['pred_rot'] = output_joint_rot.clone().detach()
            result_dict['gt_pos'] = global_pos.clone().detach()
            result_dict['gt_rot'] = target_joint_rot.clone().detach()
   
            return result_dict
        
        change_mode_epoch = 40
        root_diff = torch.abs(output_seq[...,:3] - tgt_seq[...,:3]) / self.x_std[...,0,:]

        # add root rot
        root_rot_diff = self.criterion(output_seq[...,3:9], tgt_seq[...,3:9])
        self.root_mean_loss = torch.mean(root_diff) + root_rot_diff * 0.5
        self.rotation_mse_loss = self.criterion(output_seq[...,3:], tgt_seq[...,3:])					

        # pos related loss
        pos_diff = torch.abs(global_pos - output_pos_mat) / self.x_std
        ee_diff = pos_diff[...,self.ee_idx+self.leg_idx,:]
        foot_diff = pos_diff[..., self.foot_idx,:] # output of the final layer
        
        self.pos_mean_loss = torch.mean(pos_diff)
        self.ee_mean_loss = torch.mean(ee_diff)
        self.foot_pos_loss = torch.mean(foot_diff)

        # 标准 Gaussian NLL
        if self.use_uncertainty:
            uncertainty_nll_loss = 0.5 * torch.exp(-pred_logvar) * (tgt_seq - output_seq)**2 + 0.5 * pred_logvar
            self.uncertainty_loss = uncertainty_nll_loss.mean()
            
            # EnvPoser 写法: BAD
            # eps = 1e-8
            # delta = torch.exp(0.5 * pred_logvar)
            # scaled_err = (tgt_seq - output_seq) / (delta + eps)
            # data_term = torch.linalg.vector_norm(scaled_err, ord=2, dim=-1)  # L2 (非平方)
            # reg_term  = torch.log(torch.linalg.vector_norm(delta, ord=2, dim=-1) + eps)
            # self.uncertainty_loss = (data_term + reg_term).mean()

        
        self.foot_vel_loss = None
        if self.train_epoch > change_mode_epoch:
            vel = global_pos[...,1:,:,:] - global_pos[...,:-1,:,:]
            output_vel = output_pos_mat[...,1:,:,:] - output_pos_mat[...,:-1,:,:]
            vel_diff = torch.abs(output_vel - vel)
            vel_diff = torch.abs(output_vel - vel) / self.x_std
            self.foot_vel_loss = torch.mean(vel_diff[...,self.foot_idx, :])

        # mid ee 
        mid_ee_reshape = mid_ee.reshape(batch, seq_len, -1, 3)
        mid_seq_reshape = mid_seq.reshape(batch, seq_len, -1, 3)
        mid_pos_diff = torch.abs(mid_ee_reshape - mid_seq_reshape) / self.x_std[...,self.mid_ee_idx,:]
        self.mid_mean_loss = torch.mean(mid_pos_diff)

        # est loss 
        mid_est_diff = torch.abs(output_pos_mat[...,self.mid_ee_idx,:] - mid_ee_reshape) / self.x_std[...,self.mid_ee_idx,:]

        self.est_loss = torch.mean(mid_est_diff)

        # contact classifier loss
        self.contact_loss = self.contact_criterion(contact_output, gt_contact_label)

        if self.use_freq_time_decom_loss:
            self.loss_total = \
                            self.config['train']['loss_foot_weight'] * self.foot_pos_loss + \
                            self.config['train']['loss_ee_weight'] * self.ee_mean_loss + \
                            self.config['train']['loss_mid_weight'] * self.mid_mean_loss + \
                            self.config['train']['loss_est_weight'] * self.est_loss + \
                            self.config['train']['loss_contact_weight'] * self.contact_loss + \
                            0.5 * self.config['train']['loss_quat_weight'] * self.rotation_mse_loss + \
                            0.6 * self.config['train']['loss_root_weight'] * self.root_mean_loss + \
                            self.config['train']['loss_pos_weight'] * self.loss_scale3d + \
                            self.config['train']['loss_spec3d_weight'] * self.loss_spec3d                            # STFT decomposition loss weight 0.3
        else:
            self.loss_total = self.config['train']['loss_pos_weight'] * self.pos_mean_loss + \
                            self.config['train']['loss_foot_weight'] * self.foot_pos_loss + \
                            self.config['train']['loss_ee_weight'] * self.ee_mean_loss + \
                            self.config['train']['loss_mid_weight'] * self.mid_mean_loss + \
                            self.config['train']['loss_quat_weight'] * self.rotation_mse_loss + \
                            self.config['train']['loss_root_weight'] * self.root_mean_loss + \
                            self.config['train']['loss_est_weight'] * self.est_loss + \
                            self.config['train']['loss_contact_weight'] * self.contact_loss
        
        if self.use_uncertainty:
            self.loss_total += self.w_uncert * self.uncertainty_loss
        

        if self.foot_vel_loss is not None:
            self.loss_total += self.config['train']['loss_vel_weight'] * self.foot_vel_loss

        return (output_root, transforms.rotation_6d_to_matrix(output_joint_rot)) if get_results else None
            
    def optimize(self):
        self.optimizer.zero_grad()
        self.loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
    
    def update(self, epoch, steps_per_epoch, idx):
        self.writer.add_scalar('loss_pos', self.pos_mean_loss.item(), global_step = epoch * steps_per_epoch + idx)
        self.writer.add_scalar('loss_ee', self.ee_mean_loss.item(), global_step = epoch * steps_per_epoch + idx)
        self.writer.add_scalar('loss_foot', self.foot_pos_loss.item(), global_step = epoch * steps_per_epoch + idx)
        self.writer.add_scalar('loss_mid', self.mid_mean_loss.item(), global_step = epoch * steps_per_epoch + idx)
        self.writer.add_scalar('loss_quat', self.rotation_mse_loss.item(), global_step = epoch * steps_per_epoch + idx)
        self.writer.add_scalar('loss_root', self.root_mean_loss.item(), global_step = epoch * steps_per_epoch + idx)
        self.writer.add_scalar('loss_est', self.est_loss.item(), global_step = epoch * steps_per_epoch + idx)
        # self.writer.add_scalar('loss_contact', self.contact_loss.item(), global_step = epoch * steps_per_epoch + idx)
        if self.use_uncertainty:
            self.writer.add_scalar('loss_uncertainty', self.uncertainty_loss.item(), global_step = epoch * steps_per_epoch + idx)
        if self.use_swt_ada:
            self.writer.add_scalar('loss_swt_ada', self.wavelet_arch_loss.item(), global_step = epoch * steps_per_epoch + idx)
        if self.use_freq_time_decom_loss:
            self.writer.add_scalar('loss_scale3d', self.loss_scale3d.item(), global_step = epoch * steps_per_epoch + idx)
            self.writer.add_scalar('loss_spec3d', self.loss_spec3d.item(), global_step = epoch * steps_per_epoch + idx)

        self.writer.add_scalar('loss_total', self.loss_total.item(), global_step = epoch * steps_per_epoch + idx)
        if self.foot_vel_loss is not None:
            self.writer.add_scalar('loss_foot_vel', self.foot_vel_loss.item(), global_step = epoch * steps_per_epoch + idx)


    def save(self, epoch_loss, epoch):
        if (epoch % self.save_frequency == self.save_frequency-1) or epoch_loss < self.loss_total_min:
            logging.info("Saving model")
            torch.save(self.model.state_dict(), self.model_dir + 'model.pkl')
            torch.save(self.optimizer.state_dict(), self.model_dir + 'optimizer.pkl')
            if epoch_loss < self.loss_total_min:
                self.loss_total_min = epoch_loss
        
        logging.info(f"Current Epoch: {epoch} | "
                     f"Current Loss: {epoch_loss} | "
                     f"Best Loss: {self.loss_total_min}")
  
    def eval(self):
        logging.info(f"Eval with testset ...")
        self.teacher_forcing_ratio = 0
        self.model.eval()

        self.eval_log = {}
        for metric in self.eval_metric:
            self.eval_log[metric] = []
        
        count = 0
        filenames = []
        self.eval_log_by_filename = {}

        render_result_dict = {}
        render_result_dict['fps'] = 30.0
        render_result_dict['seq_len'] = []
        render_result_dict['idx'] = []
        render_result_dict['motion'] = []

        for i, filepath in tqdm(enumerate(self.eval_files_gimo)):
            with open(filepath, "rb") as file:
                file_dict = pickle.load(file)
                file_dict.update({"filename": filepath})
            if i == 0:
                eval_log_per_file = self.run_per_file(file_dict=file_dict, save_name = f'vis/gimo_eval_{i:03d}.ply')
            else:
                eval_log_per_file = self.run_per_file(file_dict=file_dict, save_name = None)
            if eval_log_per_file['filename'] in self.eval_log_by_filename:
                embed()
            self.eval_log_by_filename[eval_log_per_file['filename']] = eval_log_per_file
            for metric in self.eval_metric:
                self.eval_log[metric].append(eval_log_per_file[metric])
        
        print(f"Done.")
        logging.info(f"-----------------------GIMO EVAL RESULT-----------------------------------------------")
        for metric in self.eval_metric:
            if 'jitter' == metric:
                continue
            print(f"metric: {metric} value: {np.mean(np.array(self.eval_log[metric])) * metrics_coeffs[metric]:.2f}")
        print(f"metric: jitter value: {np.mean(np.array(self.eval_log['pred_jitter'])) / np.mean(np.array(self.eval_log['gt_jitter'])):.2f}")
        logging.info(f"--------------------------------------------------------------------------------------")
  
        for i, filepath in tqdm(enumerate(self.eval_files_egobody)):
            with open(filepath, "rb") as file:
                file_dict = pickle.load(file)
                file_dict.update({"filename": filepath})
            if i == 0:
                eval_log_per_file = self.run_per_file(file_dict=file_dict, save_name = f'vis/egobody_eval_{i:03d}.ply')
            else:
                eval_log_per_file = self.run_per_file(file_dict=file_dict, save_name = None)
            if eval_log_per_file['filename'] in self.eval_log_by_filename:
                embed()
            self.eval_log_by_filename[eval_log_per_file['filename']] = eval_log_per_file
            for metric in self.eval_metric:
                self.eval_log[metric].append(eval_log_per_file[metric])
        
        print(f"Done.")
        logging.info(f"-----------------------Egobody EVAL RESULT--------------------------------------------")
        for metric in self.eval_metric:
            if 'jitter' == metric:
                continue
            print(f"metric: {metric} value: {np.mean(np.array(self.eval_log[metric])) * metrics_coeffs[metric]:.2f}")
        print(f"metric: jitter value: {np.mean(np.array(self.eval_log['pred_jitter'])) / np.mean(np.array(self.eval_log['gt_jitter'])):.2f}")
        logging.info(f"--------------------------------------------------------------------------------------")
  
    def run_per_file(self, file_dict, save_name = None):
        sampled_batch = file_dict
        total_length = sampled_batch['total_length']
        # create placeholder for pred pos, pred rot, gt pos and gt rot
        predicted_position = torch.zeros(size=(total_length, motion_constants.NUM_JOINTS, 3))
        predicted_rot = torch.zeros(size=(total_length, motion_constants.NUM_JOINTS, 3, 3))
        gt_position = torch.zeros(size=(total_length, motion_constants.NUM_JOINTS, 3))
        gt_rot = torch.zeros(size=(total_length, motion_constants.NUM_JOINTS, 3, 3))

        input_seq = sampled_batch['input_seq'].to(self.device)
        input_img = None
        input_pc = sampled_batch['scene_points'].to(self.device)
  
        # norm_input
        input_seq = (input_seq - self.mean) / self.std 
  
        output_tuple = self.model(input_seq.float(), input_img = input_img, input_pc = input_pc)
  
        results = self.get_loss_eval(output_tuple=output_tuple, gt_tuple=sampled_batch, \
                                    get_results=False, \
                                    get_loss=True, \
                                    is_eval=True) 
        
        start_T = sampled_batch['head_start'].to(self.device) # Start pos

        # get pred into world coord
        pred_pos_to_world = start_T[...,:3,:3].to(self.device) @ results['pred_pos'].unsqueeze(-1)
        pred_pos_to_world = pred_pos_to_world[...,0] + start_T[...,:3,3]
        pred_rotmat = transforms.rotation_6d_to_matrix(results['pred_rot'])
        pred_rotmat[...,0:1,:,:] = start_T[...,:3,:3] @ pred_rotmat[...,0:1,:,:]

        # get gt into world coord
        gt_pos_to_world = start_T[...,:3,:3].to(self.device) @ results['gt_pos'].unsqueeze(-1)
        gt_pos_to_world = gt_pos_to_world[...,0] + start_T[...,:3,3]
        gt_rotmat = transforms.rotation_6d_to_matrix(results['gt_rot'])
        gt_rotmat[...,0:1,:,:] = start_T[...,:3,:3] @ gt_rotmat[...,0:1,:,:]
      
        if save_name != None:
            save_two_pointclouds_with_colors(predicted_position.clone().detach().reshape((-1,22,3)), 
                                             gt_position.clone().detach().reshape((-1,22,3)), 
                                             save_name,
                                            )

        # into single seq
        batch, seq_len, J, _ = pred_pos_to_world.shape

        for idx, info in enumerate(sampled_batch['info']):
            start_frame = int(info['start_end'][0])
            predicted_position[start_frame:start_frame+seq_len] = pred_pos_to_world[idx]
            predicted_rot[start_frame:start_frame+seq_len] = pred_rotmat[idx]
            gt_position[start_frame:start_frame+seq_len] = gt_pos_to_world[idx]
            gt_rot[start_frame:start_frame+seq_len] = gt_rotmat[idx]


        predicted_angle_np = conversions.R2A(predicted_rot.cpu().numpy())
        predicted_angle = torch.from_numpy(predicted_angle_np).cuda().float()
        predicted_root_angle = predicted_angle[...,0,:] 

        gt_angle_np = conversions.R2A(gt_rot.cpu().numpy()) 
        gt_angle = torch.from_numpy(gt_angle_np).cuda().float() 
        gt_root_angle = gt_angle[...,0,:] 
                
        # after running iterations get numbers
        upper_index = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        lower_index = [0, 1, 2, 4, 5, 7, 8] # 10,11 is not considered in imus. (why? TIP does not have ankle joints)
        hand_index = [20, 21]
        foot_index = [7, 8]
        eval_log = {}
        for metric in self.eval_metric:
            eval_metric = get_metric_function(metric)(
                    predicted_position,
                    predicted_angle,
                    predicted_root_angle,
                    gt_position,
                    gt_angle,
                    gt_root_angle,
                    upper_index,
                    lower_index,
                    hand_index,
                    foot_index,
                    fps=motion_constants.FPS,
                    root_rel=True
                ).cpu().numpy()
            eval_log[metric] = eval_metric 
        
        # add filename
        parts = sampled_batch['filename'].split('/')
        filename = '/'.join(parts[-1:])
        eval_log['filename'] = filename
        torch.cuda.empty_cache()

        return eval_log

    def eval_vis(self):
        logging.info(f"Eval with testset ...")
        self.teacher_forcing_ratio = 0
        self.model.eval()

        self.eval_log = {}
        for metric in self.eval_metric:
            self.eval_log[metric] = []
        
        count = 0
        filenames = []
        self.eval_log_by_filename = {}

        render_result_dict = {}
        render_result_dict['fps'] = 30.0
        render_result_dict['seq_len'] = []
        render_result_dict['idx'] = []
        render_result_dict['motion'] = []

        for i, filepath in tqdm(enumerate(self.eval_files_gimo)):
            with open(filepath, "rb") as file:
                file_dict = pickle.load(file)
                file_dict.update({"filename": filepath})
            eval_log_per_file = self.run_per_file_vis(file_dict=file_dict, save_name = f'vis/gimo_eval_{i:03d}.ply', vis_interval=30)
            if eval_log_per_file['filename'] in self.eval_log_by_filename:
                embed()
            self.eval_log_by_filename[eval_log_per_file['filename']] = eval_log_per_file
            for metric in self.eval_metric:
                self.eval_log[metric].append(eval_log_per_file[metric])
        
        print(f"Done.")
        logging.info(f"-----------------------GIMO EVAL RESULT-----------------------------------------------")
        for metric in self.eval_metric:
            if 'jitter' == metric:
                continue
            print(f"metric: {metric} value: {np.mean(np.array(self.eval_log[metric])) * metrics_coeffs[metric]:.2f}")
        print(f"metric: jitter value: {np.mean(np.array(self.eval_log['pred_jitter'])) / np.mean(np.array(self.eval_log['gt_jitter'])):.2f}")
        logging.info(f"--------------------------------------------------------------------------------------")
  
        for i, filepath in tqdm(enumerate(self.eval_files_egobody)):
            with open(filepath, "rb") as file:
                file_dict = pickle.load(file)
                file_dict.update({"filename": filepath})
            eval_log_per_file = self.run_per_file_vis(file_dict=file_dict, save_name = f'vis/egobody_eval_{i:03d}.ply', vis_interval=90)
            if eval_log_per_file['filename'] in self.eval_log_by_filename:
                embed()
            self.eval_log_by_filename[eval_log_per_file['filename']] = eval_log_per_file
            for metric in self.eval_metric:
                self.eval_log[metric].append(eval_log_per_file[metric])
        
        print(f"Done.")
        logging.info(f"-----------------------Egobody EVAL RESULT--------------------------------------------")
        for metric in self.eval_metric:
            if 'jitter' == metric:
                continue
            print(f"metric: {metric} value: {np.mean(np.array(self.eval_log[metric])) * metrics_coeffs[metric]:.2f}")
        print(f"metric: jitter value: {np.mean(np.array(self.eval_log['pred_jitter'])) / np.mean(np.array(self.eval_log['gt_jitter'])):.2f}")
        logging.info(f"--------------------------------------------------------------------------------------")
  
    def run_per_file_vis(self, file_dict, save_name = None, vis_interval=30):
        sampled_batch = file_dict
        total_length = sampled_batch['total_length']
        # create placeholder for pred pos, pred rot, gt pos and gt rot
        predicted_position = torch.zeros(size=(total_length, motion_constants.NUM_JOINTS, 3))
        predicted_rot = torch.zeros(size=(total_length, motion_constants.NUM_JOINTS, 3, 3))
        gt_position = torch.zeros(size=(total_length, motion_constants.NUM_JOINTS, 3))
        gt_rot = torch.zeros(size=(total_length, motion_constants.NUM_JOINTS, 3, 3))

        input_seq = sampled_batch['input_seq'].to(self.device)
        input_img = None
        input_pc = sampled_batch['scene_points'].to(self.device)

  
        # norm_input
        input_seq = (input_seq - self.mean) / self.std 

        output_tuple = self.model(input_seq.float(), input_img = input_img, input_pc = input_pc)

        scene_vertices = sampled_batch['mesh_vertices'].to(self.device)
        scene_faces = sampled_batch['mesh_faces'].to(self.device)
        
        results = self.get_loss_eval(output_tuple=output_tuple, gt_tuple=sampled_batch, \
                                    get_results=False, \
                                    get_loss=True, \
                                    is_eval=True) 
        
        start_T = sampled_batch['head_start'].to(self.device) # Start pos

        # get pred into world coord
        pred_pos_to_world = start_T[...,:3,:3].to(self.device) @ results['pred_pos'].unsqueeze(-1)
        pred_pos_to_world = pred_pos_to_world[...,0] + start_T[...,:3,3]
        pred_rotmat = transforms.rotation_6d_to_matrix(results['pred_rot'])
        pred_rotmat[...,0:1,:,:] = start_T[...,:3,:3] @ pred_rotmat[...,0:1,:,:]

        # get gt into world coord
        gt_pos_to_world = start_T[...,:3,:3].to(self.device) @ results['gt_pos'].unsqueeze(-1)
        gt_pos_to_world = gt_pos_to_world[...,0] + start_T[...,:3,3]
        gt_rotmat = transforms.rotation_6d_to_matrix(results['gt_rot'])
        gt_rotmat[...,0:1,:,:] = start_T[...,:3,:3] @ gt_rotmat[...,0:1,:,:]

        pc_to_world = start_T[...,:3,:3].to(self.device) @ input_pc.unsqueeze(-1).unsqueeze(2)
        pc_to_world = pc_to_world[..., 0] + start_T[...,:3,3]

        scene_vertices_to_world = start_T[...,:3,:3].to(self.device) @ scene_vertices.unsqueeze(-1).unsqueeze(2)
        scene_vertices_to_world = scene_vertices_to_world[..., 0] + start_T[...,:3,3]

        # into single seq
        batch, seq_len, J, _ = pred_pos_to_world.shape

        for idx, info in enumerate(sampled_batch['info']):
            start_frame = int(info['start_end'][0])
            predicted_position[start_frame:start_frame+seq_len] = pred_pos_to_world[idx]
            predicted_rot[start_frame:start_frame+seq_len] = pred_rotmat[idx]
            gt_position[start_frame:start_frame+seq_len] = gt_pos_to_world[idx]
            gt_rot[start_frame:start_frame+seq_len] = gt_rotmat[idx]

        # if save_name != None:
        #     save_two_pointclouds_with_colors(predicted_position.clone().detach().reshape((-1,22,3)), gt_position.clone().detach().reshape((-1,22,3)), pc_to_world.detach().reshape(-1, 3), save_name)

        pred_aa_24 = torch.Tensor(conversions.R2A(predicted_rot))
        pred_global_orient = pred_aa_24[:, 0]                                        # [B,3]
        pred_body_pose = pred_aa_24[:, 1:].reshape(total_length, 21*3)

        gt_aa_24 = torch.Tensor(conversions.R2A(gt_rot))
        gt_global_orient = gt_aa_24[:, 0]                                        # [B,3]
        gt_body_pose = gt_aa_24[:, 1:].reshape(total_length, 21*3)

        # Betas
        betas = torch.zeros(total_length, 10)
        zero_pose = torch.zeros(total_length, 10)

        # Build SMPL layer
        pred_output = self.smplx_model(
            global_orient=pred_global_orient,   # [B,3]
            body_pose=pred_body_pose,           # [B,69]
            betas=betas,                   # [B,10]
            left_hand_pose=torch.zeros((total_length, 45)), 
            right_hand_pose=torch.zeros((total_length, 45)),
            jaw_pose=torch.zeros((total_length, 3)), 
            leye_pose=torch.zeros((total_length, 3)),
            reye_pose=torch.zeros((total_length, 3)),
            expression=torch.zeros((total_length, 10)),
        )
        pred_vertices = pred_output.vertices.detach() - pred_output.joints.detach()[:, 0:1] + predicted_position[:, 0:1]
        smplx_faces = self.smplx_model.faces
        save_two_meshes_with_colors(pred_vertices[::vis_interval], smplx_faces,
                                    scene_vertices_to_world[0].detach().reshape(-1, 3),
                                    scene_faces[0],
                                    smpl_color = [1.0, 0.75, 0.8],
                                    filename=os.path.splitext(save_name)[0] + "_pred" + os.path.splitext(save_name)[1])
        gt_output = self.smplx_model(
            global_orient=gt_global_orient,   # [B,3]
            body_pose=gt_body_pose,           # [B,69]
            betas=betas,                   # [B,10]
            left_hand_pose=torch.zeros((total_length, 45)), 
            right_hand_pose=torch.zeros((total_length, 45)),
            jaw_pose=torch.zeros((total_length, 3)), 
            leye_pose=torch.zeros((total_length, 3)),
            reye_pose=torch.zeros((total_length, 3)),
            expression=torch.zeros((total_length, 10)),
        )
        gt_vertices = gt_output.vertices.detach() - gt_output.joints.detach()[:, 0:1] + gt_position[:, 0:1]
        save_two_meshes_with_colors(gt_vertices[::vis_interval], smplx_faces,
                                    scene_vertices_to_world[0].detach().reshape(-1, 3),
                                    scene_faces[0],
                                    smpl_color = [0.53, 0.81, 0.98],
                                    filename=os.path.splitext(save_name)[0] + "_gt" + os.path.splitext(save_name)[1])



        
        predicted_angle_np = conversions.R2A(predicted_rot.cpu().numpy())
        predicted_angle = torch.from_numpy(predicted_angle_np).cuda().float()
        predicted_root_angle = predicted_angle[...,0,:] 

        gt_angle_np = conversions.R2A(gt_rot.cpu().numpy()) 
        gt_angle = torch.from_numpy(gt_angle_np).cuda().float() 
        gt_root_angle = gt_angle[...,0,:] 
                
        # after running iterations get numbers
        upper_index = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        lower_index = [0, 1, 2, 4, 5, 7, 8] # 10,11 is not considered in imus. (why? TIP does not have ankle joints)
        hand_index = [20, 21]
        foot_index = [7, 8]
        eval_log = {}
        for metric in self.eval_metric:
            eval_metric = get_metric_function(metric)(
                    predicted_position,
                    predicted_angle,
                    predicted_root_angle,
                    gt_position,
                    gt_angle,
                    gt_root_angle,
                    upper_index,
                    lower_index,
                    hand_index,
                    foot_index,
                    fps=motion_constants.FPS,
                    root_rel=True
                ).cpu().numpy()
            eval_log[metric] = eval_metric 
        
        # add filename
        parts = sampled_batch['filename'].split('/')
        filename = '/'.join(parts[-1:])
        eval_log['filename'] = filename
        torch.cuda.empty_cache()

        return eval_log


    def eval_multin(self, is_vis=False):
        logging.info(f"Eval multiple hypothesis with testset ...")
        self.teacher_forcing_ratio = 0
        self.model.eval()

        self.eval_log = {}
        for metric in self.eval_metric:
            self.eval_log[metric] = []
        
        count = 0
        filenames = []
        self.eval_log_by_filename = {}

        render_result_dict = {}
        render_result_dict['fps'] = 30.0
        render_result_dict['seq_len'] = []
        render_result_dict['idx'] = []
        render_result_dict['motion'] = []

        for i, filepath in tqdm(enumerate(self.eval_files_gimo)):
            with open(filepath, "rb") as file:
                file_dict = pickle.load(file)
                file_dict.update({"filename": filepath})
            if is_vis:
                eval_log_per_file = self.run_multin_per_file_vis(file_dict=file_dict, save_name = f'vis_multin/gimo_eval_{i:03d}.ply', vis_interval=30)
            else:
                eval_log_per_file = self.run_multin_per_file(file_dict=file_dict, multin=args.multin, save_name = None)
            if eval_log_per_file['filename'] in self.eval_log_by_filename:
                embed()
            self.eval_log_by_filename[eval_log_per_file['filename']] = eval_log_per_file
            for metric in self.eval_metric:
                self.eval_log[metric].append(eval_log_per_file[metric])
        
        print(f"Done.")
        logging.info(f"-----------------------GIMO EVAL RESULT: {args.multin} Hypos-----------------------------------------------")
        for metric in self.eval_metric:
            if 'jitter' == metric:
                continue
            print(f"metric: {metric} value: {np.mean(np.array(self.eval_log[metric])) * metrics_coeffs[metric]:.2f}")
        print(f"metric: jitter value: {np.mean(np.array(self.eval_log['pred_jitter'])) / np.mean(np.array(self.eval_log['gt_jitter'])):.2f}")
        logging.info(f"--------------------------------------------------------------------------------------")
  
        for i, filepath in tqdm(enumerate(self.eval_files_egobody)):
            with open(filepath, "rb") as file:
                file_dict = pickle.load(file)
                file_dict.update({"filename": filepath})
            if is_vis:
                eval_log_per_file = self.run_multin_per_file_vis(file_dict=file_dict, save_name = f'vis_multin/egobody_eval_{i:03d}.ply', vis_interval=90)
            else: 
                eval_log_per_file = self.run_multin_per_file(file_dict=file_dict, multin=args.multin, save_name = None)
            if eval_log_per_file['filename'] in self.eval_log_by_filename:
                embed()
            self.eval_log_by_filename[eval_log_per_file['filename']] = eval_log_per_file
            for metric in self.eval_metric:
                self.eval_log[metric].append(eval_log_per_file[metric])
        
        print(f"Done.")
        logging.info(f"-----------------------Egobody EVAL RESULT: {args.multin} Hypos--------------------------------------------")
        for metric in self.eval_metric:
            if 'jitter' == metric:
                continue
            print(f"metric: {metric} value: {np.mean(np.array(self.eval_log[metric])) * metrics_coeffs[metric]:.2f}")
        print(f"metric: jitter value: {np.mean(np.array(self.eval_log['pred_jitter'])) / np.mean(np.array(self.eval_log['gt_jitter'])):.2f}")
        logging.info(f"--------------------------------------------------------------------------------------")


    def run_multin_per_file(self, file_dict, multin=5, save_name = None):
        sampled_batch = file_dict
        total_length = sampled_batch['total_length']

        input_seq = sampled_batch['input_seq'].to(self.device)
        input_img = None
        input_pc = sampled_batch['scene_points'].to(self.device)
  
        # norm_input
        input_seq = (input_seq - self.mean) / self.std 
  
        output_tuple = self.model(input_seq.float(), input_img = input_img, input_pc = input_pc)
  
        _, _, pred_mean, _, pred_logvar = output_tuple
        B, T, D = pred_mean.shape

        tgt_seq = sampled_batch['tgt_seq'].to(self.device) # [batch, seq_len, dim]
        global_pos = sampled_batch['global_p'].to(self.device)
        root = sampled_batch['root'].to(self.device)

        output_mean_total = torch.zeros(size=(total_length, D))
        output_logvar_total = torch.zeros(size=(total_length, D))
        tgt_seq_total = torch.zeros(size=(total_length, D))
        tgt_global_p_total = torch.zeros(size=(total_length, global_pos.size(-2), 3))

        tau = 0.3   # NOTE hard code control noise strength
        for idx, info in enumerate(sampled_batch['info']):
            start_frame = int(info['start_end'][0])
            output_mean_total[start_frame:start_frame+T] = pred_mean[idx]
            output_logvar_total[start_frame:start_frame+T] = pred_logvar[idx]
            tgt_seq_total[start_frame:start_frame+T] = tgt_seq[idx]
            tgt_global_p_total[start_frame:start_frame+T] = global_pos[idx]
        
        output_mean_total = output_mean_total.unsqueeze(1).repeat(1, multin, 1)
        output_logvar_total = output_logvar_total.unsqueeze(1).repeat(1, multin, 1)
        tgt_seq_total = tgt_seq_total.unsqueeze(1).repeat(1, multin, 1).to(self.device)
        tgt_global_p_total = tgt_global_p_total.unsqueeze(1).repeat(1, multin, 1, 1).to(self.device)            # T_total, multin, 22, 3
        tgt_global_p_total = tgt_global_p_total.permute(1, 0, 2, 3)
        output_std_total  = torch.exp(0.5 * output_logvar_total)                        # T_total, multin, D

        gp = GPTimeNoiseTBD(total_length, lengthscale=5.0, jitter=1e-6, device=output_mean_total.device, dtype=output_mean_total.dtype)  # 可缓存
        eps = gp.sample_eps(multin, D)  # [T,B,D]
        output_theta_total = output_mean_total + tau * output_std_total * eps
        output_theta_total = output_theta_total.permute(1, 0, 2).to(self.device)            # multin, T_tot, D

        output_root = output_theta_total[...,:3]
        output_joint_rot = output_theta_total[...,3:]
        output_joint_rot = output_joint_rot.reshape(multin, total_length, -1, 6)
        target_joint_rot = tgt_seq_total[...,3:].permute(1, 0, 2).reshape(multin, total_length, -1, 6)

        output_joint_rotmat = transforms.rotation_6d_to_matrix(output_joint_rot)

        # compare pos & ee 
        if self.skel_offset.shape[0] != total_length:
            output_pos_mat = rot_matrix_fk_tensor(output_joint_rotmat, output_root, self.skel_offset[0:total_length], self.skel_parent)
        else:
            output_pos_mat = rot_matrix_fk_tensor(output_joint_rotmat, output_root, self.skel_offset, self.skel_parent)

        results = {}
        results['pred_pos'] = output_pos_mat.clone().detach()               # multin, T_tot, 22, 3
        results['pred_rot'] = output_joint_rot.clone().detach()             # multin, T_tot, 22, 6
        results['gt_pos'] = tgt_global_p_total.clone().detach()             # multin, T_tot, 22, 3
        results['gt_rot'] = target_joint_rot.clone().detach()               # multin, T_tot, 22, 6

        start_T = sampled_batch['head_start'].to(self.device) # Start pos   B,1,1,4,4
        
        predicted_position = torch.zeros_like(results['pred_pos'])
        pred_rotmat = transforms.rotation_6d_to_matrix(results['pred_rot'])
        gt_position = torch.zeros_like(results['gt_pos'])
        gt_rotmat = transforms.rotation_6d_to_matrix(results['gt_rot'])
        for idx, info in enumerate(sampled_batch['info']):
            start_frame = int(info['start_end'][0])
            predicted_position[:, start_frame:start_frame+T] = (start_T[idx, ...,:3,:3] @ results['pred_pos'][:, start_frame:start_frame+T].unsqueeze(-1)).squeeze(-1)
            predicted_position[:, start_frame:start_frame+T, 0] = predicted_position[:, start_frame:start_frame+T, 0] + start_T[idx, ...,:3,3]
            pred_rotmat[:, start_frame:start_frame+T, 0:1, :, :] = start_T[idx, ...,:3,:3] @ pred_rotmat[:, start_frame:start_frame+T, 0:1, :, :]

            gt_position[:, start_frame:start_frame+T] = (start_T[idx, ...,:3,:3] @ results['gt_pos'][:, start_frame:start_frame+T].unsqueeze(-1)).squeeze(-1)
            gt_position[:, start_frame:start_frame+T, 0] = gt_position[:, start_frame:start_frame+T, 0] + start_T[idx, ...,:3,3]
            gt_rotmat[:, start_frame:start_frame+T, 0:1, :, :] = start_T[idx, ...,:3,:3] @ gt_rotmat[:, start_frame:start_frame+T, 0:1, :, :]
        

        predicted_angle_np = conversions.R2A(pred_rotmat.cpu().numpy())
        predicted_angle = torch.from_numpy(predicted_angle_np).cuda().float()
        predicted_root_angle = predicted_angle[...,0,:] 

        gt_angle_np = conversions.R2A(gt_rotmat.cpu().numpy()) 
        gt_angle = torch.from_numpy(gt_angle_np).cuda().float() 
        gt_root_angle = gt_angle[...,0,:] 
                
        # after running iterations get numbers
        upper_index = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        lower_index = [0, 1, 2, 4, 5, 7, 8] # 10,11 is not considered in imus. (why? TIP does not have ankle joints)
        hand_index = [20, 21]
        foot_index = [7, 8]
        eval_log = {}
        for metric in self.eval_metric:
            eval_log[metric] = []
            for j in range(multin):
                eval_metric = get_metric_function(metric)(
                        predicted_position[j],
                        predicted_angle[j],
                        predicted_root_angle[j],
                        gt_position[j],
                        gt_angle[j],
                        gt_root_angle[j],
                        upper_index,
                        lower_index,
                        hand_index,
                        foot_index,
                        fps=motion_constants.FPS,
                        root_rel=True
                    ).cpu().numpy().item()
                eval_log[metric].append(eval_metric)
        
        all_min_indices = {}
        for metric, values in eval_log.items():
            min_index = np.argmin(values)
            # print(f"Metric '{metric}': 最小值的index是 {min_index}")
            all_min_indices[metric] = min_index

        min_idx = all_min_indices['mpjpe']

        for metric, values in eval_log.items():
            eval_log[metric] = values[min_idx]

        
        # add filename
        parts = sampled_batch['filename'].split('/')
        filename = '/'.join(parts[-1:])
        eval_log['filename'] = filename
        torch.cuda.empty_cache()

        return eval_log

    def run_multin_per_file_vis(self, file_dict, multin=5, save_name = None, vis_interval=30):
        sampled_batch = file_dict
        total_length = sampled_batch['total_length']

        input_seq = sampled_batch['input_seq'].to(self.device)
        input_img = None
        input_pc = sampled_batch['scene_points'].to(self.device)
  
        # norm_input
        input_seq = (input_seq - self.mean) / self.std 
  
        output_tuple = self.model(input_seq.float(), input_img = input_img, input_pc = input_pc)

        scene_vertices = sampled_batch['mesh_vertices'].to(self.device)
        scene_faces = sampled_batch['mesh_faces'].to(self.device)
  
        _, _, pred_mean, _, pred_logvar = output_tuple
        B, T, D = pred_mean.shape

        tgt_seq = sampled_batch['tgt_seq'].to(self.device) # [batch, seq_len, dim]
        global_pos = sampled_batch['global_p'].to(self.device)
        root = sampled_batch['root'].to(self.device)

        output_mean_total = torch.zeros(size=(total_length, D))
        output_logvar_total = torch.zeros(size=(total_length, D))
        tgt_seq_total = torch.zeros(size=(total_length, D))
        tgt_global_p_total = torch.zeros(size=(total_length, global_pos.size(-2), 3))


        tau = 0.3   # NOTE hard code control noise strength
        for idx, info in enumerate(sampled_batch['info']):
            start_frame = int(info['start_end'][0])
            output_mean_total[start_frame:start_frame+T] = pred_mean[idx]
            output_logvar_total[start_frame:start_frame+T] = pred_logvar[idx]
            tgt_seq_total[start_frame:start_frame+T] = tgt_seq[idx]
            tgt_global_p_total[start_frame:start_frame+T] = global_pos[idx]
        
        output_mean_total = output_mean_total.unsqueeze(1).repeat(1, multin, 1)
        output_logvar_total = output_logvar_total.unsqueeze(1).repeat(1, multin, 1)
        tgt_seq_total = tgt_seq_total.unsqueeze(1).repeat(1, multin, 1).to(self.device)
        tgt_global_p_total = tgt_global_p_total.unsqueeze(1).repeat(1, multin, 1, 1).to(self.device)            # T_total, multin, 22, 3
        tgt_global_p_total = tgt_global_p_total.permute(1, 0, 2, 3)
        output_std_total  = torch.exp(0.5 * output_logvar_total)                        # T_total, multin, D

        gp = GPTimeNoiseTBD(total_length, lengthscale=5.0, jitter=1e-6, device=output_mean_total.device, dtype=output_mean_total.dtype)  # 可缓存
        eps = gp.sample_eps(multin, D)  # [T,B,D]
        eps[:, :, :12] = 0.
        output_theta_total = output_mean_total + tau * output_std_total * eps
        output_theta_total = output_theta_total.permute(1, 0, 2).to(self.device)            # multin, T_tot, D

        output_root = output_theta_total[...,:3]
        output_joint_rot = output_theta_total[...,3:]
        output_joint_rot = output_joint_rot.reshape(multin, total_length, -1, 6)
        target_joint_rot = tgt_seq_total[...,3:].permute(1, 0, 2).reshape(multin, total_length, -1, 6)

        output_joint_rotmat = transforms.rotation_6d_to_matrix(output_joint_rot)

        # compare pos & ee 
        if self.skel_offset.shape[0] != total_length:
            output_pos_mat = rot_matrix_fk_tensor(output_joint_rotmat, output_root, self.skel_offset[0:total_length], self.skel_parent)
        else:
            output_pos_mat = rot_matrix_fk_tensor(output_joint_rotmat, output_root, self.skel_offset, self.skel_parent)

        results = {}
        results['pred_pos'] = output_pos_mat.clone().detach()               # multin, T_tot, 22, 3
        results['pred_rot'] = output_joint_rot.clone().detach()             # multin, T_tot, 22, 6
        results['gt_pos'] = tgt_global_p_total.clone().detach()             # multin, T_tot, 22, 3
        results['gt_rot'] = target_joint_rot.clone().detach()               # multin, T_tot, 22, 6

        start_T = sampled_batch['head_start'].to(self.device) # Start pos   B,1,1,4,4
        
        predicted_position = torch.zeros_like(results['pred_pos'])
        pred_rotmat = transforms.rotation_6d_to_matrix(results['pred_rot'])
        gt_position = torch.zeros_like(results['gt_pos'])
        gt_rotmat = transforms.rotation_6d_to_matrix(results['gt_rot'])
        for idx, info in enumerate(sampled_batch['info']):
            start_frame = int(info['start_end'][0])
            predicted_position[:, start_frame:start_frame+T] = (start_T[idx, ...,:3,:3] @ results['pred_pos'][:, start_frame:start_frame+T].unsqueeze(-1)).squeeze(-1)
            predicted_position[:, start_frame:start_frame+T, 0] = predicted_position[:, start_frame:start_frame+T, 0] + start_T[idx, ...,:3,3]
            pred_rotmat[:, start_frame:start_frame+T, 0:1, :, :] = start_T[idx, ...,:3,:3] @ pred_rotmat[:, start_frame:start_frame+T, 0:1, :, :]

            gt_position[:, start_frame:start_frame+T] = (start_T[idx, ...,:3,:3] @ results['gt_pos'][:, start_frame:start_frame+T].unsqueeze(-1)).squeeze(-1)
            gt_position[:, start_frame:start_frame+T, 0] = gt_position[:, start_frame:start_frame+T, 0] + start_T[idx, ...,:3,3]
            gt_rotmat[:, start_frame:start_frame+T, 0:1, :, :] = start_T[idx, ...,:3,:3] @ gt_rotmat[:, start_frame:start_frame+T, 0:1, :, :]
        
        pc_to_world = start_T[...,:3,:3].to(self.device) @ input_pc.unsqueeze(-1).unsqueeze(2)
        pc_to_world = pc_to_world[..., 0] + start_T[...,:3,3]

        scene_vertices_to_world = start_T[...,:3,:3].to(self.device) @ scene_vertices.unsqueeze(-1).unsqueeze(2)
        scene_vertices_to_world = scene_vertices_to_world[..., 0] + start_T[...,:3,3]           # B, num_of_points, 1, 3


        for k in range(multin):
            predicted_rot = pred_rotmat[k].clone().cpu()
            gt_rot = gt_rotmat[k].clone().cpu()
            pred_aa_24 = torch.Tensor(conversions.R2A(predicted_rot))
            pred_global_orient = pred_aa_24[:, 0]                                        # [B,3]
            pred_body_pose = pred_aa_24[:, 1:].reshape(total_length, 21*3)

            gt_aa_24 = torch.Tensor(conversions.R2A(gt_rot))
            gt_global_orient = gt_aa_24[:, 0]                                        # [B,3]
            gt_body_pose = gt_aa_24[:, 1:].reshape(total_length, 21*3)

            # Betas
            betas = torch.zeros(total_length, 10)
            zero_pose = torch.zeros(total_length, 10)

            # Build SMPL layer
            pred_output = self.smplx_model(
                global_orient=pred_global_orient,   # [B,3]
                body_pose=pred_body_pose,           # [B,69]
                betas=betas,                   # [B,10]
                left_hand_pose=torch.zeros((total_length, 45)), 
                right_hand_pose=torch.zeros((total_length, 45)),
                jaw_pose=torch.zeros((total_length, 3)), 
                leye_pose=torch.zeros((total_length, 3)),
                reye_pose=torch.zeros((total_length, 3)),
                expression=torch.zeros((total_length, 10)),
            )
            pred_vertices = pred_output.vertices.detach() - pred_output.joints.detach()[:, 0:1] + predicted_position[k, :, 0:1].detach().cpu()
            smplx_faces = self.smplx_model.faces
            save_two_meshes_with_colors(pred_vertices[::vis_interval], smplx_faces,
                                        scene_vertices_to_world[0].detach().reshape(-1, 3),
                                        scene_faces[0],
                                        smpl_color = [1.0, 0.75, 0.8],
                                        filename=os.path.splitext(save_name)[0] + f"_pred_{k:02d}" + os.path.splitext(save_name)[1])
            if k == 0:
                gt_output = self.smplx_model(
                    global_orient=gt_global_orient,   # [B,3]
                    body_pose=gt_body_pose,           # [B,69]
                    betas=betas,                   # [B,10]
                    left_hand_pose=torch.zeros((total_length, 45)), 
                    right_hand_pose=torch.zeros((total_length, 45)),
                    jaw_pose=torch.zeros((total_length, 3)), 
                    leye_pose=torch.zeros((total_length, 3)),
                    reye_pose=torch.zeros((total_length, 3)),
                    expression=torch.zeros((total_length, 10)),
                )
                gt_vertices = gt_output.vertices.detach() - gt_output.joints.detach()[:, 0:1] + gt_position[k, :, 0:1].detach().cpu()
                save_two_meshes_with_colors(gt_vertices[::vis_interval], smplx_faces,
                                            scene_vertices_to_world[0].detach().reshape(-1, 3),
                                            scene_faces[0],
                                            smpl_color = [0.53, 0.81, 0.98],
                                            filename=os.path.splitext(save_name)[0] + f"_gt" + os.path.splitext(save_name)[1])
        
        
        predicted_angle_np = conversions.R2A(pred_rotmat.cpu().numpy())
        predicted_angle = torch.from_numpy(predicted_angle_np).cuda().float()
        predicted_root_angle = predicted_angle[...,0,:] 

        gt_angle_np = conversions.R2A(gt_rotmat.cpu().numpy()) 
        gt_angle = torch.from_numpy(gt_angle_np).cuda().float() 
        gt_root_angle = gt_angle[...,0,:] 
                
        # after running iterations get numbers
        upper_index = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        lower_index = [0, 1, 2, 4, 5, 7, 8] # 10,11 is not considered in imus. (why? TIP does not have ankle joints)
        hand_index = [20, 21]
        foot_index = [7, 8]
        eval_log = {}
        for metric in self.eval_metric:
            eval_log[metric] = []
            for j in range(multin):
                eval_metric = get_metric_function(metric)(
                        predicted_position[j],
                        predicted_angle[j],
                        predicted_root_angle[j],
                        gt_position[j],
                        gt_angle[j],
                        gt_root_angle[j],
                        upper_index,
                        lower_index,
                        hand_index,
                        foot_index,
                        fps=motion_constants.FPS,
                        root_rel=True
                    ).cpu().numpy().item()
                eval_log[metric].append(eval_metric)
        
        all_min_indices = {}
        for metric, values in eval_log.items():
            min_index = np.argmin(values)
            # print(f"Metric '{metric}': 最小值的index是 {min_index}")
            all_min_indices[metric] = min_index

        min_idx = all_min_indices['mpjpe']

        for metric, values in eval_log.items():
            eval_log[metric] = values[min_idx]

        
        # add filename
        parts = sampled_batch['filename'].split('/')
        filename = '/'.join(parts[-1:])
        eval_log['filename'] = filename
        torch.cuda.empty_cache()

        return eval_log

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--config_file", type=str, default="")
    parser.add_argument("--test_name", type=str, default="")
    parser.add_argument("--mode", type=str, default="", choices=["test", "train", "custom", "test_vis"])
    parser.add_argument("--multin", type=int, default=1)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--setting", type=str, default="vr", choices = ['hmc', 'vr'])

    args = parser.parse_args()
    if not os.path.exists("./output/"):
        os.mkdir("./output/")
    
    imu2body_network = IMU2BodyNetwork(args=args)

    if args.mode == "train":
        imu2body_network.train()
    elif args.mode == 'test':
        imu2body_network.pretrain = True
        imu2body_network.build_network_gimo()
        imu2body_network.model.cuda()
        if args.multin > 1:
            print("#################### Multi hypothesis mode ####################")
            with torch.no_grad():
                imu2body_network.eval_multin()
        else:
            with torch.no_grad():
                imu2body_network.eval()
    elif args.mode == 'test_vis':
        imu2body_network.pretrain = True
        imu2body_network.build_network_gimo()
        imu2body_network.model.cuda()
        if args.multin > 1:
            print("#################### Multi hypothesis mode ####################")
            with torch.no_grad():
                imu2body_network.eval_multin(is_vis=True)
        else:
            with torch.no_grad():
                imu2body_network.eval_vis()
    else:
        imu2body_network.run(mode=args.mode)