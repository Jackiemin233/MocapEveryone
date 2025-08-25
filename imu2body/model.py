import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
from imu2body.model_base import TransformerEncoderModel, TransformerSceneEncoderModel, TransformerSceneFiLMModel, \
                                TransformerEncoderModel_Uncertain, TransformerSceneFiLMModel_Uncertain
# from imu2body.model_base import PointNet2SemSegSSGShape, PointNet, FPModule
# from imu2body.model_base import WaveletEmbedding
from imu2body.model_base_swt import WaveletSWTBlock, TransformerSceneFiLMModel_SWT
from imu2body.pointnet2 import PointNet2Encoder

import torch
import torch.nn as nn
from IPython import embed
import einops

# Baseline Model
# class IMU2BodyModel(nn.Module):
#     def __init__(self, data_config, model_config):
#         super(IMU2BodyModel, self).__init__()

#         input_dim = data_config['input_dim']
#         mid_dim = data_config['mid_dim']
#         output_dim = data_config['output_dim']
#         print(input_dim, mid_dim, output_dim)

#         self.use_sep_encoder = model_config['sep_encoder']
#         self.temporal = model_config['temporal_model']
#         self.hand_estimator = model_config['hand_estimator']
#         self.visual_input = model_config['visual_input']
#         self.environment_enc = model_config['environment_enc']
        
#         if self.use_sep_encoder:
#             hand2body_input_dim = (input_dim, mid_dim)
#         else:
#             hand2body_input_dim = input_dim + mid_dim
            
#         if self.visual_input:
#             pass
        
#         if self.environment_enc:
#             # imu + head -> ee pose 
#             self.imu2hand = TransformerSceneEncoderModel(
#                 input_dim=input_dim,
#                 output_dim=mid_dim,
#                 hidden_dim=model_config['hidden_dim1'],
#                 num_heads=model_config['num_head1'],
#                 temporal=self.temporal
#             )
#         else: 
#             # imu + head -> ee pose 
#             self.imu2hand = TransformerEncoderModel(
#                 input_dim=input_dim,
#                 output_dim=mid_dim,
#                 hidden_dim=model_config['hidden_dim1'],
#                 num_heads=model_config['num_head1'],
#                 temporal=self.temporal
#             )

#         # imu + head + ee pose -> contact, output
#         self.hand2body = TransformerEncoderModel(
#             input_dim=hand2body_input_dim,
#             output_dim=output_dim,
#             hidden_dim=model_config['hidden_dim2'],
#             num_heads=model_config['num_head2'],
#             estimate_contact=True,
#             temporal=self.temporal
#         )

#     def init_weights(self):
#         self.imu2hand.init_weights()
#         self.hand2body.init_weights()
    
#     def forward(self, input_seq, input_img = None, input_pc = None):
#         # if self.environment_enc:
#         #     scene_feats, scene_global_feats = self.environment_enc(input_pc.repeat(1, 1, 2)) #[64, 1280, 10000], [64, 1280]
#         #     motion_scene_feats = self.pointnet(scene_feats)#.reshape((1, , -1))
            
#         # else:
#         #     scene_feats = None
#         #     scene_global_feats = None
#         if self.environment_enc:
#             _, ee = self.imu2hand(input_seq, input_pc)
#             input_concat = torch.cat((input_seq, ee), -1)
#             contact, output = self.hand2body(input_concat, input_pc) #output: [batch. seq, 135]
#         else: 
#             _, ee = self.imu2hand(input_seq) # hand: [batch, seq, 12]
#             input_concat = torch.cat((input_seq, ee), -1)  #concatenate input and hand | ee: [batch, seq, 135]
#             contact, output = self.hand2body(input_concat) #output: [batch. seq, 135]
#         return ee, contact, output
    
    

class IMU2BodyModel(nn.Module):
    def __init__(self, data_config, model_config):
        super(IMU2BodyModel, self).__init__()

        input_dim = data_config['input_dim']
        mid_dim = data_config['mid_dim']
        output_dim = data_config['output_dim']
        # print(input_dim, mid_dim, output_dim)

        self.use_sep_encoder = model_config['sep_encoder']
        self.temporal = model_config['temporal_model']
        self.hand_estimator = model_config['hand_estimator']
        self.visual_input = model_config['visual_input']
        self.environment_enc = model_config.get('environment_enc', False)
        self.use_uncertainty = model_config.get('uncertainty_model', False)
        self.use_film = model_config.get('use_film', False)
        self.use_swt = model_config.get('use_swt', False)

        if self.environment_enc:
            self.pcd_encoder = PointNet2Encoder()
        
        if self.use_sep_encoder:
            hand2body_input_dim = (input_dim, mid_dim)
        else:
            hand2body_input_dim = input_dim + mid_dim
            
        if self.visual_input:
            pass
        
        # imu + head -> ee pose 
        self.imu2hand = TransformerEncoderModel(
            input_dim=input_dim,
            output_dim=mid_dim,
            hidden_dim=model_config['hidden_dim1'],
            num_heads=model_config['num_head1'],
        )

        # imu + head + ee pose -> contact, output
        if self.environment_enc:
            if self.use_film and not self.use_uncertainty and not self.use_swt:
                self.hand2body = TransformerSceneFiLMModel(
                    input_dim=hand2body_input_dim,
                    output_dim=output_dim,
                    hidden_dim=model_config['hidden_dim2'],
                    num_heads=model_config['num_head2'],
                    estimate_contact=True,
                    context_dim=1280
                )
            elif self.use_film and self.use_uncertainty and not self.use_swt:
                print("USING UNCERTAINTY FILM MODEL")
                self.hand2body = TransformerSceneFiLMModel_Uncertain(
                    input_dim=hand2body_input_dim,
                    output_dim=output_dim,
                    hidden_dim=model_config['hidden_dim2'],
                    num_heads=model_config['num_head2'],
                    estimate_contact=True,
                    context_dim=1280
                )
            elif self.use_uncertainty and not self.use_swt:
                print("USING UNCERTAINTY MODEL")
                self.hand2body = TransformerEncoderModel_Uncertain(
                    input_dim=hand2body_input_dim,
                    output_dim=output_dim,
                    hidden_dim=model_config['hidden_dim2'],
                    num_heads=model_config['num_head2'],
                    estimate_contact=True,
                )
            elif self.use_swt:
                print("USING WAVELET TRANSFORM")
                self.hand2body = TransformerSceneFiLMModel_SWT(
                    input_dim=hand2body_input_dim,
                    output_dim=output_dim,
                    hidden_dim=model_config['hidden_dim2'],
                    num_heads=model_config['num_head2'],
                    estimate_contact=True,
                    context_dim=1280,
                    wavelet_levels=3
                )
            else:
                self.hand2body = TransformerSceneEncoderModel(
                    input_dim=hand2body_input_dim,
                    output_dim=output_dim,
                    hidden_dim=model_config['hidden_dim2'],
                    num_heads=model_config['num_head2'],
                    estimate_contact=True,
                )
        else:
            self.hand2body = TransformerEncoderModel(
                input_dim=hand2body_input_dim,
                output_dim=output_dim,
                hidden_dim=model_config['hidden_dim2'],
                num_heads=model_config['num_head2'],
                estimate_contact=True,
            )
        
    def init_weights(self):
        self.imu2hand.init_weights()
        self.hand2body.init_weights()
    
    def forward(self, input_seq, input_img = None, input_pc = None):
        # if self.environment_enc:
        #     scene_feats, scene_global_feats = self.environment_enc(input_pc.repeat(1, 1, 2)) #[64, 1280, 10000], [64, 1280]
        _, ee = self.imu2hand(input_seq) # hand: [batch, seq, 12]
        input_concat = torch.cat((input_seq, ee), -1)  #concatenate input and hand | ee: [batch, seq, 135]
        if self.environment_enc:
            input_pc = input_pc - input_pc.mean(dim=1, keepdim=True)
            input_pc = input_pc / (input_pc.norm(dim=2, keepdim=True).amax(dim=1, keepdim=True) + 1e-8)
            env_context = self.pcd_encoder(input_pc.permute(0, 2, 1))       # B,N,3 -> B,3,N
            if self.use_uncertainty:
                contact, mean, logvar, sampled_output = self.hand2body(input_concat, context=env_context, sample=True)  # 加入点云特征
                return ee, contact, mean, sampled_output, logvar          # B,T,D
            else:
                contact, output = self.hand2body(input_concat, context=env_context)  # 加入点云特征
        else:
            contact, output = self.hand2body(input_concat) #output: [batch. seq, 135]
        return ee, contact, output
    
def load_model(data_config, model_config):
    return IMU2BodyModel(data_config=data_config, model_config=model_config)




