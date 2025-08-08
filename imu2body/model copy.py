import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
from imu2body.model_base import TransformerEncoderModel, TransformerSceneEncoderModel, Dinov2Backbone, WaveletTransformerEncoderModel
from imu2body.model_base import PointNet2SemSegSSGShape, PointNet, FPModule
from imu2body.model_base import WaveletEmbedding
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch
import torch.nn as nn
from IPython import embed
import einops


# class IMU2BodyModel(nn.Module):
#     def __init__(self, data_config, model_config):
#         super(IMU2BodyModel, self).__init__()

#         input_dim = data_config['input_dim']
#         mid_dim = data_config['mid_dim']
#         output_dim = data_config['output_dim']

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

        input_dim = data_config['input_dim'] #* 3
        mid_dim = data_config['mid_dim'] #* 3 
        output_dim = data_config['output_dim'] #* 3

        self.use_sep_encoder = model_config['sep_encoder']
        self.temporal = model_config['temporal_model']
        self.hand_estimator = model_config['hand_estimator']
        self.visual_input = model_config['visual_input']
        self.environment_enc = model_config['environment_enc']
        self.wavelet_embedding = model_config['wavelet_emb']
        
        if self.wavelet_embedding:
            self.waveletdecomp_hand = WaveletEmbedding(d_channel = 40, swt = True, m = 3, requires_grad=True)
            self.waveletrecon_hand = WaveletEmbedding(d_channel = 40, swt = False, m = 3, requires_grad=True)
            
            self.waveletdecomp_body = WaveletEmbedding(d_channel = 40, swt = True, m = 3, requires_grad=True)
            self.waveletrecon_body = WaveletEmbedding(d_channel = 40, swt = False, m = 3, requires_grad=True)
            
            self.m = 3
                
        if self.use_sep_encoder:
            hand2body_input_dim = (input_dim, mid_dim)

        else:
            hand2body_input_dim = input_dim + mid_dim
            
        if self.visual_input:
            pass
        
        # imu + head -> ee pose 
        self.imu2hand = TransformerEncoderModel(
            input_dim=input_dim ,
            output_dim=mid_dim,
            hidden_dim=model_config['hidden_dim1'],
            num_heads=model_config['num_head1'],
            temporal=self.temporal
        )
        # # imu + head -> ee pose 
        self.imu2hand_high = TransformerEncoderModel(
            input_dim=input_dim,
            output_dim=mid_dim,
            hidden_dim=model_config['hidden_dim1'],
            num_heads=model_config['num_head1'],
            temporal=self.temporal
        )
        
        # self.imu2hand = WaveletTransformerEncoderModel(
        #     input_dim=input_dim * 2,
        #     output_dim=mid_dim * 4,
        #     hidden_dim=model_config['hidden_dim1'],
        #     num_heads=model_config['num_head1'],
        #     temporal=self.temporal
        # )
        
        # # imu + head + ee pose -> contact, output
        # self.hand2body = WaveletTransformerEncoderModel(
        #     input_dim = hand2body_input_dim,
        #     output_dim=output_dim * 4,
        #     hidden_dim=model_config['hidden_dim2'],
        #     num_heads=model_config['num_head2'],
        #     estimate_contact=True,
        #     temporal=self.temporal,
        # )
        
    def init_weights(self):
        self.imu2hand.init_weights()
        self.hand2body.init_weights()
        self.imu2hand_high.init_weights()
        
    def forward(self, input_seq, input_img = None, input_pc = None):
        _, ee = self.imu2hand(input_seq)
        input_concat = torch.cat((input_seq, ee), -1)  #concatenate input and hand | ee: [batch, seq, 135]
        contact, output = self.hand2body(input_concat) #output: [batch. seq, 135]
        return ee, contact, output
    
    def forward(self, input_seq, input_img = None, input_pc = None):        
        input_embedding = self.waveletdecomp_hand(input_seq) # [batch, seq, o, channel]
        low_embedding = input_embedding[:,:,:2,:]
        hf_embedding = input_embedding[:,:,2:,:]
        
        hf_embedding = einops.rearrange(hf_embedding, 'b seq o c-> b seq (o c)')
        low_embedding = einops.rearrange(low_embedding, 'b seq o c-> b seq (o c)')
            
        _, ee_embedding = self.imu2hand(hf_embedding, low_embedding) # hand: [batch, seq, 12]
        ee_embedding = einops.rearrange(ee_embedding, 'b seq (o c)-> b seq o c', o=4)
        ee = self.waveletrecon_hand(ee_embedding)
        
        input_concat = torch.cat((input_seq, ee), -1)
        input_concat_embedding = self.waveletdecomp_body(input_concat)
        
        low_concat_embedding = input_concat_embedding[:,:,:2,:]
        hf_concat_embedding = input_concat_embedding[:,:,2:,:]
        
        hf_concat_embedding = einops.rearrange(hf_concat_embedding, 'b seq o c-> b seq (o c)')
        low_concat_embedding = einops.rearrange(low_concat_embedding, 'b seq o c-> b seq (o c)')
        
        contact, output_embedding = self.hand2body(hf_concat_embedding, low_concat_embedding) #output: [batch. seq, 135]
        
        output_embedding = einops.rearrange(output_embedding, 'b seq (o c)-> b seq o c', o=4)
        output = self.waveletrecon_body(output_embedding)

        return ee, contact, output
    
    def forward(self, input_seq, input_img = None, input_pc = None):
        if self.wavelet_embedding:
            input_embeddings = self.waveletdecomp(input_seq) # [batch, seq, o, channel]
            #input_embedding = einops.rearrange(input_embedding, 'b seq o c-> b seq (o c)')
        
        output_embedding_list = []
        ee_embedding_list = []
        for i in range(input_embeddings.shape[-2]):
            input_embedding = input_embeddings[:,:,i,:]
            _, ee_embedding = self.imu2hand(input_embedding) # hand: [batch, seq, 12]
            input_concat = torch.cat((input_embedding, ee_embedding), -1)  #concatenate input and hand | ee: [batch, seq, 135]
            contact, output_embedding = self.hand2body(input_concat) #output: [batch. seq, 135]
            
            output_embedding_list.append(output_embedding.unsqueeze(-2))
            ee_embedding_list.append(ee_embedding.unsqueeze(-2))
            
        ee_embedding = torch.cat(ee_embedding_list, dim = -2)
        output_embedding = torch.cat(output_embedding_list, dim = -2)
        
        if self.wavelet_embedding:
            #ee_embedding = einops.rearrange(ee_embedding, 'b seq (o c)-> b seq o c', o=3)
            ee = self.waveletrecon(ee_embedding)
            #output_embedding = einops.rearrange(output_embedding, 'b seq (o c)-> b seq o c', o=3)
            output = self.waveletrecon(output_embedding)

        return ee, contact, output
    
def load_model(data_config, model_config):
    return IMU2BodyModel(data_config=data_config, model_config=model_config)



