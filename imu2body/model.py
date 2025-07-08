import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
from imu2body.model_base import TransformerEncoderModel, TransformerSceneEncoderModel, Dinov2Backbone
from imu2body.model_base import PointNet2SemSegSSGShape, PointNet, FPModule

import torch
import torch.nn as nn
from IPython import embed


class IMU2BodyModel(nn.Module):
    def __init__(self, data_config, model_config):
        super(IMU2BodyModel, self).__init__()

        input_dim = data_config['input_dim']
        mid_dim = data_config['mid_dim']
        output_dim = data_config['output_dim']

        self.use_sep_encoder = model_config['sep_encoder']
        self.temporal = model_config['temporal_model']
        self.hand_estimator = model_config['hand_estimator']
        self.visual_input = model_config['visual_input']
        self.environment_enc = model_config['environment_enc']
        
        if self.use_sep_encoder:
            hand2body_input_dim = (input_dim, mid_dim)
        else:
            hand2body_input_dim = input_dim + mid_dim
            
        if self.visual_input:
            pass
        
        if self.environment_enc:
            # imu + head -> ee pose 
            self.imu2hand = TransformerSceneEncoderModel(
                input_dim=input_dim,
                output_dim=mid_dim,
                hidden_dim=model_config['hidden_dim1'],
                num_heads=model_config['num_head1'],
                temporal=self.temporal
            )
        else: 
            # imu + head -> ee pose 
            self.imu2hand = TransformerEncoderModel(
                input_dim=input_dim,
                output_dim=mid_dim,
                hidden_dim=model_config['hidden_dim1'],
                num_heads=model_config['num_head1'],
                temporal=self.temporal
            )

        # imu + head + ee pose -> contact, output
        self.hand2body = TransformerEncoderModel(
            input_dim=hand2body_input_dim,
            output_dim=output_dim,
            hidden_dim=model_config['hidden_dim2'],
            num_heads=model_config['num_head2'],
            estimate_contact=True,
            temporal=self.temporal
        )

    def init_weights(self):
        self.imu2hand.init_weights()
        self.hand2body.init_weights()
    
    def forward(self, input_seq, input_img = None, input_pc = None):
        # if self.environment_enc:
        #     scene_feats, scene_global_feats = self.environment_enc(input_pc.repeat(1, 1, 2)) #[64, 1280, 10000], [64, 1280]
        #     motion_scene_feats = self.pointnet(scene_feats)#.reshape((1, , -1))
            
        # else:
        #     scene_feats = None
        #     scene_global_feats = None
        if self.environment_enc:
            _, ee = self.imu2hand(input_seq, input_pc)
            input_concat = torch.cat((input_seq, ee), -1)
            contact, output = self.hand2body(input_concat, input_pc) #output: [batch. seq, 135]
        else: 
            _, ee = self.imu2hand(input_seq) # hand: [batch, seq, 12]
            input_concat = torch.cat((input_seq, ee), -1)  #concatenate input and hand | ee: [batch, seq, 135]
            contact, output = self.hand2body(input_concat) #output: [batch. seq, 135]
        return ee, contact, output
    
def load_model(data_config, model_config):
    return IMU2BodyModel(data_config=data_config, model_config=model_config)