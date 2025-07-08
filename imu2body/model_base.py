# (FairMotion) Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import LayerNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn.init import xavier_uniform_

import random
from IPython import embed

# Models for scene encoder
from pointnet2_ops import pointnet2_utils
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG

# add transformer encoder module 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000): # d_model: ninp/hidden_dim in original
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [5000] -> [5000, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        ) # [0.5*d_model]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # unsqueeze: [1, max_len, d_model] -> transpose: [max_len, 1, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model] # self.pe size [max_len, 1, d_model]
        x = x + self.pe[:x.size(0), :] # cut pe into [seq_len, 1, d_model] and add to all batches
        return x
        # return self.dropout(x)

class TransformerEncoderModel(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim=1024, num_layers=4, num_heads=8, dropout=0.1, estimate_contact=False, temporal = True
    ):
        """
        input_dim: this is the dimension of the input
        ninp: 1024 this is the dimension of the hidden layer
        hidden_dim: same as ninp
        num_layers: the number of layers in transformer encoder and decoder. can be either 1 or 4

        """
        self.temporal = temporal
        self.mid_dim = None
        if isinstance(input_dim, tuple):
            self.input_dim, self.mid_dim = input_dim

        self.hidden_dim = hidden_dim

        super(TransformerEncoderModel, self).__init__()
        self.model_type = "TransformerEncoder"

        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, num_heads, hidden_dim, dropout
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=LayerNorm(hidden_dim),
        )
        
        if self.temporal:
            self.time_encoder = torch.nn.LSTM(hidden_dim, hidden_dim, num_layers=1, bidirectional=False, batch_first=False)

        # Use Linear instead of Embedding for continuous valued input
        if self.mid_dim is not None:
            half_hidden_dim = int(hidden_dim/2)
            self.mid_encoder = nn.Linear(self.mid_dim, half_hidden_dim)
            self.input_encoder = nn.Linear(self.input_dim, half_hidden_dim)

        else:
            self.encoder = nn.Linear(input_dim, hidden_dim)

        self.hidden_dim = hidden_dim
        
        # foot fc 
        decode_dim = hidden_dim

        self.estimate_contact = estimate_contact
        if self.estimate_contact:
            self.contact_decoder = nn.Sequential(
                                nn.Linear(hidden_dim, 256),
                                nn.ReLU(),
                                nn.Linear(256, 2)
                )        
            decode_dim += 2

        self.linear_decoder = nn.Sequential(
                            nn.Linear(decode_dim, 256),
                            nn.ReLU(),
                            nn.Linear(256, output_dim)
            )
        
        self.init_weights()

    def init_weights(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


    def forward(self, src):
        # Transformer expects src and tgt in format (len, batch_size, dim)
        src = src.transpose(0, 1) # by transpose, [seq, batch, ninp]
        if self.mid_dim is None:
            projected_src = self.encoder(src) * np.sqrt(self.hidden_dim) # why add np.sqrt? [seq, batch, hidden_dim]
        else:
            half_hidden_dim = int(self.hidden_dim/2)

            src_input, src_mid = src[...,:self.input_dim], src[...,self.input_dim:]
            projected_input_src = self.input_encoder(src_input)
            projected_mid_src = self.mid_encoder(src_mid)
            projected_src = torch.cat((projected_input_src, projected_mid_src),-1) * np.sqrt(self.hidden_dim)

        pos_encoded_src = self.pos_encoder(projected_src) # [seq, batch, hidden_dim]
        encoder_output = self.transformer_encoder(pos_encoded_src) # [seq, batch, ninp] encoder output
        #40,batch,1280
        if self.temporal:
            encoder_output, _ = self.time_encoder(encoder_output)
            #encoder_output = self.temporal_attn(encoder_output)

        if self.estimate_contact:
            contact_output = self.contact_decoder(encoder_output) # [seq, batch, 18]
            encoder_output = torch.cat((encoder_output, contact_output), dim=2)

        # TODO check dimensions 
        output = self.linear_decoder(encoder_output) # [seq, batch, output_dim]

        if self.estimate_contact:
            return contact_output.transpose(0, 1), output.transpose(0, 1)

        return None, output.transpose(0, 1) # [batch, seq, output_dim]

        # return output.transpose(0, 1) # [batch, seq, output_dim]
        
class TransformerSceneEncoderModel(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim=1024, num_layers=4, num_heads=8, dropout=0.1, estimate_contact=False, temporal = True
    ):
        """
        input_dim: this is the dimension of the input
        ninp: 1024 this is the dimension of the hidden layer
        hidden_dim: same as ninp
        num_layers: the number of layers in transformer encoder and decoder. can be either 1 or 4
        """
        self.environment_enc = PointNet2SemSegSSGShape({'feat_dim': hidden_dim})
        #self.pointnet = PointNet(model_config['hidden_dim_scene'])
        
        self.temporal = temporal
        self.mid_dim = None
        if isinstance(input_dim, tuple):
            self.input_dim, self.mid_dim = input_dim

        self.hidden_dim = hidden_dim

        super(TransformerEncoderModel, self).__init__()
        self.model_type = "TransformerEncoder"

        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, num_heads, hidden_dim, dropout
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=LayerNorm(hidden_dim),
        )
        
        if self.temporal:
            self.time_encoder = torch.nn.LSTM(hidden_dim, hidden_dim, num_layers=1, bidirectional=False, batch_first=False)

        # Use Linear instead of Embedding for continuous valued input
        if self.mid_dim is not None:
            half_hidden_dim = int(hidden_dim/2)
            self.mid_encoder = nn.Linear(self.mid_dim, half_hidden_dim)
            self.input_encoder = nn.Linear(self.input_dim, half_hidden_dim)

        else:
            self.encoder = nn.Linear(input_dim, hidden_dim)

        self.hidden_dim = hidden_dim
        
        # foot fc 
        decode_dim = hidden_dim

        self.estimate_contact = estimate_contact
        if self.estimate_contact:
            self.contact_decoder = nn.Sequential(
                                nn.Linear(hidden_dim, 256),
                                nn.ReLU(),
                                nn.Linear(256, 2)
                )        
            decode_dim += 2

        self.linear_decoder = nn.Sequential(
                            nn.Linear(decode_dim, 256),
                            nn.ReLU(),
                            nn.Linear(256, output_dim)
            )
        
        self.init_weights()

    def init_weights(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


    def forward(self, src, input_pcs = None):
        # Encode the Scene point clouds 
        scene_feats, scene_global_feats = self.environment_enc(input_pcs.repeat(1, 1, 2)) #[64, 1280, 10000], [64, 1280]
        
        
        # Transformer expects src and tgt in format (len, batch_size, dim)
        src = src.transpose(0, 1) # by transpose, [seq, batch, ninp]
        if self.mid_dim is None:
            projected_src = self.encoder(src) * np.sqrt(self.hidden_dim) # why add np.sqrt? [seq, batch, hidden_dim]
        else:
            half_hidden_dim = int(self.hidden_dim/2)

            src_input, src_mid = src[...,:self.input_dim], src[...,self.input_dim:]
            projected_input_src = self.input_encoder(src_input)
            projected_mid_src = self.mid_encoder(src_mid)
            projected_src = torch.cat((projected_input_src, projected_mid_src),-1) * np.sqrt(self.hidden_dim)

        pos_encoded_src = self.pos_encoder(projected_src) # [seq, batch, hidden_dim]
        encoder_output = self.transformer_encoder(pos_encoded_src) # [seq, batch, ninp] encoder output
        #40,batch,1280
        if self.temporal:
            encoder_output, _ = self.time_encoder(encoder_output)
            #encoder_output = self.temporal_attn(encoder_output)

        if self.estimate_contact:
            contact_output = self.contact_decoder(encoder_output) # [seq, batch, 18]
            encoder_output = torch.cat((encoder_output, contact_output), dim=2)

        # TODO check dimensions 
        output = self.linear_decoder(encoder_output) # [seq, batch, output_dim]

        if self.estimate_contact:
            return contact_output.transpose(0, 1), output.transpose(0, 1)

        return None, output.transpose(0, 1) # [batch, seq, output_dim]

        # return output.transpose(0, 1) # [batch, seq, output_dim]
        
class Dinov2Backbone(nn.Module):
    def __init__(self, name='dinov2_vitb14', pretrained=False):
        super().__init__()
        self.name = name
        self.encoder = torch.hub.load('facebookresearch/dinov2', self.name, pretrained=pretrained)
        self.patch_size = self.encoder.patch_size
        self.embed_dim = self.encoder.embed_dim

    def forward(self, x):
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs,3,w,h]
        Return:
            - y: torch.Tensor of shape [bs,k,d] - image in patchified mode
        """
        assert len(x.shape) == 4
        y = self.encoder.get_intermediate_layers(x)[0] # ViT-L+896x896: [bs,4096,1024] - [bs,nb_patches,emb]
        return y



'''
    This implementation is borrowed from GIMO, used as environment embeddings
'''
class FPModule(nn.Module):
    def __init__(self):
        super(FPModule, self).__init__()

    # B x N x 3, B x M X 3, B x F x M
    # output: B x F x N
    def forward(self, unknown, known, known_feats):
        dist, idx = pointnet2_utils.three_nn(unknown, known)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm

        interpolated_feats = pointnet2_utils.three_interpolate(
            known_feats, idx, weight
        )

        new_features = interpolated_feats
        new_features = new_features.unsqueeze(-1)
        return new_features.squeeze(-1)


class PointNet2SemSegSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,
                radius=0.1,
                nsample=32,
                mlp=[3, 32, 32, 64],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 3, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, self.hparams['feat_dim'], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0])


class PointNet2SemSegSSGShape(PointNet2ClassificationSSG): # Used Backbone
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.2,
                nsample=64,
                mlp=[3, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 256, 256],
                use_xyz=True,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + 3, 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, self.hparams['feat_dim'], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )
        self.fc_layer2 = nn.Sequential(
            nn.Linear(256, self.hparams['feat_dim']),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        bottleneck_feats = l_features[-1].squeeze(-1)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0]), self.fc_layer2(bottleneck_feats)


class PointNet(nn.Module):
    def __init__(self, feat_dim):
        super(PointNet, self).__init__()

        self.conv1 = nn.Conv1d(feat_dim, feat_dim, 1)
        self.conv2 = nn.Conv1d(feat_dim, feat_dim, 1)
        self.conv3 = nn.Conv1d(feat_dim, feat_dim, 1)

        self.bn1 = nn.BatchNorm1d(feat_dim)
        self.bn2 = nn.BatchNorm1d(feat_dim)
        self.bn3 = nn.BatchNorm1d(feat_dim)

    # B x 2F x N
    # output: B x F
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=-1)[0]
        return x

# class TemporalAttention(nn.Module):
#     def __init__(self, in_dim=1280, out_dim=1280, hidden_dim=512, num_layers=6, num_heads=4, residual=False):
#         super(TemporalAttention, self).__init__()
#         self.hdim = hidden_dim
#         self.out_dim = out_dim
#         self.residual = residual
#         # self.l1 = nn.Linear(in_dim, hidden_dim)
#         # self.l2 = nn.Linear(hidden_dim, out_dim)

#         #self.pos_embedding = PositionalEncoding(hidden_dim, dropout=0.1)
#         TranLayer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=1024,
#                                                dropout=0.1, activation='gelu')
#         self.trans = nn.TransformerEncoder(TranLayer, num_layers=num_layers)
        
#         # nn.init.xavier_uniform_(self.l1.weight, gain=0.01)
#         # nn.init.xavier_uniform_(self.l2.weight, gain=0.01)

#     def forward(self, x):
#         #x = x.permute(1,0,2)  # (b,t,c) -> (t,b,c)

#         #h = self.l1(x)
#         #h = self.pos_embedding(h)
#         x = self.trans(x)
#         #h = self.l2(h)

#         if self.residual:
#             x = x[..., :self.out_dim] + x
#         else:
#             x = x
#         x = x.permute(1,0,2)

#         return x


if __name__ == '__main__':
    '''
        Debug script for Environment encoder
    '''
    scene_pointnet = PointNet2SemSegSSGShape({'feat_dim': 1280}).cuda()
    x = torch.randn((1, 200000, 3)) 
    x = x.repeat((1, 1, 2)).cuda() 
    print(x.shape) 
    f1, f2 = scene_pointnet(x) 
    print(f1.shape, f2.shape) 
