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

# import pywt

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
        else:
            self.input_dim = input_dim

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
            self.temporal_encoder = torch.nn.LSTM(self.input_dim, hidden_dim, num_layers=1, bidirectional=False, batch_first=False)

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
        src = src.transpose(0, 1) # by transpose, [seq, batch, ninp]
        
        # Transformer expects src and tgt in format (len, batch_size, dim)
        if self.mid_dim is None:
            projected_src = self.encoder(src) * np.sqrt(self.hidden_dim) # why add np.sqrt? [seq, batch, hidden_dim]
        else:
            src_input, src_mid = src[...,:self.input_dim], src[...,self.input_dim:]
            projected_input_src = self.input_encoder(src_input)
            projected_mid_src = self.mid_encoder(src_mid)
            projected_src = torch.cat((projected_input_src, projected_mid_src),-1) * np.sqrt(self.hidden_dim)

        pos_encoded_src = self.pos_encoder(projected_src) # [seq, batch, hidden_dim]
        encoder_output = self.transformer_encoder(pos_encoded_src) # [seq, batch, ninp] encoder output

        if self.estimate_contact:
            contact_output = self.contact_decoder(encoder_output) # [seq, batch, 18]
            encoder_output = torch.cat((encoder_output, contact_output), dim=2)

        # TODO check dimensions 
        output = self.linear_decoder(encoder_output) # [seq, batch, output_dim]

        if self.estimate_contact:
            return contact_output.transpose(0, 1), output.transpose(0, 1)

        return None, output.transpose(0, 1) # [batch, seq, output_dim]
    
class WaveletTransformerEncoderModel(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim=1024, num_layers=4, num_heads=8, dropout=0.1, estimate_contact=False, temporal = True, m_num = 3
    ):
        
        self.temporal = temporal
        self.mid_dim = None
        if isinstance(input_dim, tuple):
            self.input_dim, self.mid_dim = input_dim
        else:
            self.input_dim = input_dim

        self.m_num = m_num
        self.hidden_dim = hidden_dim

        super(WaveletTransformerEncoderModel, self).__init__()
        self.model_type = "WaveletTransformerEncoderModel"

        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, num_heads, hidden_dim, dropout
        )
        hilo_encoder_layer = TransformerEncoderLayer(
            hidden_dim * 2 , num_heads * 2, hidden_dim * 2, dropout
        )
        
        self.transformer_encoder_lf = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers - 1,
            norm=LayerNorm(hidden_dim),
        )
        
        self.transformer_encoder_hf = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers - 1,
            norm=LayerNorm(hidden_dim),
        )
        
        self.transformer_encoder_hilo = TransformerEncoder(
            encoder_layer=hilo_encoder_layer,
            num_layers=1,
            norm=LayerNorm(hidden_dim * 2),
        )
        
        # Use Linear instead of Embedding for continuous valued input
        if self.mid_dim is not None:
            half_hidden_dim = int(hidden_dim/2)
            self.mid_encoder = nn.Linear(self.mid_dim * 2, half_hidden_dim)
            self.input_encoder = nn.Linear(self.input_dim * 2, half_hidden_dim )

        else:
            self.encoder = nn.Linear(input_dim, hidden_dim)

        self.hidden_dim = hidden_dim * 2
        
        # foot fc 
        decode_dim = hidden_dim * 2

        self.estimate_contact = estimate_contact
        if self.estimate_contact:
            self.contact_decoder = nn.Sequential(
                                nn.Linear(self.hidden_dim, 256),
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

    def forward(self, src_hf, src_lf):
        src_hf = src_hf.transpose(0, 1) # by transpose, [seq, batch, ninp]
        src_lf = src_lf.transpose(0, 1) # by transpose, [seq, batch, ninp]
        
        # Transformer expects src and tgt in format (len, batch_size, dim)
        if self.mid_dim is None:
            projected_src_lf = self.encoder(src_lf) * np.sqrt(self.hidden_dim)
            projected_src_hf = self.encoder(src_hf) * np.sqrt(self.hidden_dim)
            
        else: # for body
            src_input_hf= torch.cat([src_hf[...,:self.input_dim], src_hf[...,self.input_dim + self.mid_dim : (self.input_dim * 2 + self.mid_dim)]], dim = -1)
            src_mid_hf = torch.cat([src_hf[...,self.input_dim: self.input_dim + self.mid_dim], src_hf[..., (self.input_dim * 2 + self.mid_dim):]], dim = -1)
            projected_input_src_hf = self.input_encoder(src_input_hf)
            projected_mid_src_hf = self.mid_encoder(src_mid_hf)
            projected_src_hf = torch.cat((projected_input_src_hf, projected_mid_src_hf),-1) * np.sqrt(self.hidden_dim)
            
            src_input_lf= torch.cat([src_lf[...,:self.input_dim], src_lf[...,self.input_dim + self.mid_dim : (self.input_dim * 2 + self.mid_dim)]], dim = -1)
            src_mid_lf = torch.cat([src_lf[...,self.input_dim: self.input_dim + self.mid_dim], src_lf[..., (self.input_dim * 2 + self.mid_dim):]], dim = -1)
            projected_input_src_lf = self.input_encoder(src_input_lf)
            projected_mid_src_lf = self.mid_encoder(src_mid_lf)
            projected_src_lf = torch.cat((projected_input_src_lf, projected_mid_src_lf),-1) * np.sqrt(self.hidden_dim)

        pos_encoded_src_hf = self.pos_encoder(projected_src_hf) # [seq, batch, hidden_dim]
        pos_encoded_src_lf = self.pos_encoder(projected_src_lf)
        
        encoder_output_hf = self.transformer_encoder_hf(pos_encoded_src_hf) # [seq, batch, ninp] encoder output
        encoder_output_lf = self.transformer_encoder_lf(pos_encoded_src_lf)
        
        encoder_output_hilo = torch.cat((encoder_output_hf, encoder_output_lf), dim=2) # [seq, batch, 2*hidden_dim]
        encoder_output = self.transformer_encoder_hilo(encoder_output_hilo) # [seq, batch, ninp] encoder output
        

        if self.estimate_contact:
            contact_output = self.contact_decoder(encoder_output) # [seq, batch, 18]
            encoder_output = torch.cat((encoder_output, contact_output), dim=2)

        # TODO check dimensions 
        output = self.linear_decoder(encoder_output) # [seq, batch, output_dim]

        if self.estimate_contact:
            return contact_output.transpose(0, 1), output.transpose(0, 1)

        return None, output.transpose(0, 1) # [batch, seq, output_dim]



    
    

    
class GeometryTransformerEncoderModel(nn.Module):
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
            src_input, src_mid = src[...,:self.input_dim], src[...,self.input_dim:]
            projected_input_src = self.input_encoder(src_input)
            projected_mid_src = self.mid_encoder(src_mid)
            projected_src = torch.cat((projected_input_src, projected_mid_src),-1) * np.sqrt(self.hidden_dim)

        pos_encoded_src = self.pos_encoder(projected_src) # [seq, batch, hidden_dim]
        encoder_output = self.transformer_encoder(pos_encoded_src) # [seq, batch, ninp] encoder output
        
        if self.temporal:
            encoder_output, _ = self.time_encoder(encoder_output)

        if self.estimate_contact:
            contact_output = self.contact_decoder(encoder_output) # [seq, batch, 18]
            encoder_output = torch.cat((encoder_output, contact_output), dim=2)

        # TODO check dimensions 
        output = self.linear_decoder(encoder_output) # [seq, batch, output_dim]

        if self.estimate_contact:
            return contact_output.transpose(0, 1), output.transpose(0, 1)

        return None, output.transpose(0, 1) # [batch, seq, output_dim]


# class WaveletTransformerEncoderModel(nn.Module):
#     def __init__(
#         self, input_dim, output_dim, hidden_dim=1024, num_layers=4, num_heads=8, dropout=0.1, estimate_contact=False, temporal = True, m_num = 3
#     ):
#         """
#         input_dim: this is the dimension of the input
#         ninp: 1024 this is the dimension of the hidden layer
#         hidden_dim: same as ninp
#         num_layers: the number of layers in transformer encoder and decoder. can be either 1 or 4

#         """
#         self.temporal = temporal
#         self.m_num = m_num
#         self.mid_dim = None
#         if isinstance(input_dim, tuple):
#             self.input_dim, self.mid_dim = input_dim

#         self.hidden_dim = hidden_dim

#         super(WaveletTransformerEncoderModel, self).__init__()
#         self.model_type = "TransformerEncoder"

#         self.pos_encoder = PositionalEncoding(hidden_dim)
#         encoder_layer = TransformerEncoderLayer(
#             hidden_dim, num_heads, hidden_dim, dropout
#         )
#         self.transformer_encoder = TransformerEncoder(
#             encoder_layer=encoder_layer,
#             num_layers=num_layers,
#             norm=LayerNorm(hidden_dim),
#         )
        
#         self.waveletdecomp = WaveletEmbedding(d_channel = 40, swt = True, requires_grad=True)
#         self.waveletrecon = WaveletEmbedding(d_channel = 40, swt = False, requires_grad=True, kernel_size=2)
#         self.m = 2
        
#         # Use Linear instead of Embedding for continuous valued input
#         if self.mid_dim is not None:
#             half_hidden_dim = int(hidden_dim/2)
#             self.mid_encoder = nn.Linear(self.mid_dim, half_hidden_dim)
#             self.input_encoder = nn.Linear(self.input_dim, half_hidden_dim)

#         else:
#             self.encoder = nn.Linear(input_dim, hidden_dim)

#         self.hidden_dim = hidden_dim
        
#         # foot fc 
#         decode_dim = hidden_dim

#         self.estimate_contact = estimate_contact
        
#         if self.estimate_contact:
#             self.contact_decoder = nn.Sequential(
#                                 nn.Linear(hidden_dim, 256),
#                                 nn.ReLU(),
#                                 nn.Linear(256, 2)
#                 )        
#             decode_dim += 2

#         self.linear_decoder = nn.Sequential(
#                             nn.Linear(decode_dim, 256),
#                             nn.ReLU(),
#                             nn.Linear(256, output_dim)
#             )
        
#         # self.Intra_SWT = nn.Sequential(
#         #     nn.Linear(hidden_dim*3, hidden_dim),
#         #     nn.LayerNorm(hidden_dim),
#         #     nn.GELU(),
#         #     nn.Linear(hidden_dim, hidden_dim),
#         # )
        
#         # self.Intra_SWT_Reverse = nn.Sequential(
#         #     nn.Linear(hidden_dim, 128),
#         #     nn.LayerNorm(128),
#         #     nn.GELU(),
#         #     nn.Linear(128, 22 * 3 * 2),
#         # )
        
        
#         self.init_weights()

#     def init_weights(self):
#         """Initiate parameters in the transformer model."""
#         for p in self.parameters():
#             if p.dim() > 1:
#                 xavier_uniform_(p)


#     def forward(self, src):
#         # Transformer expects src and tgt in format (len, batch_size, dim)
        
#         src = self.waveletdecomp(src)
        
#         src = src.transpose(0, 1) # by transpose, [seq, batch, m+1, ninp] 
        
#         if self.mid_dim is None:
#             projected_src = self.encoder(src) * np.sqrt(self.hidden_dim) # why add np.sqrt? [seq, batch, hidden_dim]
#         else:
#             src_input, src_mid = src[...,:self.input_dim], src[...,self.input_dim:]
#             projected_input_src = self.input_encoder(src_input)
#             projected_mid_src = self.mid_encoder(src_mid)
#             projected_src = torch.cat((projected_input_src, projected_mid_src),-1) * np.sqrt(self.hidden_dim)

#         encoder_output_list = []
#         contact_output_list = []
#         for i in range(self.m_num): # For each freq, intra SWT
#             pos_encoded_src = self.pos_encoder(projected_src[..., i, :]) # [seq, batch, hidden_dim] #projected_src[..., i, :]
#             encoder_output = self.transformer_encoder(pos_encoded_src) # [seq, batch, ninp] encoder output

#             if self.estimate_contact:
#                 contact_output = self.contact_decoder(encoder_output) # [seq, batch, 18]
#                 encoder_output = torch.cat((encoder_output, contact_output), dim=2)
#                 contact_output_list.append(contact_output.unsqueeze(-2)) # [seq, batch, 18]
                
#             encoder_output_list.append(encoder_output.unsqueeze(-2)) # [seq, batch, output_dim]
            
#         encoder_output = torch.cat(encoder_output_list, dim = -2) # inter SWT
        
#         output = self.linear_decoder(encoder_output).transpose(0, 1)
        
#         output = self.waveletrecon(output) # [seq, batch, hidden_dim]
        
#         if self.estimate_contact:
#             contact_output = torch.cat(contact_output_list, dim = -2).transpose(0, 1)
            
#             contact_output = self.waveletrecon(contact_output) # [seq, batch, 2]
            
#             return contact_output, output
#         else:
#             return None, output # [batch, seq, output_dim]
        
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
        """
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

class WaveletEmbedding(nn.Module):
    def __init__(self, d_channel=16, swt=True, requires_grad=True, wv='db1', m=2,
                 kernel_size=None):
        super().__init__()

        self.swt = swt
        self.d_channel = d_channel
        self.m = m  # Number of decomposition levels of detailed coefficients
        
        if kernel_size is None:
            self.wavelet = pywt.Wavelet(wv)
            if self.swt:
                h0 = torch.tensor(self.wavelet.dec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.dec_hi[::-1], dtype=torch.float32)
            else:
                h0 = torch.tensor(self.wavelet.rec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.rec_hi[::-1], dtype=torch.float32)
            self.h0 = nn.Parameter(torch.tile(h0[None, None, :], [self.d_channel, 1, 1]), requires_grad=requires_grad)
            self.h1 = nn.Parameter(torch.tile(h1[None, None, :], [self.d_channel, 1, 1]), requires_grad=requires_grad)
            self.kernel_size = self.h0.shape[-1]
        else:
            self.kernel_size = kernel_size
            self.h0 = nn.Parameter(torch.Tensor(self.d_channel, 1, self.kernel_size), requires_grad=requires_grad)
            self.h1 = nn.Parameter(torch.Tensor(self.d_channel, 1, self.kernel_size), requires_grad=requires_grad)
            nn.init.xavier_uniform_(self.h0)
            nn.init.xavier_uniform_(self.h1)
        
            with torch.no_grad():
                self.h0.data = self.h0.data / torch.norm(self.h0.data, dim=-1, keepdim=True)
                self.h1.data = self.h1.data / torch.norm(self.h1.data, dim=-1, keepdim=True)

    def forward(self, x):
        if self.swt:
            coeffs = self.swt_decomposition(x, self.h0, self.h1, self.m, self.kernel_size)
        else:
            coeffs = self.swt_reconstruction(x, self.h0, self.h1, self.m, self.kernel_size)
        return coeffs

    def swt_decomposition(self, x, h0, h1, depth, kernel_size):
        approx_coeffs = x
        coeffs = []
        dilation = 1
        for _ in range(depth):
            padding = dilation * (kernel_size - 1)
            padding_r = (kernel_size * dilation) // 2
            pad = (padding - padding_r, padding_r)
            approx_coeffs_pad = F.pad(approx_coeffs, pad, "circular")
            detail_coeff = F.conv1d(approx_coeffs_pad, h1, dilation=dilation, groups=x.shape[1]) # Linear projection
            approx_coeffs = F.conv1d(approx_coeffs_pad, h0, dilation=dilation, groups=x.shape[1]) # Apply Filter
            coeffs.append(detail_coeff)
            dilation *= 2
        coeffs.append(approx_coeffs)

        return torch.stack(list(reversed(coeffs)), -2)

    def swt_reconstruction(self, coeffs, g0, g1, m, kernel_size):
        dilation = 2 ** (m - 1)
        approx_coeff = coeffs[:,:,0,:]
        detail_coeffs = coeffs[:,:,1:,:]
        
        for i in range(m):
            detail_coeff = detail_coeffs[:,:,i,:]
            padding = dilation * (kernel_size - 1)
            padding_l = (dilation * kernel_size) // 2
            pad = (padding_l, padding - padding_l)
            approx_coeff_pad = F.pad(approx_coeff, pad, "circular")
            detail_coeff_pad = F.pad(detail_coeff, pad, "circular")
            
            y = F.conv1d(approx_coeff_pad, g0, groups=approx_coeff.shape[1], dilation=dilation) + \
                F.conv1d(detail_coeff_pad, g1, groups=detail_coeff.shape[1], dilation=dilation)
            approx_coeff = y / 2
            dilation //= 2
            
        return approx_coeff
    
class GeomAttentionLayer(nn.Module):
    def __init__(self, attention, d_model,
                 requires_grad=True, wv='db2', m=2, kernel_size=None,
                 d_channel=None, geomattn_dropout=0.5,):
        super(GeomAttentionLayer, self).__init__()

        self.d_channel = d_channel
        self.inner_attention = attention
        
        self.swt = WaveletEmbedding(d_channel=self.d_channel, swt=True, requires_grad=requires_grad, wv=wv, m=m, kernel_size=kernel_size)
        self.query_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.key_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.value_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(geomattn_dropout)
        )
        self.out_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            WaveletEmbedding(d_channel=self.d_channel, swt=False, requires_grad=requires_grad, wv=wv, m=m, kernel_size=kernel_size),
        )
        
    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        queries = self.swt(queries)
        keys = self.swt(keys)
        values = self.swt(values)

        queries = self.query_projection(queries).permute(0,3,2,1)
        keys = self.key_projection(keys).permute(0,3,2,1)
        values = self.value_projection(values).permute(0,3,2,1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
        )

        out = self.out_projection(out.permute(0,3,2,1))

        return out, attn


class GeomAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, 
                 output_attention=False,
                 alpha=1.,):
        super(GeomAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        self.alpha = alpha 

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, _ = values.shape
        scale = self.scale or 1. / sqrt(E)

        dot_product = torch.einsum("blhe,bshe->bhls", queries, keys)

        queries_norm2 = torch.sum(queries**2, dim=-1)
        keys_norm2 = torch.sum(keys**2, dim=-1)
        queries_norm2 = queries_norm2.permute(0, 2, 1).unsqueeze(-1)         # (B, H, L, 1)
        keys_norm2 = keys_norm2.permute(0, 2, 1).unsqueeze(-2)               # (B, H, 1, S)
        wedge_norm2 = queries_norm2 * keys_norm2 - dot_product ** 2          # (B, H, L, S)
        wedge_norm2 = F.relu(wedge_norm2)
        wedge_norm = torch.sqrt(wedge_norm2 + 1e-8)

        scores = (1 - self.alpha) * dot_product + self.alpha * wedge_norm
        scores = scores * scale

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = torch.tril(torch.ones(L, S)).to(scores.device)
            scores.masked_fill_(attn_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))

        A = self.dropout(torch.softmax(scores, dim=-1)) 

        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous()
        else:
            return (V.contiguous(), scores.abs().mean())  
    
    


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

# if __name__ == '__main__':
#     '''
#         Debug script for Wavelet encoder
#     '''
#     waveletdecomp = WaveletEmbedding(d_channel = 40, swt = True)
#     waveletrecon = WaveletEmbedding(d_channel = 40, swt = False)
    
#     x = torch.randn((256, 40, 31))  # [b, c, h]
#     x_decomp = waveletdecomp(x)
#     x_recon = waveletrecon(x_decomp)
#     print(x_recon.shape)