# (FairMotion) Copyright (c) Facebook, Inc. and its affiliates.

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import LayerNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn.init import xavier_uniform_

#from functions import GPTimeNoiseTBD

import random
from IPython import embed

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
        self, input_dim, output_dim, hidden_dim=1024, num_layers=4, num_heads=8, dropout=0.1, estimate_contact=False
    ):
        """
        input_dim: this is the dimension of the input
        ninp: 1024 this is the dimension of the hidden layer
        hidden_dim: same as ninp
        num_layers: the number of layers in transformer encoder and decoder. can be either 1 or 4

        """
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

        # self.estimate_foot = estimate_foot
        # if self.estimate_foot:
            # self.foot_decoder = nn.Sequential(
            #                     nn.Linear(hidden_dim, 256),
            #                     nn.ReLU(),
            #                     nn.Linear(256, 6)
            #     )        
            # decode_dim += 6

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

        if self.estimate_contact:
            contact_output = self.contact_decoder(encoder_output) # [seq, batch, 18]
            encoder_output = torch.cat((encoder_output, contact_output), dim=2)

        # TODO check dimensions 
        output = self.linear_decoder(encoder_output) # [seq, batch, output_dim]

        if self.estimate_contact:
            return contact_output.transpose(0, 1), output.transpose(0, 1)

        return None, output.transpose(0, 1) # [batch, seq, output_dim]

        # return output.transpose(0, 1) # [batch, seq, output_dim]

# class CrossAttention(nn.Module):
#     def __init__(self, hidden_dim, num_heads):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

#     def forward(self, query, context):
#         # query: [B, T, hidden_dim]
#         # context: [B, T', hidden_dim]
#         attn_output, _ = self.attn(query=query, key=context, value=context)
#         return attn_output

class TransformerEncoderModel_Uncertain(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim=1024, num_layers=4, num_heads=8, dropout=0.1, estimate_contact=False
    ):
        """
        input_dim: this is the dimension of the input
        ninp: 1024 this is the dimension of the hidden layer
        hidden_dim: same as ninp
        num_layers: the number of layers in transformer encoder and decoder. can be either 1 or 4

        """
        self.mid_dim = None
        if isinstance(input_dim, tuple):
            self.input_dim, self.mid_dim = input_dim

        self.hidden_dim = hidden_dim

        super(TransformerEncoderModel_Uncertain, self).__init__()
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

        self.shared_decoder = nn.Sequential(
                                nn.Linear(decode_dim, 256),
                                nn.ReLU(),
                            )
        self.pose_mean_head   = nn.Linear(256, output_dim)  # θ^e
        self.pose_logvar_head = nn.Linear(256, output_dim)  # log σ^2
        
        self.cross_attn = CrossAttention(hidden_dim, num_heads)
        
        self.init_weights()

    def init_weights(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
    
    def forward(self, src, context=None, sample=True):
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

        if context is not None:
            # context: [bs, 512]
            context_proj = context.unsqueeze(1).repeat(1, encoder_output.size(0), 1).permute(1, 0, 2)  # [seq, batch, 1280]
            encoder_output = self.cross_attn(encoder_output, context_proj)
        # contact 分支
        contact_output = None
        dec_input = encoder_output
        if self.estimate_contact:
            contact_output = self.contact_decoder(encoder_output)   # [T,B,2]
            dec_input = torch.cat((encoder_output, contact_output), dim=2)  # [T,B,H+2]
        
        # 共享干路 + 双头
        dec_feat = self.shared_decoder(dec_input)                   # [T,B,256]
        mean     = self.pose_mean_head(dec_feat)                    # [T,B,output_dim]
        logvar   = self.pose_logvar_head(dec_feat)                  # [T,B,output_dim]

        # 数值稳定：裁剪 logvar
        logvar = torch.clamp(logvar, min=-10.0, max=6.0)            # σ^2 ∈ [e^-10, e^6]

        if sample:
            std  = torch.exp(0.5 * logvar)
            
            # 1. Naive Sampling
            eps  = torch.randn_like(std)
            theta = mean + std * eps
            
            # 2. AR(1) sampling
            # alpha = 0.95  # 相关系数，越大越平滑
            # alpha_new = math.sqrt(1 - alpha**2)
            # eps = torch.randn_like(std)
            # for t in range(1, eps.size(0)):  # [B, T, D]
            #     eps[t, ...] = alpha * eps[t-1, ...] + alpha_new * eps[t, ...]
            # theta = mean + std * eps
            
            # 3. RBF-GP Time noise 
            # tau = 0.3                                   # 采样温度，先小后大更稳
            # T, B, D = mean.shape
            # # 用法：
            # gp = GPTimeNoiseTBD(T, lengthscale=5.0, jitter=1e-6, device=mean.device, dtype=mean.dtype)  # 可缓存
            # eps = gp.sample_eps(B, D)  # [T,B,D]
            # theta = mean + tau * std * eps

        else:
            theta = mean

        # 还原回 [B,T,*]
        mean   = mean.transpose(0, 1)
        logvar = logvar.transpose(0, 1)
        theta  = theta.transpose(0, 1)
        if self.estimate_contact:
            contact_output = contact_output.transpose(0, 1)  # [B,T,2]

        return contact_output, mean, logvar, theta


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, attn_drop=0.1, proj_drop=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=attn_drop, batch_first=False
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 残差门控，初始为 0，确保一开始几乎等价于不加 cross-attn
        self.gate = nn.Parameter(torch.zeros(1))
        # self.gate = 1.
        print(f"######## GATE VALUE: {self.gate} ########")

        # 关键：把输出线性层置零初始化，进一步保证初期稳定
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, context):
        # x: [B,T,E], context: [B,Tc,E] 或 [B,1,E]
        q = self.norm_q(x)
        kv = self.norm_kv(context)
        out, _ = self.attn(query=q, key=kv, value=kv, need_weights=False)
        out = self.proj_drop(self.proj(out))
        y = x + self.gate * out
        return y


class TransformerSceneEncoderModel(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim=1024, num_layers=4, num_heads=8, dropout=0.1, estimate_contact=False
    ):
        """
        input_dim: this is the dimension of the input
        ninp: 1024 this is the dimension of the hidden layer
        hidden_dim: same as ninp
        num_layers: the number of layers in transformer encoder and decoder. can be either 1 or 4

        """
        self.mid_dim = None
        if isinstance(input_dim, tuple):
            self.input_dim, self.mid_dim = input_dim

        self.hidden_dim = hidden_dim

        super(TransformerSceneEncoderModel, self).__init__()
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
        
        self.cross_attn = CrossAttention(hidden_dim, num_heads)

        
        
        self.init_weights()

    def init_weights(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


    def forward(self, src, context=None):
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

        if context is not None:
            # context: [bs, 512]
            context_proj = context.unsqueeze(1).repeat(1, encoder_output.size(0), 1).permute(1, 0, 2)  # [seq, batch, 1280]
            encoder_output = self.cross_attn(encoder_output, context_proj)

        if self.estimate_contact:
            contact_output = self.contact_decoder(encoder_output) # [seq, batch, 18]
            encoder_output = torch.cat((encoder_output, contact_output), dim=2)

        # TODO check dimensions 
        output = self.linear_decoder(encoder_output) # [seq, batch, output_dim]

        if self.estimate_contact:
            return contact_output.transpose(0, 1), output.transpose(0, 1)

        return None, output.transpose(0, 1) # [batch, seq, output_dim]


# ---------------------------
# FiLM-style 调制模块（稳）
# ---------------------------
class FiLMMod(nn.Module):
    """
    FiLM modulation for [S, B, E] with global context.
    - context 可为:
        [B, C]                全局向量（推荐）
        [S_c, B, C] 或 [B, S_c, C]  也可，内部会先做 mean pool -> [B, C]
    y = x + gate * (tanh(gamma) * LN(x) + beta)
    """
    def __init__(self, hidden_dim: int, context_dim: int, mlp_hidden: int = 256,
                 dropout: float = 0.1, use_gate: bool = True, init_gamma_zero: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_hidden, 2 * hidden_dim)  # -> [gamma, beta]
        )
        self.use_gate = use_gate
        if use_gate:
            self.gate = nn.Parameter(torch.zeros(1))  # 初始不扰动主干
            print(self.gate)

        # 零初始化最后一层：gamma、beta 初值全 0（更稳）
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

        # 如果想让 gamma 初始为 1，可把上面置零改成下面这段：
        if not init_gamma_zero:
            with torch.no_grad():
                # bias 前半是 gamma，后半是 beta
                self.mlp[-1].bias[:hidden_dim].fill_(1.0)
                self.mlp[-1].bias[hidden_dim:].zero_()

    @staticmethod
    def _to_global_context(context: torch.Tensor) -> torch.Tensor:
        # 统一成 [B, C]
        if context.dim() == 2:
            return context  # [B, C]
        elif context.dim() == 3:
            # 可能是 [S_c, B, C] 或 [B, S_c, C]
            if context.shape[0] == context.shape[1]:  # 罕见歧义，默认当 [S_c, B, C]
                # 当 [S_c, B, C]：mean over S_c
                return context.mean(dim=0)
            if context.shape[0] < context.shape[1]:
                # 多数情况 [S_c, B, C]
                return context.mean(dim=0)
            else:
                # [B, S_c, C]
                return context.mean(dim=1)
        else:
            raise ValueError("context must be [B, C], [S_c, B, C], or [B, S_c, C]")

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x: [S, B, E]
        context: [B, C] or seq pooled into [B, C]
        """
        S, B, E = x.shape
        x_ln = self.norm(x)  # [S, B, E]

        gctx = self._to_global_context(context)     # [B, C]
        gb = self.mlp(gctx)                         # [B, 2E]
        gamma, beta = gb.split(E, dim=-1)           # [B, E], [B, E]
        gamma = torch.tanh(gamma)                   # 限幅，防止尺度爆

        gamma = gamma.unsqueeze(0).expand(S, B, E)  # [S, B, E]
        beta  = beta.unsqueeze(0).expand(S, B, E)   # [S, B, E]
        mod = gamma * x_ln + beta                   # [S, B, E]

        if self.use_gate:
            return x + self.gate * mod
        else:
            return x + mod



class TransformerSceneFiLMModel(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim=1024, num_layers=4, num_heads=8, dropout=0.1, estimate_contact=False,
        context_dim: int | None = None,
        film_mlp_hidden: int = 256,
        film_dropout: float = 0.1,
        film_use_gate: bool = True,
        film_init_gamma_zero: bool = True,

    ):
        """
        input_dim: this is the dimension of the input
        ninp: 1024 this is the dimension of the hidden layer
        hidden_dim: same as ninp
        num_layers: the number of layers in transformer encoder and decoder. can be either 1 or 4

        """
        self.mid_dim = None
        if isinstance(input_dim, tuple):
            self.input_dim, self.mid_dim = input_dim

        self.hidden_dim = hidden_dim

        super(TransformerSceneFiLMModel, self).__init__()
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
        
        self.film = FiLMMod(
                hidden_dim=hidden_dim,
                context_dim=context_dim,
                mlp_hidden=film_mlp_hidden,
                dropout=film_dropout,
                use_gate=film_use_gate,
                init_gamma_zero=film_init_gamma_zero
            )


        
        
        self.init_weights()

    def init_weights(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


    def forward(self, src, context=None):
        # Transformer expects src and tgt in format (len, batch_size, dim)
        src = src.transpose(0, 1) # by transpose, [seq, batch, ninp]
        if self.mid_dim is None:
            projected_src = self.encoder(src) * np.sqrt(self.hidden_dim) # why add np.sqrt? [seq, batch, hidden_dim]
        else:
            half_hidden_dim = self.hidden_dim // 2
            src_input, src_mid = src[...,:self.input_dim], src[...,self.input_dim:]
            projected_input_src = self.input_encoder(src_input)
            projected_mid_src = self.mid_encoder(src_mid)
            projected_src = torch.cat((projected_input_src, projected_mid_src),-1) * np.sqrt(self.hidden_dim)

        x = self.pos_encoder(projected_src) # [seq, batch, hidden_dim]
        x = self.transformer_encoder(x) # [seq, batch, ninp] encoder output

        if context is not None:
            x = self.film(x, context)

        if self.estimate_contact:
            contact_output = self.contact_decoder(x) # [seq, batch, 18]
            x_dec = torch.cat((x, contact_output), dim=2)

        # TODO check dimensions 
        output = self.linear_decoder(x_dec) # [seq, batch, output_dim]

        if self.estimate_contact:
            return contact_output.transpose(0, 1), output.transpose(0, 1)

        return None, output.transpose(0, 1) # [batch, seq, output_dim]


class TransformerSceneFiLMModel_Uncertain(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim=1024, num_layers=4, num_heads=8, dropout=0.1, estimate_contact=False,
        context_dim: int | None = None,
        film_mlp_hidden: int = 256,
        film_dropout: float = 0.1,
        film_use_gate: bool = True,
        film_init_gamma_zero: bool = True,

    ):
        """
        input_dim: this is the dimension of the input
        ninp: 1024 this is the dimension of the hidden layer
        hidden_dim: same as ninp
        num_layers: the number of layers in transformer encoder and decoder. can be either 1 or 4

        """
        self.mid_dim = None
        if isinstance(input_dim, tuple):
            self.input_dim, self.mid_dim = input_dim

        self.hidden_dim = hidden_dim

        super(TransformerSceneFiLMModel_Uncertain, self).__init__()
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

        # self.linear_decoder = nn.Sequential(
        #                     nn.Linear(decode_dim, 256),
        #                     nn.ReLU(),
        #                     nn.Linear(256, output_dim)
        #     )
        
        self.shared_decoder = nn.Sequential(
                                nn.Linear(decode_dim, 256),
                                nn.ReLU(),
                            )
        self.pose_mean_head   = nn.Linear(256, output_dim)  # θ^e
        self.pose_logvar_head = nn.Linear(256, output_dim)  # log σ^2
        
        self.film = FiLMMod(
                hidden_dim=hidden_dim,
                context_dim=context_dim,
                mlp_hidden=film_mlp_hidden,
                dropout=film_dropout,
                use_gate=film_use_gate,
                init_gamma_zero=film_init_gamma_zero
            )


        
        
        self.init_weights()

    def init_weights(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


    def forward(self, src, context=None, sample=True):
        # Transformer expects src and tgt in format (len, batch_size, dim)
        src = src.transpose(0, 1) # by transpose, [seq, batch, ninp]
        if self.mid_dim is None:
            projected_src = self.encoder(src) * np.sqrt(self.hidden_dim) # why add np.sqrt? [seq, batch, hidden_dim]
        else:
            half_hidden_dim = self.hidden_dim // 2
            src_input, src_mid = src[...,:self.input_dim], src[...,self.input_dim:]
            projected_input_src = self.input_encoder(src_input)
            projected_mid_src = self.mid_encoder(src_mid)
            projected_src = torch.cat((projected_input_src, projected_mid_src),-1) * np.sqrt(self.hidden_dim)

        x = self.pos_encoder(projected_src) # [seq, batch, hidden_dim]
        x = self.transformer_encoder(x) # [seq, batch, ninp] encoder output

        if context is not None:
            x = self.film(x, context)

        contact_output = None
        if self.estimate_contact:
            contact_output = self.contact_decoder(x) # [seq, batch, 18]
            x_dec = torch.cat((x, contact_output), dim=2)

        # 共享干路 + 双头
        dec_feat = self.shared_decoder(x_dec)                   # [T,B,256]
        mean     = self.pose_mean_head(dec_feat)                    # [T,B,output_dim]
        logvar   = self.pose_logvar_head(dec_feat)                  # [T,B,output_dim]

        # 数值稳定：裁剪 logvar
        logvar = torch.clamp(logvar, min=-10.0, max=6.0)            # σ^2 ∈ [e^-10, e^6]

        if sample:
            std  = torch.exp(0.5 * logvar)
            
            # 1. Naive Sampling
            eps  = torch.randn_like(std)
            theta = mean + std * eps
            
        else:
            theta = mean

        # 还原回 [B,T,*]
        mean   = mean.transpose(0, 1)
        logvar = logvar.transpose(0, 1)
        theta  = theta.transpose(0, 1)
        if self.estimate_contact:
            contact_output = contact_output.transpose(0, 1)  # [B,T,2]

        return contact_output, mean, logvar, theta


# 假设你已有 PositionalEncoding / TransformerEncoder / TransformerEncoderLayer / FiLMMod
# from your_module import PositionalEncoding, TransformerEncoder, TransformerEncoderLayer, FiLMMod

class TransformerSceneFiLMModel_Uncertain_BiSmoother(nn.Module):
    """
    Bidirectional Smoothing 版：
    - 两路：前向(正序+因果mask) 和 反向(倒序+因果mask 再翻回正序)
    - 各自产生 (mean, logvar)，使用 Product-of-Experts (PoE) 做不确定性感知融合
    - contact 输出为两路平均（也可按需改成 PoE/门控）
    - 仍兼容 FiLM 场景条件
    """
    def __init__(
        self, input_dim, output_dim, hidden_dim=1024, num_layers=4, num_heads=8, dropout=0.1, estimate_contact=False,
        context_dim: int | None = None,
        film_mlp_hidden: int = 256,
        film_dropout: float = 0.1,
        film_use_gate: bool = True,
        film_init_gamma_zero: bool = True,
        logvar_min: float = -10.0,
        logvar_max: float = 6.0,
        use_causal_mask: bool = True,    # True: 前向只看过去；反向只看“未来”（倒序后变过去）
        fusion: str = "poe"              # 预留：目前实现 poe（推荐）。也可扩展 "gate"
    ):
        super().__init__()

        self.mid_dim = None
        if isinstance(input_dim, tuple):
            self.input_dim, self.mid_dim = input_dim

        self.hidden_dim = hidden_dim
        self.model_type = "TransformerEncoder"
        self.output_dim = output_dim
        self.estimate_contact = estimate_contact
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.use_causal_mask = use_causal_mask
        self.fusion = fusion

        # 编码干路（前/反向共用权重；你也可复制一套做不共享）
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_layers, norm=LayerNorm(hidden_dim)
        )

        # 连续值输入：Linear 投影
        if self.mid_dim is not None:
            half_hidden_dim = hidden_dim // 2
            self.input_encoder = nn.Linear(self.input_dim, half_hidden_dim)
            self.mid_encoder   = nn.Linear(self.mid_dim, half_hidden_dim)
        else:
            self.encoder = nn.Linear(self.input_dim, hidden_dim)

        # FiLM 场景条件
        self.film = FiLMMod(
            hidden_dim=hidden_dim, context_dim=context_dim,
            mlp_hidden=film_mlp_hidden, dropout=film_dropout,
            use_gate=film_use_gate, init_gamma_zero=film_init_gamma_zero
        )

        # contact（可选），作为解码时的附加特征
        decode_dim = hidden_dim
        if self.estimate_contact:
            self.contact_decoder = nn.Sequential(
                nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Linear(256, 2)
            )
            decode_dim += 2

        # 共享解码干路 + 不确定性双头（两路共用解码器参数；各自前向一遍）
        self.shared_decoder   = nn.Sequential(nn.Linear(decode_dim, 256), nn.ReLU())
        self.pose_mean_head   = nn.Linear(256, output_dim)   # μ
        self.pose_logvar_head = nn.Linear(256, output_dim)   # log σ^2

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    @staticmethod
    def _causal_mask(T, device):
        # True 表示不允许注意（上三角，不含主对角）
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    @staticmethod
    def _time_reverse(x):
        # 反转 time 维（[T,B,*]）
        return x.flip(0)

    def _encode_stream(self, src_TBD, context=None, mask=None):
        """
        单路编码：输入 [T,B,in] → [T,B,H]
        """
        if self.mid_dim is None:
            projected = self.encoder(src_TBD) * math.sqrt(self.hidden_dim)
        else:
            src_input, src_mid = src_TBD[..., :self.input_dim], src_TBD[..., self.input_dim:]
            half_hidden_dim = self.hidden_dim // 2
            proj_in  = self.input_encoder(src_input)
            proj_mid = self.mid_encoder(src_mid)
            projected = torch.cat((proj_in, proj_mid), dim=-1) * math.sqrt(self.hidden_dim)

        x = self.pos_encoder(projected)                    # [T,B,H]
        x = self.transformer_encoder(x, mask=mask)        # [T,B,H]
        if context is not None:
            x = self.film(x, context)                     # [T,B,H]
        return x

    def _decode_heads(self, x_TBH):
        """
        解码到 (mean, logvar)，必要时拼接 contact 作为附加特征
        """
        if self.estimate_contact:
            contact = self.contact_decoder(x_TBH)         # [T,B,2]
            x_dec   = torch.cat((x_TBH, contact), dim=2)  # [T,B,H+2]
        else:
            contact = None
            x_dec   = x_TBH

        feat = self.shared_decoder(x_dec)                 # [T,B,256]
        mean   = self.pose_mean_head(feat)                # [T,B,D]
        logvar = self.pose_logvar_head(feat)              # [T,B,D]
        logvar = torch.clamp(logvar, min=self.logvar_min, max=self.logvar_max)
        return mean, logvar, contact

    def _poe_fuse(self, mu_f, logv_f, mu_b, logv_b):
        """
        Product-of-Experts 融合（逐维）
        """
        # precision = exp(-logvar) = 1 / var
        prec_f = torch.exp(-logv_f)
        prec_b = torch.exp(-logv_b)
        prec_sum = prec_f + prec_b + 1e-8

        mu_fused = (prec_f * mu_f + prec_b * mu_b) / prec_sum
        logvar_fused = -torch.log(prec_sum)               # log(1 / (prec_f+prec_b))
        return mu_fused, logvar_fused

    def forward(self, src, context=None, sample=True):
        """
        src: [B,T,in]  →  返回 (contact[B,T,2]或None, mean[B,T,D], logvar[B,T,D], theta[B,T,D])
        """
        # [B,T,*] → [T,B,*]
        src_TBD = src.transpose(0, 1)
        T, B = src_TBD.size(0), src_TBD.size(1)
        mask = self._causal_mask(T, src_TBD.device) if self.use_causal_mask else None

        # ---- 前向流（正序） ----
        x_fwd = self._encode_stream(src_TBD, context=context, mask=mask)   # [T,B,H]
        mu_fwd, logv_fwd, contact_fwd = self._decode_heads(x_fwd)          # [T,B,D],[T,B,D],[T,B,2] or None

        # ---- 反向流（倒序→编码→再翻回正序）----
        src_rev = self._time_reverse(src_TBD)                               # [T,B,*]
        x_bwd_rev = self._encode_stream(src_rev, context=context, mask=mask)
        x_bwd = self._time_reverse(x_bwd_rev)                               # [T,B,H]
        mu_bwd, logv_bwd, contact_bwd = self._decode_heads(x_bwd)

        # ---- 融合（默认 PoE）----
        if self.fusion == "poe":
            mean_TBD, logvar_TBD = self._poe_fuse(mu_fwd, logv_fwd, mu_bwd, logv_bwd)
        else:
            # 预留门控方案（如精度软最大）；这里先退化为 PoE 以保证数值稳定
            mean_TBD, logvar_TBD = self._poe_fuse(mu_fwd, logv_fwd, mu_bwd, logv_bwd)

        # 采样或用均值
        if sample:
            std_TBD = torch.exp(0.5 * logvar_TBD)
            eps = torch.randn_like(std_TBD)
            theta_TBD = mean_TBD + std_TBD * eps
        else:
            theta_TBD = mean_TBD

        # contact 输出（两路平均；如需 PoE/门控，可自行替换）
        if self.estimate_contact:
            contact_TBD = 0.5 * (contact_fwd + contact_bwd)
        else:
            contact_TBD = None

        # 还原到 [B,T,*]
        mean   = mean_TBD.transpose(0, 1)
        logvar = logvar_TBD.transpose(0, 1)
        theta  = theta_TBD.transpose(0, 1)
        if contact_TBD is not None:
            contact = contact_TBD.transpose(0, 1)  # [B,T,2]
        else:
            contact = None

        return contact, mean, logvar, theta


class FiLMMod_(nn.Module):
    """
    Generic FiLM modulation over [T,B,C] features using context.
    - context can be [B, Cc] (global) or [B, T, Cc] (timewise). When timewise, we align along T.
    - y = (1 + gamma) * x + beta; optional gate to mix with identity.
    """
    def __init__(
        self,
        hidden_dim: int,
        context_dim,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
        use_gate: bool = True,
        init_gamma_zero: bool = True,
        timewise: bool = False,
    ):
        super().__init__()
        self.C = hidden_dim
        self.context_dim = context_dim
        self.use_gate = use_gate
        self.timewise = timewise
        self.ln_x = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        if context_dim is not None:
            self.ln_c = nn.LayerNorm(context_dim)
            in_dim = hidden_dim + context_dim
        else:
            self.ln_c = None
            in_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, 2 * hidden_dim)  # -> [gamma, beta]
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden // 2),
            nn.GELU(),
            nn.Linear(mlp_hidden // 2, hidden_dim)
        ) if use_gate else None

        # init last layer of modulation to zeros for stability => gamma≈0, beta≈0 initially
        with torch.no_grad():
            last = self.mlp[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)
            if self.gate_mlp is not None:
                lastg = self.gate_mlp[-1]
                nn.init.zeros_(lastg.weight)
                nn.init.zeros_(lastg.bias)

    def forward(self, x: torch.Tensor, context = None) -> torch.Tensor:
        # x: [T,B,C]
        T, B, C = x.shape
        xn = self.ln_x(x)
        if self.context_dim is not None and context is not None:
            if context.dim() == 2:  # [B,Cc]
                c = self.ln_c(context).unsqueeze(0).expand(T, -1, -1)  # [T,B,Cc]
            elif context.dim() == 3:  # [B,T,Cc]
                if context.shape[1] != T:
                    # broadcast time if needed
                    c = self.ln_c(context).transpose(0, 1)
                    if c.shape[0] != T:
                        c = c[0:1].expand(T, -1, -1)
                else:
                    c = self.ln_c(context).transpose(0, 1)
            else:
                raise ValueError("context must be [B,Cc] or [B,T,Cc]")
            inp = torch.cat([xn, c], dim=-1)
        else:
            inp = xn

        gam_beta = self.mlp(self.dropout(inp))  # [T,B,2C]
        gamma, beta = gam_beta[..., :C], gam_beta[..., C:]
        y = (1.0 + gamma) * x + beta

        if self.use_gate:
            g = torch.sigmoid(self.gate_mlp(inp))
            y = g * y + (1.0 - g) * x
        return y


