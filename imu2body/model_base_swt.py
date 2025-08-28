import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

from torch.nn import LayerNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from torch.nn.init import xavier_uniform_
from typing import Optional, Dict, List

from imu2body.model_base import PositionalEncoding, FiLMMod

# ----------------------------
# Wavelet (SWT) Feature Block
# ----------------------------
class WaveletSWTBlock(nn.Module):
    """
    Multi-level stationary wavelet transform (SWT) along time for [B, T, C] features,
    followed by gated residual fusion back into the stream.

    - Wavelet family: db4 (fixed analysis filters).
    - Levels: L (e.g., 3).
    - Padding: reflect (approx symmetric).
    - Output: fused features with LN, plus internal stats stored for logging.
    """
    def __init__(
        self,
        hidden_dim: int,
        levels: int = 3,
        dropout: float = 0.1,
        gate_channels_timewise: bool = False,  # False: gate shape [B,1,C]; True: [B,T,C]
        init_lowfreq_bias: float = 1.0,        # make low-freq gate start a bit more open
        residual_scale: float = 0.2,
    ):
        super().__init__()
        assert levels >= 1
        self.C = hidden_dim
        self.L = levels
        self.dropout = nn.Dropout(dropout)
        self.ln_in = nn.LayerNorm(hidden_dim)
        self.ln_out = nn.LayerNorm(hidden_dim)

        # ---- db4 analysis filters (PyWavelets convention) ----
        # dec_lo / dec_hi
        db4_dec_lo = [                  # pywt.Wavelet('db4').dec_lo
            -0.0105974017850021,  0.0328830116669829,  0.0308413818355607,
            -0.1870348117188811, -0.0279837694169839, 0.6308807679298589,
             0.7148465705525415,  0.2303778133088552
        ]
        db4_dec_hi = [                  # pywt.Wavelet('db4').dec_hi
            -0.2303778133088552,  0.7148465705525415, -0.6308807679298589,
            -0.0279837694169839,  0.1870348117188811, 0.0308413818355607,
            -0.0328830116669829, -0.0105974017850021
        ]
        # torch conv1d is cross-correlation; to emulate convolution, reverse the filter
        lo = torch.tensor(list(reversed(db4_dec_lo)), dtype=torch.float32).view(1, 1, -1)
        hi = torch.tensor(list(reversed(db4_dec_hi)), dtype=torch.float32).view(1, 1, -1)

        # Register as buffers so they move with .to(device) and don't get trained
        self.register_buffer("w_lo", lo)  # [1,1,K]
        self.register_buffer("w_hi", hi)  # [1,1,K]

        # Small gating MLP shared across scales; LN before MLP
        gate_hidden = max(64, self.C // 4)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.C * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.C)
        )
        self.gate_timewise = gate_channels_timewise
        self.init_lowfreq_bias = init_lowfreq_bias

        # NOTE
        self.res_scale = residual_scale

        # stats holder (set each forward)
        self.last_stats: Optional[Dict[str, torch.Tensor]] = None

    def _depthwise_corr1d(self, x: torch.Tensor, filt: torch.Tensor, dilation: int) -> torch.Tensor:
        """
        x: [B, T, C] -> conv1d expects [B, C, T]
        filt: [1,1,K] -> expanded to [C,1,K], groups=C
        """
        B, T, C = x.shape
        k = filt.shape[-1]
        eff = dilation * (k - 1)         # effective receptive width minus 1
        pad_l = eff // 2                 # floor
        pad_r = eff - pad_l              # ceil
        x_ch = x.transpose(1, 2)         # [B, C, T]
        w = filt.expand(C, 1, k)         # [C, 1, K]
        # NOTE: use asymmetric padding when eff is odd (e.g., 7 -> 3,4) to keep length
        x_pad = F.pad(x_ch, (pad_l, pad_r), mode="reflect")
        y = F.conv1d(x_pad, w, stride=1, padding=0, dilation=dilation, groups=C)  # [B, C, T]
        return y.transpose(1, 2)         # [B, T, C]

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        H: [B, T, C]
        returns fused: [B, T, C]
        """
        B, T, C = H.shape
        Hn = self.ln_in(H)

        # SWT: iteratively filter the approximation with upsampled (dilated) filters
        approx = Hn
        details: List[torch.Tensor] = []
        energies: List[torch.Tensor] = []

        for l in range(1, self.L + 1):
            d = 2 ** (l - 1)  # dilation doubles per level
            a = self._depthwise_corr1d(approx, self.w_lo, d)   # low-pass
            dcoef = self._depthwise_corr1d(approx, self.w_hi, d)  # high-pass
            details.append(dcoef)
            approx = a

            # energy per level (mean L1 across time & channels)
            energies.append(dcoef.abs().mean(dim=(1, 2), keepdim=True))  # [B,1,1]

        # Per-scale contributions (time-aligned)
        U_A = approx                         # last-level approximation [B,T,C]
        U_list = [U_A] + details[::-1]       # order: [A_L, D_L, D_{L-1}, ..., D_1]

        # gates
        fused = H
        stats: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            hf_energy = torch.cat(energies, dim=2) if len(energies) > 0 else torch.zeros(B, 1, 1, device=H.device)
            # simple: ratio of HF energy over total energy proxy
            stats["hf_energy_levels_meanL1"] = hf_energy.squeeze(-1)  # [B,1,L]
        # build and apply gates
        # low-frequency first, then high-frequency from coarse->fine
        for idx, U in enumerate(U_list):
            # small dropout on scale features
            Ud = self.dropout(U)

            if self.gate_timewise:
                # Gate per time-step (heavier): concat LN(mean-pooled H, current U)
                g_in = torch.cat([Hn, Ud], dim=-1)                 # [B,T,2C]
                g = torch.sigmoid(self.gate_mlp(g_in))             # [B,T,C]
            else:
                # Gate per channel only (stable): pool over time
                h_pool = Hn.mean(dim=1, keepdim=True)              # [B,1,C]
                u_pool = Ud.mean(dim=1, keepdim=True)              # [B,1,C]
                g_in = torch.cat([h_pool, u_pool], dim=-1)         # [B,1,2C]
                g = torch.sigmoid(self.gate_mlp(g_in))             # [B,1,C]
                g = g.expand(-1, T, -1)                            # [B,T,C]

            # bias low-frequency gate to open more at init
            if idx == 0 and self.init_lowfreq_bias != 0.0:
                g = torch.clamp(g + self.init_lowfreq_bias, 0.0, 1.0)

            fused = fused + g * Ud

            # collect stats
            stats[f"gate_scale_{idx}"] = g.mean(dim=(1, 2))  # [B,C] -> log channel-avg by caller if needed

        fused = self.ln_out(fused)
        self.last_stats = stats
        return fused
            
        # # ★ 关键两行：把“累加的残差”整体乘一个系数 s 再加回
        # delta = fused - H                   # = Σ (g * U)
        # fused = H + self.res_scale * delta  # s ∈ [0,1]，如 0.2

        # fused = self.ln_out(fused)          # 最后再做输出LN
        # self.last_stats = stats
        # return fused


# ------------------------------------------------------
# Your model with Wavelet block plugged in (feature end)
# ------------------------------------------------------
class TransformerSceneFiLMModel_SWT(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim=1024, num_layers=4, num_heads=8, dropout=0.1, estimate_contact=False,
        context_dim: int | None = None,
        film_mlp_hidden: int = 256,
        film_dropout: float = 0.1,
        film_use_gate: bool = True,
        film_init_gamma_zero: bool = True,

        # ---- Wavelet configs (new) ----
        use_wavelet: bool = True,
        wavelet_levels: int = 3,
        wavelet_dropout: float = 0.10,
        wavelet_gate_timewise: bool = False,
        wavelet_lowfreq_bias: float = 1.0,
    ):
        """
        input_dim: this is the dimension of the input
        ninp: 1024 this is the dimension of the hidden layer
        hidden_dim: same as ninp
        num_layers: the number of layers in transformer encoder and decoder. can be either 1 or 4
        """
        super(TransformerSceneFiLMModel_SWT, self).__init__()

        self.mid_dim = None
        if isinstance(input_dim, tuple):
            self.input_dim, self.mid_dim = input_dim

        self.hidden_dim = hidden_dim
        self.model_type = "TransformerEncoder"
        self.estimate_contact = estimate_contact
        self.use_wavelet = use_wavelet

        # Positional + Transformer encoder
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=LayerNorm(hidden_dim),
        )

        # Input projection (continuous features)
        if self.mid_dim is not None:
            half_hidden_dim = hidden_dim // 2
            self.mid_encoder = nn.Linear(self.mid_dim, half_hidden_dim)
            self.input_encoder = nn.Linear(self.input_dim, half_hidden_dim)
        else:
            self.encoder = nn.Linear(input_dim, hidden_dim)

        # FiLM (contextual modulation)
        self.film = FiLMMod(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            mlp_hidden=film_mlp_hidden,
            dropout=film_dropout,
            use_gate=film_use_gate,
            init_gamma_zero=film_init_gamma_zero
        )

        # Wavelet block (WFB-Enc)
        if self.use_wavelet:
            self.wavelet_block = WaveletSWTBlock(
                hidden_dim=hidden_dim,
                levels=wavelet_levels,
                dropout=wavelet_dropout,
                gate_channels_timewise=wavelet_gate_timewise,
                init_lowfreq_bias=wavelet_lowfreq_bias,
            )
        else:
            self.wavelet_block = None

        # Heads
        decode_dim = hidden_dim
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

        # exposed for logging
        self.last_wavelet_stats: Optional[Dict[str, torch.Tensor]] = None

    def init_weights(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, context=None):
        """
        src: [B, T, input_dim] (you were passing [T,B,*] then transpose; we keep your original logic)
        context: optional context for FiLM, typically [B, T, Cc] or [B, Cc]; handled by your FiLMMod.
        returns:
            if estimate_contact: (contact_logits [B,T,2], output [B,T,output_dim])
            else: (None, output [B,T,output_dim])
        """
        # Your original code expects [T, B, *] inside Transformer;
        # keep external API the same: accept [B,T,*], then transpose.
        if src.dim() != 3:
            raise ValueError("src must be [B, T, D]")

        B, T, _ = src.shape
        src_tb = src.transpose(0, 1)  # [T, B, D]

        # Input projection
        if self.mid_dim is None:
            projected_src = self.encoder(src_tb) * np.sqrt(self.hidden_dim)  # [T,B,C]
        else:
            half_hidden_dim = self.hidden_dim // 2
            src_input, src_mid = src_tb[..., :self.input_dim], src_tb[..., self.input_dim:]
            projected_input_src = self.input_encoder(src_input)
            projected_mid_src = self.mid_encoder(src_mid)
            projected_src = torch.cat((projected_input_src, projected_mid_src), -1) * np.sqrt(self.hidden_dim)

        # Positional + Encoder
        x = self.pos_encoder(projected_src)              # [T,B,C]
        x = self.transformer_encoder(x)                  # [T,B,C]

        # FiLM context modulation (if any)
        if context is not None:
            x = self.film(x, context)                    # [T,B,C]

        # ---- Wavelet feature fusion (SWT) ----
        if self.wavelet_block is not None:
            x_bt = x.transpose(0, 1)                    # [B,T,C]
            x_bt = self.wavelet_block(x_bt)             # [B,T,C]
            self.last_wavelet_stats = self.wavelet_block.last_stats  # optional: log outside
            x = x_bt.transpose(0, 1)                    # [T,B,C]
        else:
            self.last_wavelet_stats = None

        # Contact head (optional)
        x_dec = x                                        # define default
        if self.estimate_contact:
            contact_output = self.contact_decoder(x)     # [T,B,2]
            x_dec = torch.cat((x, contact_output), dim=2)

        # Regression head
        output = self.linear_decoder(x_dec)              # [T,B,output_dim]

        if self.estimate_contact:
            return contact_output.transpose(0, 1), output.transpose(0, 1)  # [B,T,*]
        return None, output.transpose(0, 1)                                  # [B,T,*]


# ------------------------------------------------------
# Your model with Wavelet block plugged in (feature end)
# ------------------------------------------------------
class TransformerSceneFiLMModel_SWT_Uncertain(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim=1024, num_layers=4, num_heads=8, dropout=0.1, estimate_contact=False,
        context_dim: int | None = None,
        film_mlp_hidden: int = 256,
        film_dropout: float = 0.1,
        film_use_gate: bool = True,
        film_init_gamma_zero: bool = True,

        # ---- Wavelet configs (new) ----
        use_wavelet: bool = True,
        wavelet_levels: int = 3,
        wavelet_dropout: float = 0.10,
        wavelet_gate_timewise: bool = False,
        wavelet_lowfreq_bias: float = 1.0,
    ):
        """
        input_dim: this is the dimension of the input
        ninp: 1024 this is the dimension of the hidden layer
        hidden_dim: same as ninp
        num_layers: the number of layers in transformer encoder and decoder. can be either 1 or 4
        """
        super(TransformerSceneFiLMModel_SWT_Uncertain, self).__init__()

        self.mid_dim = None
        if isinstance(input_dim, tuple):
            self.input_dim, self.mid_dim = input_dim

        self.hidden_dim = hidden_dim
        self.model_type = "TransformerEncoder"
        self.estimate_contact = estimate_contact
        self.use_wavelet = use_wavelet

        # Positional + Transformer encoder
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layer = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=LayerNorm(hidden_dim),
        )

        # Input projection (continuous features)
        if self.mid_dim is not None:
            half_hidden_dim = hidden_dim // 2
            self.mid_encoder = nn.Linear(self.mid_dim, half_hidden_dim)
            self.input_encoder = nn.Linear(self.input_dim, half_hidden_dim)
        else:
            self.encoder = nn.Linear(input_dim, hidden_dim)

        # FiLM (contextual modulation)
        self.film = FiLMMod(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            mlp_hidden=film_mlp_hidden,
            dropout=film_dropout,
            use_gate=film_use_gate,
            init_gamma_zero=film_init_gamma_zero
        )

        # Wavelet block (WFB-Enc)
        if self.use_wavelet:
            self.wavelet_block = WaveletSWTBlock(
                hidden_dim=hidden_dim,
                levels=wavelet_levels,
                dropout=wavelet_dropout,
                gate_channels_timewise=wavelet_gate_timewise,
                init_lowfreq_bias=wavelet_lowfreq_bias,
            )
        else:
            self.wavelet_block = None

        # Heads
        decode_dim = hidden_dim
        if self.estimate_contact:
            self.contact_decoder = nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            )
            decode_dim += 2

        # self.linear_decoder = nn.Sequential(
        #     nn.Linear(decode_dim, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, output_dim)
        # )

        self.shared_decoder = nn.Sequential(
                                nn.Linear(decode_dim, 256),
                                nn.ReLU(),
                            )
        self.pose_mean_head   = nn.Linear(256, output_dim)  # θ^e
        self.pose_logvar_head = nn.Linear(256, output_dim)  # log σ^2

        self.init_weights()

        # exposed for logging
        self.last_wavelet_stats: Optional[Dict[str, torch.Tensor]] = None

    def init_weights(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, context=None, sample=True):
        """
        src: [B, T, input_dim] (you were passing [T,B,*] then transpose; we keep your original logic)
        context: optional context for FiLM, typically [B, T, Cc] or [B, Cc]; handled by your FiLMMod.
        returns:
            if estimate_contact: (contact_logits [B,T,2], output [B,T,output_dim])
            else: (None, output [B,T,output_dim])
        """
        # Your original code expects [T, B, *] inside Transformer;
        # keep external API the same: accept [B,T,*], then transpose.
        if src.dim() != 3:
            raise ValueError("src must be [B, T, D]")

        B, T, _ = src.shape
        src_tb = src.transpose(0, 1)  # [T, B, D]

        # Input projection
        if self.mid_dim is None:
            projected_src = self.encoder(src_tb) * np.sqrt(self.hidden_dim)  # [T,B,C]
        else:
            half_hidden_dim = self.hidden_dim // 2
            src_input, src_mid = src_tb[..., :self.input_dim], src_tb[..., self.input_dim:]
            projected_input_src = self.input_encoder(src_input)
            projected_mid_src = self.mid_encoder(src_mid)
            projected_src = torch.cat((projected_input_src, projected_mid_src), -1) * np.sqrt(self.hidden_dim)

        # Positional + Encoder
        x = self.pos_encoder(projected_src)              # [T,B,C]
        x = self.transformer_encoder(x)                  # [T,B,C]

        # FiLM context modulation (if any)
        if context is not None:
            x = self.film(x, context)                    # [T,B,C]

        # ---- Wavelet feature fusion (SWT) ----
        if self.wavelet_block is not None:
            x_bt = x.transpose(0, 1)                    # [B,T,C]
            x_bt = self.wavelet_block(x_bt)             # [B,T,C]
            self.last_wavelet_stats = self.wavelet_block.last_stats  # optional: log outside
            x = x_bt.transpose(0, 1)                    # [T,B,C]
        else:
            self.last_wavelet_stats = None

        # Contact head (optional)
        x_dec = x
        contact_output = None
        if self.estimate_contact:
            contact_output = self.contact_decoder(x)     # [T,B,2]
            x_dec = torch.cat((x, contact_output), dim=2)

        # Regression head
        # output = self.linear_decoder(x_dec)              # [T,B,output_dim]
            
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


class TransformerSceneFiLMModel_Uncertain_SWT_BiSmoother(nn.Module):
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
        fusion: str = "poe",              # 预留：目前实现 poe（推荐）。也可扩展 "gate"

        # ---- Wavelet configs (new) ----
        use_wavelet: bool = True,
        wavelet_levels: int = 3,
        wavelet_dropout: float = 0.10,
        wavelet_gate_timewise: bool = False,
        wavelet_lowfreq_bias: float = 1.0,
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
        self.use_wavelet = use_wavelet

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

                # Wavelet block (WFB-Enc)
        if self.use_wavelet:
            self.wavelet_block = WaveletSWTBlock(
                hidden_dim=hidden_dim,
                levels=wavelet_levels,
                dropout=wavelet_dropout,
                gate_channels_timewise=wavelet_gate_timewise,
                init_lowfreq_bias=wavelet_lowfreq_bias,
            )
        else:
            self.wavelet_block = None

        self.init_weights()

        self.last_wavelet_stats: Optional[Dict[str, torch.Tensor]] = None

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
        # ---- Wavelet feature fusion (SWT) ----
        if self.wavelet_block is not None:
            x_bt = x.transpose(0, 1)                    # [B,T,C]
            x_bt = self.wavelet_block(x_bt)             # [B,T,C]
            self.last_wavelet_stats = self.wavelet_block.last_stats  # optional: log outside
            x = x_bt.transpose(0, 1)                    # [T,B,C]
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
