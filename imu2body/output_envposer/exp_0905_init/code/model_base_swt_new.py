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

from imu2body.model_base import PositionalEncoding, FiLMMod, FiLMMod_





# -------------------------------------------------
# Wavelet SWT with W-FiLM inside
# -------------------------------------------------
class WaveletSWTBlock(nn.Module):
    """
    Multi-level Stationary Wavelet Transform (SWT) along time for [B,T,C] features,
    followed by per-scale FiLM (W-FiLM) using external context, and gated residual fusion.

    Args:
        hidden_dim: channels C
        levels: number of SWT levels (L)
        dropout: dropout applied on scale features before gate
        gate_channels_timewise: if True, gate is [B,T,C]; else gate uses time pooling → [B,1,C]
        init_lowfreq_bias: bias added to the gate at the lowest scale (A_L) to start more open
        residual_scale: reserved for optional scaled residual
        context_dim: channels of external context used for W-FiLM (can be global [B,Cc] or timewise [B,T,Cc])
        film_timewise: if True, compute FiLM gamma/beta timewise (heavier). Default False (channel-only; broadcast on T).
        lowfreq_gamma_range/highfreq_gamma_range: ranges for tanh-scaled gamma around 1 (stability)
        lowfreq_beta_range/highfreq_beta_range: ranges for tanh-scaled beta around 0
        gate_use_context: if True, also feed context into gate MLP (lighter than cross-attn)
    """
    def __init__(
        self,
        hidden_dim: int,
        levels: int = 3,
        dropout: float = 0.1,
        gate_channels_timewise: bool = False,
        init_lowfreq_bias: float = 1.0,
        residual_scale: float = 0.2,
        context_dim: Optional[int] = 1280,
        film_timewise: bool = False,
        lowfreq_gamma_range: float = 0.20,
        lowfreq_beta_range: float = 0.05,
        highfreq_gamma_range: float = 0.40,
        highfreq_beta_range: float = 0.05,
        gate_use_context: bool = False,
    ):
        super().__init__()
        assert levels >= 1
        self.C = hidden_dim
        self.L = levels
        self.dropout = nn.Dropout(dropout)
        self.ln_in = nn.LayerNorm(hidden_dim)
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.gate_timewise = gate_channels_timewise
        self.init_lowfreq_bias = init_lowfreq_bias
        self.res_scale = residual_scale

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
        gate_in_C = self.C * 2 + (context_dim or 0)
        gate_hidden = max(64, self.C // 4)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_C, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.C)
        )
        self.gate_use_context = gate_use_context

        # W-FiLM heads
        self.context_dim = context_dim
        self.film_timewise = film_timewise
        if context_dim is not None:
            self.context_ln = nn.LayerNorm(context_dim)
            film_in = context_dim + self.C  # concat context summary with scale embedding (C)
        else:
            self.context_ln = None
            film_in = self.C

        film_hidden = max(64, self.C // 2)
        self.film_head = nn.Sequential(
            nn.Linear(film_in, film_hidden),
            nn.GELU(),
            nn.Linear(film_hidden, 2 * self.C)
        )
        # init last linear to zeros → gamma≈0, beta≈0 initially
        with torch.no_grad():
            last = self.film_head[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

        # Per-scale learnable embeddings: [A_L, D_L, D_{L-1}, ..., D_1]
        self.scale_emb = nn.Parameter(torch.randn(self.L + 1, self.C))

        # Ranges for FiLM constraints
        self.lowfreq_gamma_range = lowfreq_gamma_range
        self.lowfreq_beta_range  = lowfreq_beta_range
        self.highfreq_gamma_range = highfreq_gamma_range
        self.highfreq_beta_range  = highfreq_beta_range

        # Stats holder
        self.last_stats: Optional[Dict[str, torch.Tensor]] = None

    @staticmethod
    def _depthwise_corr1d(x: torch.Tensor, filt: torch.Tensor, dilation: int) -> torch.Tensor:
        # x: [B,T,C] → conv1d [B,C,T]
        B, T, C = x.shape
        k = filt.shape[-1]
        eff = dilation * (k - 1)
        pad_l = eff // 2
        pad_r = eff - pad_l
        x_ch = x.transpose(1, 2)  # [B,C,T]
        w = filt.expand(C, 1, k)
        x_pad = F.pad(x_ch, (pad_l, pad_r), mode="reflect")
        y = F.conv1d(x_pad, w, stride=1, padding=0, dilation=dilation, groups=C)
        return y.transpose(1, 2)

    def _summarize_context(self, context: Optional[torch.Tensor], T: int) -> Optional[torch.Tensor]:
        if self.context_dim is None or context is None:
            return None
        if context.dim() == 2:  # [B,Cc]
            c = self.context_ln(context)
        elif context.dim() == 3:  # [B,T,Cc]
            # pool over time to [B,Cc] for stability; timewise FiLM can be enabled if needed
            c = self.context_ln(context.mean(dim=1))
        else:
            raise ValueError("context must be [B,Cc] or [B,T,Cc]")
        return c

    def forward(self, H: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """H: [B,T,C] → [B,T,C]"""
        B, T, C = H.shape
        Hn = self.ln_in(H)

        # 1) SWT decomposition
        approx = Hn
        details: List[torch.Tensor] = []
        energies: List[torch.Tensor] = []
        for l in range(1, self.L + 1):
            d = 2 ** (l - 1)
            a = self._depthwise_corr1d(approx, self.w_lo, d)   # low-pass
            dcoef = self._depthwise_corr1d(approx, self.w_hi, d)  # high-pass
            details.append(dcoef)
            approx = a
            energies.append(dcoef.abs().mean(dim=(1, 2), keepdim=True))  # [B,1,1]

        U_A = approx
        U_list = [U_A] + details[::-1]  # [A_L, D_L, D_{L-1}, ..., D_1]

        # 2) Prepare context summary for FiLM and for gate (optional)
        c_sum = self._summarize_context(context, T)  # [B,Cc] or None
        assert c_sum is not None

        # 3) Per-scale FiLM (W-FiLM) → Gate → Residual inject
        fused = H
        stats: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            hf_energy = torch.cat(energies, dim=2) if len(energies) > 0 else torch.zeros(B, 1, 1, device=H.device)
            stats["hf_energy_levels_meanL1"] = hf_energy.squeeze(-1)  # [B,1,L]

        for idx, U in enumerate(U_list):
            # -- FiLM: use context + scale embedding
            scale_token = self.scale_emb[idx].unsqueeze(0).expand(B, -1)  # [B,C]
            if c_sum is not None:
                film_in = torch.cat([c_sum, scale_token], dim=-1)
            else:
                film_in = scale_token
            gb = self.film_head(film_in)  # [B,2C]
            gamma_raw, beta_raw = gb[:, :C], gb[:, C:]

            # constrain ranges for stability
            if idx == 0:  # low-frequency A_L
                gamma = 1.0 + torch.tanh(gamma_raw) * self.lowfreq_gamma_range
                beta  = torch.tanh(beta_raw) * self.lowfreq_beta_range
            else:         # high-frequency bands
                gamma = 1.0 + torch.tanh(gamma_raw) * self.highfreq_gamma_range
                beta  = torch.tanh(beta_raw) * self.highfreq_beta_range

            U = gamma.unsqueeze(1) * U + beta.unsqueeze(1)  # [B,T,C]

            # -- Gate (with optional context)
            Ud = self.dropout(U)
            if self.gate_timewise:
                g_in = torch.cat([Hn, Ud], dim=-1) if not self.gate_use_context else \
                       torch.cat([Hn, Ud, c_sum.unsqueeze(1).expand(-1, T, -1)], dim=-1)
                g = torch.sigmoid(self.gate_mlp(g_in))  # [B,T,C]
            else:
                h_pool = Hn.mean(dim=1, keepdim=True)  # [B,1,C]
                u_pool = Ud.mean(dim=1, keepdim=True)  # [B,1,C]
                if self.gate_use_context and (c_sum is not None):
                    c_pool = c_sum.unsqueeze(1)  # [B,1,Cc]
                    g_in = torch.cat([h_pool, u_pool, c_pool], dim=-1)  # [B,1,2C+Cc]
                else:
                    g_in = torch.cat([h_pool, u_pool], dim=-1)  # [B,1,2C]
                g = torch.sigmoid(self.gate_mlp(g_in)).expand(-1, T, -1)

            if idx == 0 and self.init_lowfreq_bias != 0.0:
                g = torch.clamp(g + self.init_lowfreq_bias, 0.0, 1.0)

            fused = fused + g * Ud

            # stats
            stats[f"gate_scale_{idx}"] = g.mean(dim=(1, 2))  # [B,C] channel mean would be summarized by caller
            stats[f"gamma_scale_{idx}"] = gamma.mean(dim=-1)  # [B]
            stats[f"beta_scale_{idx}"]  = beta.mean(dim=-1)   # [B]

        fused = self.ln_out(fused)
        self.last_stats = stats
        return fused


# -------------------------------------------------
# Main Model with W-FiLM inside Wavelet block
# -------------------------------------------------
class TransformerSceneFiLMModel_WFiLM_Uncertain(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim: int,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        estimate_contact: bool = False,
        # FiLM (outer) configs
        context_dim: Optional[int] = None,
        film_mlp_hidden: int = 256,
        film_dropout: float = 0.1,
        film_use_gate: bool = True,
        film_init_gamma_zero: bool = True,
        film_timewise: bool = False,
        # Wavelet / W-FiLM configs
        use_wavelet: bool = True,
        wavelet_levels: int = 3,
        wavelet_dropout: float = 0.10,
        wavelet_gate_timewise: bool = False,
        wavelet_lowfreq_bias: float = 1.0,
        wavelet_context_dim: Optional[int] = None,   # if None, defaults to context_dim
        wavelet_film_timewise: bool = False,
        wavelet_gate_use_context: bool = False,
    ):
        super().__init__()

        self.mid_dim = None
        if isinstance(input_dim, tuple):
            self.input_dim, self.mid_dim = input_dim
        else:
            self.input_dim = input_dim

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

        # Input projection
        if self.mid_dim is not None:
            half_hidden_dim = hidden_dim // 2
            self.mid_encoder = nn.Linear(self.mid_dim, half_hidden_dim)
            self.input_encoder = nn.Linear(self.input_dim, half_hidden_dim)
        else:
            self.encoder = nn.Linear(self.input_dim, hidden_dim)

        # Outer FiLM (global modulation before wavelet)
        self.film = FiLMMod_(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            mlp_hidden=film_mlp_hidden,
            dropout=film_dropout,
            use_gate=film_use_gate,
            init_gamma_zero=film_init_gamma_zero,
            timewise=film_timewise,
        ) if context_dim is not None else None

        # Wavelet block with internal W-FiLM
        if self.use_wavelet:
            w_ctx_dim = wavelet_context_dim if wavelet_context_dim is not None else context_dim
            self.wavelet_block = WaveletSWTBlock(
                hidden_dim=hidden_dim,
                levels=wavelet_levels,
                dropout=wavelet_dropout,
                gate_channels_timewise=wavelet_gate_timewise,
                init_lowfreq_bias=wavelet_lowfreq_bias,
                context_dim=w_ctx_dim,
                film_timewise=wavelet_film_timewise,
                gate_use_context=wavelet_gate_use_context,
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

        self.shared_decoder = nn.Sequential(
            nn.Linear(decode_dim, 256),
            nn.ReLU(),
        )
        self.pose_mean_head   = nn.Linear(256, output_dim)
        self.pose_logvar_head = nn.Linear(256, output_dim)

        self.init_weights()

        # exposed for logging
        self.last_wavelet_stats: Optional[Dict[str, torch.Tensor]] = None

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def _project_input(self, src_tb: torch.Tensor) -> torch.Tensor:
        # src_tb: [T,B,D]
        if self.mid_dim is None:
            return self.encoder(src_tb) * math.sqrt(self.hidden_dim)
        else:
            half_hidden_dim = self.hidden_dim // 2
            src_input, src_mid = src_tb[..., :self.input_dim], src_tb[..., self.input_dim:]
            projected_input_src = self.input_encoder(src_input)
            projected_mid_src = self.mid_encoder(src_mid)
            projected_src = torch.cat((projected_input_src, projected_mid_src), -1)
            return projected_src * math.sqrt(self.hidden_dim)

    def forward(
        self,
        src: torch.Tensor,                # [B,T,input_dim]
        context: Optional[torch.Tensor] = None,  # [B,Cc] or [B,T,Cc]
        sample: bool = True,
    ):
        if src.dim() != 3:
            raise ValueError("src must be [B, T, D]")
        B, T, _ = src.shape
        src_tb = src.transpose(0, 1)  # [T,B,D]

        # Input → PosEnc → Transformer
        x = self._project_input(src_tb)                 # [T,B,C]
        x = self.pos_encoder(x)                         # [T,B,C]
        x = self.transformer_encoder(x)                 # [T,B,C]

        # Outer FiLM (optional)
        if self.film is not None and context is not None:
            x = self.film(x, context)                   # [T,B,C]

        # Wavelet W-FiLM block
        if self.wavelet_block is not None:
            x_bt = x.transpose(0, 1)                    # [B,T,C]
            x_bt = self.wavelet_block(x_bt, context=context)  # [B,T,C]
            self.last_wavelet_stats = self.wavelet_block.last_stats
            x = x_bt.transpose(0, 1)
        else:
            self.last_wavelet_stats = None

        # Contact head (optional)
        contact_output = None
        x_dec = x
        if self.estimate_contact:
            contact_output = self.contact_decoder(x)     # [T,B,2]
            x_dec = torch.cat((x, contact_output), dim=2)

        # Regression heads
        dec_feat = self.shared_decoder(x_dec)            # [T,B,256]
        mean     = self.pose_mean_head(dec_feat)         # [T,B,Dout]
        logvar   = self.pose_logvar_head(dec_feat)       # [T,B,Dout]
        logvar = torch.clamp(logvar, min=-10.0, max=6.0)

        if sample:
            std  = torch.exp(0.5 * logvar)
            eps  = torch.randn_like(std)
            theta = mean + std * eps
        else:
            theta = mean

        # back to [B,T,*]
        mean   = mean.transpose(0, 1)
        logvar = logvar.transpose(0, 1)
        theta  = theta.transpose(0, 1)
        if self.estimate_contact:
            contact_output = contact_output.transpose(0, 1)

        return contact_output, mean, logvar, theta