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
        wavelet_levels: int = 2,
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
