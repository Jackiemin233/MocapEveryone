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

from typing import Optional, Dict, List, Tuple


# -------- helper: load wavelet filters (PyWavelets if available) --------
def _load_wavelet_dec_filters(names: List[str]) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Return lists of dec_lo, dec_hi for each wavelet family name.
    If pywt is available, we use exact taps; otherwise fallback to db4 only.
    """
    try:
        dec_los, dec_his = [], []
        for nm in names:
            w = pywt.Wavelet(nm)
            dec_los.append(list(w.dec_lo))
            dec_his.append(list(w.dec_hi))
        return dec_los, dec_his
    except Exception:
        print("NOOOOOOOOOOOOOO ADA; FALL BACK TO DB4")
        # Fallback: only db4 (your existing taps)
        db4_dec_lo = [
            -0.0105974017850021,  0.0328830116669829,  0.0308413818355607,
            -0.1870348117188811, -0.0279837694169839, 0.6308807679298589,
             0.7148465705525415,  0.2303778133088552
        ]
        db4_dec_hi = [
            -0.2303778133088552,  0.7148465705525415, -0.6308807679298589,
            -0.0279837694169839,  0.1870348117188811, 0.0308413818355607,
            -0.0328830116669829, -0.0105974017850021
        ]
        names = ["db4"]
        return [db4_dec_lo], [db4_dec_hi]

def _pad_center_1d(taps: List[float], Kmax: int) -> torch.Tensor:
    """Center-pad filter taps to length Kmax (both even) and reverse for conv1d."""
    k = len(taps)
    pad_total = Kmax - k
    left = pad_total // 2
    right = pad_total - left
    # reverse for conv1d to emulate convolution (not x-correlation)
    arr = list(reversed(taps))
    arr = [0.0]*left + arr + [0.0]*right
    return torch.tensor(arr, dtype=torch.float32).view(1, 1, -1)

class WaveletSWTBlockAuto(nn.Module):
    """
    自适应小波特征块：
      - 在多个小波族上（默认 ['db2','db4','db6']）做可微混合选择（family_probs）
      - 在 1..L_max 的各层上学习层权重（level_gate），实现“用几层”自适应
      - 与你现有的门控/残差融合完全兼容；返回张量不变
      - 在 forward 中计算一个结构正则项 self.last_arch_loss，训练时将其加到总损失

    关键超参（可按需调整）：
      - families: 小波族名列表（需 pywt；否则自动退化为 ['db4']）
      - L_max: 最大层数（建议 3~4）
      - use_gumbel: 是否用 Gumbel-Softmax 产生 family_probs（鼓励接近 one-hot）
      - tau_family / tau_level: 温度（越低越尖锐）
      - reg_family_entropy / reg_level_entropy: 熵正则权重（鼓励简单架构）
      - reg_level_budget: 惩罚深层使用的预算项，鼓励更少层（对 level_gate 的 L1）
    """
    def __init__(
        self,
        hidden_dim: int,
        L_max: int = 4,
        families: Optional[List[str]] = None,
        dropout: float = 0.10,
        gate_channels_timewise: bool = False,
        init_lowfreq_bias: float = 1.0,
        residual_scale: float = 0.2,

        # architecture selection knobs
        use_gumbel: bool = True,
        tau_family: float = 2.0,
        tau_level: float = 2.0,
        reg_family_entropy: float = 1e-3,
        reg_level_entropy: float = 1e-3,
        reg_level_budget: float = 1e-3,
    ):
        super().__init__()
        assert L_max >= 1
        self.C = hidden_dim
        self.L_max = L_max
        self.dropout = nn.Dropout(dropout)
        self.ln_in  = nn.LayerNorm(hidden_dim)
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.gate_timewise = gate_channels_timewise
        self.init_lowfreq_bias = init_lowfreq_bias
        self.res_scale = residual_scale

        # ---------- filter bank ----------
        if families is None:
            families = ["db2", "db4", "db6"]
        dec_los, dec_his = _load_wavelet_dec_filters(families)
        self.families = families
        self.num_fam  = len(families)
        Kmax = max(len(t) for t in dec_los)  # unify kernel length
        w_lo_bank = []
        w_hi_bank = []
        for lo, hi in zip(dec_los, dec_his):
            w_lo_bank.append(_pad_center_1d(lo, Kmax))
            w_hi_bank.append(_pad_center_1d(hi, Kmax))
        self.register_buffer("w_lo_bank", torch.cat(w_lo_bank, dim=0))  # [F,1,Kmax]
        self.register_buffer("w_hi_bank", torch.cat(w_hi_bank, dim=0))  # [F,1,Kmax]
        self.Kmax = Kmax

        # ---------- small gate for scale fusion (同你旧版的一致) ----------
        gate_hidden = max(64, self.C // 4)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.C * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, self.C)
        )

        # ---------- selectors for NAS ----------
        # 1) choose family (per-sample), based on pooled context
        self.family_selector = nn.Sequential(
            nn.Linear(self.C, 64), nn.GELU(),
            nn.Linear(64, self.num_fam)
        )
        # 2) choose level usage (A_L + D_L + ... + D_1) -> 共 L_max+1 个门
        self.level_selector = nn.Sequential(
            nn.Linear(self.C, 64), nn.GELU(),
            nn.Linear(64, self.L_max + 1)
        )

        # ---------- arch knobs ----------
        self.use_gumbel = use_gumbel
        self.tau_family = tau_family
        self.tau_level  = tau_level
        self.reg_family_entropy = reg_family_entropy
        self.reg_level_entropy  = reg_level_entropy
        self.reg_level_budget   = reg_level_budget

        # ---------- stats ----------
        self.last_stats: Optional[Dict[str, torch.Tensor]] = None
        self.last_arch_loss: Optional[torch.Tensor] = None
        self.last_arch_stats: Optional[Dict[str, torch.Tensor]] = None
    
    # -------- helper: 基于 T / Kmax / L_max 计算可用层数，确保 reflect pad 合法 --------
    def _max_levels_allowed(self, T: int, Kmax: int, L_max: int) -> int:
        """
        reflect 模式要求 pad_l = (d*(Kmax-1))//2 <= T-1
        逐层检查可行性，返回允许的最大层数（至少为 1）
        """
        L_allowed, d = 0, 1
        while L_allowed < L_max:
            eff = d * (Kmax - 1)
            if (eff // 2) <= (T - 1):
                L_allowed += 1
                d <<= 1  # d *= 2
            else:
                break
        return max(L_allowed, 1)

    # depthwise corr with a *single* filter tensor [1,1,K]
    def _depthwise_corr1d(self, x: torch.Tensor, filt: torch.Tensor, dilation: int) -> torch.Tensor:
        B, T, C = x.shape
        k = filt.shape[-1]
        eff = dilation * (k - 1)
        pad_l = eff // 2
        pad_r = eff - pad_l
        x_ch = x.transpose(1, 2)      # [B,C,T]
        w = filt.expand(C, 1, k)      # [C,1,K]
        x_pad = F.pad(x_ch, (pad_l, pad_r), mode="reflect")
        y = F.conv1d(x_pad, w, stride=1, padding=0, dilation=dilation, groups=C)  # [B,C,T]
        return y.transpose(1, 2)

    # do conv for each family and mix by alpha
    def _mix_family_conv(self, x: torch.Tensor, dilation: int, which: str, alpha: torch.Tensor) -> torch.Tensor:
        """
        which: 'lo' or 'hi'
        alpha: [B,F] soft/hard probs
        returns: [B,T,C]
        """
        bank = self.w_lo_bank if which == 'lo' else self.w_hi_bank  # [F,1,K]
        outs = []
        for f in range(self.num_fam):
            y_f = self._depthwise_corr1d(x, bank[f:f+1], dilation)  # [B,T,C]
            outs.append(y_f)
        Y = torch.stack(outs, dim=1)                                # [B,F,T,C]
        alpha_bc = alpha.view(alpha.shape[0], self.num_fam, 1, 1)   # [B,F,1,1]
        return (Y * alpha_bc).sum(dim=1)                            # [B,T,C]

    @staticmethod
    def _entropy_categorical(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # p: [...,K], sum=1
        return -(p * (p + eps).log()).sum(dim=-1)

    @staticmethod
    def _entropy_bernoulli(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        # p in (0,1)
        return -(p * (p + eps).log() + (1 - p) * (1 - p + eps).log())

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        H: [B, T, C]
        returns fused [B, T, C]
        """

        B, T, C = H.shape
        Hn = self.ln_in(H)

        # -------- architecture selection (per-sample) --------
        pooled = Hn.mean(dim=1)                          # [B,C]
        fam_logits = self.family_selector(pooled)        # [B,F]
        if self.use_gumbel:
            family_probs = F.gumbel_softmax(
                fam_logits, tau=self.tau_family, hard=False, dim=-1
            )                                            # [B,F]
        else:
            family_probs = F.softmax(fam_logits / self.tau_family, dim=-1)  # [B,F]

        # level gates: 独立 sigmoid in (0,1)
        # 顺序 = [A_L, D_L, D_{L-1}, ..., D_1]  共 L_max+1 个门
        level_logits = self.level_selector(pooled)       # [B, L_max+1]
        level_gate   = torch.sigmoid(level_logits / self.tau_level)         # [B, L_max+1]

        # -------- 基于当前 T 自适应截断层数 --------
        L_actual = self._max_levels_allowed(T, self.Kmax, self.L_max)

        # -------- SWT with family mixing (仅做到 L_actual) --------
        approx = Hn
        details: List[torch.Tensor] = []
        energies: List[torch.Tensor] = []

        for l in range(1, L_actual + 1):
            d = 2 ** (l - 1)
            a_mix = self._mix_family_conv(approx, d, 'lo', family_probs)    # [B,T,C]
            d_mix = self._mix_family_conv(approx, d, 'hi', family_probs)    # [B,T,C]
            details.append(d_mix)
            approx = a_mix
            energies.append(d_mix.abs().mean(dim=(1, 2), keepdim=True))     # [B,1,1]

        # 组成同原逻辑一致的列表
        U_A = approx
        U_list = [U_A] + details[::-1]                                      # K_actual = L_actual + 1

        # 截断 level_gate，避免索引越界；仅对实际参与的层做正则
        K_actual = L_actual + 1
        level_gate_used = level_gate[:, :K_actual]                           # [B, K_actual]

        # -------- 融合（加上层门控）--------
        fused = H
        stats: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            if len(energies) > 0:
                hf_energy = torch.cat(energies, dim=2)                      # [B,1,L_actual]
            else:
                hf_energy = torch.zeros(B, 1, 1, device=H.device)
            stats["hf_energy_levels_meanL1"] = hf_energy.squeeze(-1)        # [B,1,L_actual]

        for idx, U in enumerate(U_list):
            Ud = self.dropout(U)

            # 原有门控（通道/时序）
            if self.gate_timewise:
                g_in = torch.cat([Hn, Ud], dim=-1)                          # [B,T,2C]
                g = torch.sigmoid(self.gate_mlp(g_in))                      # [B,T,C]
            else:
                h_pool = Hn.mean(dim=1, keepdim=True)                       # [B,1,C]
                u_pool = Ud.mean(dim=1, keepdim=True)                       # [B,1,C]
                g_in = torch.cat([h_pool, u_pool], dim=-1)                  # [B,1,2C]
                g = torch.sigmoid(self.gate_mlp(g_in))                      # [B,1,C]
                g = g.expand(-1, T, -1)                                     # [B,T,C]

            # 低频 gate 初值偏置更开
            if idx == 0 and self.init_lowfreq_bias != 0.0:
                g = torch.clamp(g + self.init_lowfreq_bias, 0.0, 1.0)

            # 乘以“层选择”门控（per-sample）
            w_level = level_gate_used[:, idx].view(B, 1, 1)                 # [B,1,1]
            g = g * w_level

            fused = fused + g * Ud
            stats[f"gate_scale_{idx}"] = g.mean(dim=(1, 2))                 # [B]

        fused = self.ln_out(fused)

        # -------- arch regularization loss（仅对参与层统计） --------
        def _entropy_categorical(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
            return -(p * (p + eps).log()).sum(dim=-1)

        def _entropy_bernoulli(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
            return -(p * (p + eps).log() + (1 - p) * (1 - p + eps).log())

        fam_H  = _entropy_categorical(family_probs).mean()                  # encourage one family
        lvl_H  = _entropy_bernoulli(level_gate_used).mean()                 # encourage binary-like gates
        # 预算仅对高频门（idx>=1）惩罚，鼓励少用深层
        lvl_L1 = level_gate_used[:, 1:].mean() if K_actual > 1 else torch.zeros((), device=H.device)

        arch_loss = (self.reg_family_entropy * fam_H
                    + self.reg_level_entropy  * lvl_H
                    + self.reg_level_budget   * lvl_L1)

        # -------- record stats --------
        self.last_stats = stats
        self.last_arch_loss = arch_loss
        self.last_arch_stats = {
            "family_probs": family_probs.detach(),     # [B,F]
            "level_gate":   level_gate_used.detach(),  # [B,K_actual]
            "fam_entropy":  fam_H.detach().unsqueeze(0),
            "lvl_entropy":  lvl_H.detach().unsqueeze(0),
            "lvl_budget":   lvl_L1.detach().unsqueeze(0),
        }
        return fused

class TransformerSceneFiLMModel_SWT_Ada_Uncertain(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim=1024, num_layers=4, num_heads=8, dropout=0.1, estimate_contact=False,
        context_dim: int | None = None,
        film_mlp_hidden: int = 256,
        film_dropout: float = 0.1,
        film_use_gate: bool = True,
        film_init_gamma_zero: bool = True,

        # ---- Wavelet configs ----
        use_wavelet: bool = True,

        # 固定版参数（兼容你的旧逻辑）
        wavelet_levels: int = 3,
        wavelet_dropout: float = 0.10,
        wavelet_gate_timewise: bool = False,
        wavelet_lowfreq_bias: float = 1.0,

        # —— 新增：自适应开关 & 超参 —— 
        wavelet_adaptive: bool = True,
        wavelet_L_max: int = 4,
        wavelet_families: Optional[List[str]] = None,   # e.g. ["db2","db4","db6"]
        wavelet_use_gumbel: bool = True,
        wavelet_tau_family: float = 2.0,
        wavelet_tau_level: float  = 2.0,
        wavelet_reg_family_entropy: float = 1e-3,
        wavelet_reg_level_entropy:  float = 1e-3,
        wavelet_reg_level_budget:   float = 1e-3,
    ):
        super(TransformerSceneFiLMModel_SWT_Ada_Uncertain, self).__init__()
        self.mid_dim = None
        if isinstance(input_dim, tuple):
            self.input_dim, self.mid_dim = input_dim
        else:
            self.input_dim = input_dim

        self.hidden_dim = hidden_dim
        self.model_type = "TransformerEncoder"
        self.estimate_contact = estimate_contact
        self.use_wavelet = use_wavelet
        self.wavelet_adaptive = wavelet_adaptive

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

        # FiLM
        self.film = FiLMMod(
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            mlp_hidden=film_mlp_hidden,
            dropout=film_dropout,
            use_gate=film_use_gate,
            init_gamma_zero=film_init_gamma_zero
        )

        # Wavelet block
        if self.use_wavelet:
            if self.wavelet_adaptive:
                self.wavelet_block = WaveletSWTBlockAuto(
                    hidden_dim=hidden_dim,
                    L_max=wavelet_L_max,
                    families=wavelet_families,             # None -> ["db2","db4","db6"]
                    dropout=wavelet_dropout,
                    gate_channels_timewise=wavelet_gate_timewise,
                    init_lowfreq_bias=wavelet_lowfreq_bias,
                    residual_scale=0.2,
                    use_gumbel=wavelet_use_gumbel,
                    tau_family=wavelet_tau_family,
                    tau_level=wavelet_tau_level,
                    reg_family_entropy=wavelet_reg_family_entropy,
                    reg_level_entropy=wavelet_reg_level_entropy,
                    reg_level_budget=wavelet_reg_level_budget,
                )
            else:
                raise NotImplementedError
                # 仍支持你的固定版本（保底不动）
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

        self.shared_decoder = nn.Sequential(
            nn.Linear(decode_dim, 256),
            nn.ReLU(),
        )
        self.pose_mean_head   = nn.Linear(256, output_dim)
        self.pose_logvar_head = nn.Linear(256, output_dim)

        self.init_weights()

        self.last_wavelet_stats: Optional[Dict[str, torch.Tensor]] = None
        self.last_wavelet_arch_loss: Optional[torch.Tensor] = None  # <—— 新增

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, context=None, sample=True):
        if src.dim() != 3:
            raise ValueError("src must be [B, T, D]")

        B, T, _ = src.shape
        src_tb = src.transpose(0, 1)  # [T,B,D]

        if self.mid_dim is None:
            projected_src = self.encoder(src_tb) * math.sqrt(self.hidden_dim)
        else:
            half_hidden_dim = self.hidden_dim // 2
            src_input, src_mid = src_tb[..., :self.input_dim], src_tb[..., self.input_dim:]
            projected_input_src = self.input_encoder(src_input)
            projected_mid_src   = self.mid_encoder(src_mid)
            projected_src = torch.cat((projected_input_src, projected_mid_src), -1) * math.sqrt(self.hidden_dim)

        x = self.pos_encoder(projected_src)          # [T,B,C]
        x = self.transformer_encoder(x)              # [T,B,C]

        if context is not None:
            x = self.film(x, context)                # [T,B,C]

        # Wavelet
        if self.wavelet_block is not None:
            x_bt = x.transpose(0, 1)                # [B,T,C]
            x_bt = self.wavelet_block(x_bt)         # [B,T,C]
            x    = x_bt.transpose(0, 1)             # [T,B,C]
            # expose stats/loss for training logs
            self.last_wavelet_stats = getattr(self.wavelet_block, "last_stats", None)
            self.last_wavelet_arch_stats = getattr(self.wavelet_block, "last_arch_stats", None)
            self.last_wavelet_arch_loss = getattr(self.wavelet_block, "last_arch_loss", None)
        else:
            self.last_wavelet_stats = None
            self.last_wavelet_arch_loss = None

        # contact head
        x_dec = x
        contact_output = None
        if self.estimate_contact:
            contact_output = self.contact_decoder(x)     # [T,B,2]
            x_dec = torch.cat((x, contact_output), dim=2)

        dec_feat = self.shared_decoder(x_dec)            # [T,B,256]
        mean   = self.pose_mean_head(dec_feat)           # [T,B,D]
        logvar = self.pose_logvar_head(dec_feat)         # [T,B,D]
        logvar = torch.clamp(logvar, min=-10.0, max=6.0)

        if sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            theta = mean + std * eps
        else:
            theta = mean

        # 回到 [B,T,*]
        mean   = mean.transpose(0, 1)
        logvar = logvar.transpose(0, 1)
        theta  = theta.transpose(0, 1)
        if self.estimate_contact:
            contact_output = contact_output.transpose(0, 1)

        return contact_output, mean, logvar, theta

