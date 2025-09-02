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

class WaveletSWTBlockAuto_Cxt(nn.Module):
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
        gate_timewise: bool = False,
        init_lowfreq_bias: float = 1.0,
        residual_scale: float = 0.2,

        # ------ context & FiLM ------
        context_dim: Optional[int] = None,   # 若提供，将可用于 FiLM / Gate
        use_film: bool = True,               # 是否启用每尺度 FiLM
        lowfreq_gamma_range: float = 0.25,   # 低频 FiLM 的 γ 振幅
        lowfreq_beta_range:  float = 0.10,   # 低频 FiLM 的 β 振幅
        highfreq_gamma_range: float = 0.15,  # 高频 FiLM 的 γ 振幅
        highfreq_beta_range:  float = 0.05,  # 高频 FiLM 的 β 振幅
        gate_use_context: bool = False,      # Gate 是否使用 context

        # ------ NAS knobs ------
        use_gumbel: bool = True,
        tau_family: float = 2.0,
        tau_level:  float = 2.0,
        reg_family_entropy: float = 1e-3,
        reg_level_entropy:  float = 1e-3,
        reg_level_budget:   float = 1e-3,
        # ------ 接收旧名字等其它未知参数，做兼容 ------
        **kwargs
    ):
        super().__init__()
        assert L_max >= 1
        self.C = hidden_dim
        self.L_max = L_max
        self.dropout = nn.Dropout(dropout)
        self.ln_in  = nn.LayerNorm(hidden_dim)
        self.ln_out = nn.LayerNorm(hidden_dim)

        self.gate_timewise = gate_timewise
        self.init_lowfreq_bias = init_lowfreq_bias
        self.res_scale = residual_scale

        # --- 旧参数名别名映射 ---
        if 'gate_channels_timewise' in kwargs:
            gate_timewise = kwargs.pop('gate_channels_timewise')
        if 'levels' in kwargs:               # 若旧代码传了 levels=3
            L_max = kwargs.pop('levels')
        # 如果还有别的未知参数，给出清晰报错以便定位

        # ===== wavelet filter bank =====
        if families is None:
            families = ["db2", "db4", "db6"]
        self.families = families
        self.num_fam  = len(families)

        dec_los, dec_his = _load_wavelet_dec_filters(families)  # 需配合前面提供的 helper
        Kmax = max(len(t) for t in dec_los)
        w_lo_bank, w_hi_bank = [], []
        for lo, hi in zip(dec_los, dec_his):
            w_lo_bank.append(_pad_center_1d(lo, Kmax))  # [1,1,Kmax]
            w_hi_bank.append(_pad_center_1d(hi, Kmax))  # [1,1,Kmax]
        self.register_buffer("w_lo_bank", torch.cat(w_lo_bank, dim=0))  # [F,1,Kmax]
        self.register_buffer("w_hi_bank", torch.cat(w_hi_bank, dim=0))  # [F,1,Kmax]
        self.Kmax = Kmax

        # ===== small gate MLP for scale fusion =====
        # 如果 Gate 使用 context，则会先投影到 C 维，再与 [Hn, Ud] 拼接 → 输入维度 = 3C，否则 = 2C
        gate_in_dim = 2 * hidden_dim + (hidden_dim if (gate_use_context and context_dim is not None) else 0)
        gate_hidden = max(64, hidden_dim // 4)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, hidden_dim),
        )
        self.gate_use_context = gate_use_context
        self.gate_ctx_proj = nn.Linear(context_dim, hidden_dim) if (gate_use_context and context_dim is not None) else None

        # ===== FiLM (per-scale) =====
        self.use_film = use_film
        self.context_dim = context_dim
        self.lowfreq_gamma_range  = lowfreq_gamma_range
        self.lowfreq_beta_range   = lowfreq_beta_range
        self.highfreq_gamma_range = highfreq_gamma_range
        self.highfreq_beta_range  = highfreq_beta_range

        if use_film:
            film_in_dim = (context_dim or 0) + hidden_dim  # concat [c_sum, scale_token]
            # 轻量：单层线性也够；如需更强可加隐藏层
            self.film_head = nn.Linear(film_in_dim, 2 * hidden_dim)
            # K = L_max+1 个“尺度 token”
            self.scale_emb = nn.Parameter(torch.zeros(L_max + 1, hidden_dim))
            nn.init.normal_(self.scale_emb, mean=0.0, std=0.02)
        else:
            self.film_head = None
            self.scale_emb = None

        # ===== selectors for NAS =====
        # family 选择（类别型）
        self.family_selector = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(),
            nn.Linear(64, self.num_fam)
        )
        # level 选择（独立 sigmoid，顺序 [A_L, D_L, D_{L-1}, ..., D_1]）
        self.level_selector = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.GELU(),
            nn.Linear(64, L_max + 1)
        )

        # ===== NAS knobs / regs =====
        self.use_gumbel = use_gumbel
        self.tau_family = tau_family
        self.tau_level  = tau_level
        self.reg_family_entropy = reg_family_entropy
        self.reg_level_entropy  = reg_level_entropy
        self.reg_level_budget   = reg_level_budget

        # ===== runtime stats =====
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

    # ---- helper: 汇总 context 到 [B,Cc] ----
    def _summarize_context(self, ctx: Optional[torch.Tensor], T: int) -> Optional[torch.Tensor]:
        if ctx is None:
            return None
        if ctx.dim() == 3 and ctx.size(1) == T:   # [B,T,Cc]
            return ctx.mean(dim=1)               # [B,Cc]
        if ctx.dim() == 2:                       # [B,Cc]
            return ctx
        return None

    def forward(self, H: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        H: [B, T, C]  -> returns fused [B, T, C]
        context: 可为 [B,T,Cc] 或 [B,Cc]，将被时间汇聚到 [B,Cc]
        """

        B, T, C = H.shape
        Hn = self.ln_in(H)

        # -------- summarize context --------
        c_sum = self._summarize_context(context, T)   # [B,Cc] or None
        assert c_sum is not None

        # -------- architecture selection (family / level) --------
        pooled = Hn.mean(dim=1)                          # [B,C]
        fam_logits = self.family_selector(pooled)        # [B,F]
        if self.use_gumbel:
            family_probs = F.gumbel_softmax(
                fam_logits, tau=self.tau_family, hard=False, dim=-1
            )                                            # [B,F]
        else:
            family_probs = F.softmax(fam_logits / self.tau_family, dim=-1)  # [B,F]

        level_logits = self.level_selector(pooled)       # [B, L_max+1]
        level_gate   = torch.sigmoid(level_logits / self.tau_level)                               # [B, L_max+1]

        # -------- adapt levels by sequence length --------
        L_actual = self._max_levels_allowed(T, self.Kmax, self.L_max)

        # -------- SWT with family mixing (only up to L_actual) --------
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

        U_A = approx
        U_list = [U_A] + details[::-1]                                      # K_actual = L_actual + 1
        K_actual = L_actual + 1
        level_gate_used = level_gate[:, :K_actual]                           # [B, K_actual]

        # -------- optional FiLM (per-scale) + Gate (optionally with context) --------
        fused = H
        stats: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            if len(energies) > 0:
                hf_energy = torch.cat(energies, dim=2)                      # [B,1,L_actual]
            else:
                hf_energy = torch.zeros(B, 1, 1, device=H.device)
            stats["hf_energy_levels_meanL1"] = hf_energy.squeeze(-1)        # [B,1,L_actual]

        # 默认的 FiLM 幅度范围（若类里未定义）
        low_g_range  = getattr(self, "lowfreq_gamma_range", 0.25)
        low_b_range  = getattr(self, "lowfreq_beta_range",  0.10)
        high_g_range = getattr(self, "highfreq_gamma_range", 0.15)
        high_b_range = getattr(self, "highfreq_beta_range",  0.05)

        # gate 上是否使用 context 以及是否有投影
        gate_use_ctx = bool(getattr(self, "gate_use_context", False))
        gate_ctx_proj = getattr(self, "gate_ctx_proj", None)   # Optional[nn.Linear], 映射到 C 维

        for idx, U in enumerate(U_list):
            # ---- FiLM（若具备 film_head/scale_emb 才启用；否则退化为恒等）----
            gamma = torch.ones(B, C, device=H.device, dtype=H.dtype)
            beta  = torch.zeros(B, C, device=H.device, dtype=H.dtype)

            film_head = getattr(self, "film_head", None)       # 期望: nn.Linear(Cc + C, 2C) 或 nn.Linear(C, 2C)
            scale_emb = getattr(self, "scale_emb", None)       # 期望: nn.ParameterList / nn.Embedding / [K,C]
            if (film_head is not None) and (scale_emb is not None):
                # 取当前尺度 token
                if isinstance(scale_emb, nn.ParameterList):
                    scale_token = scale_emb[idx]                           # [C]
                elif isinstance(scale_emb, nn.Embedding):
                    scale_token = scale_emb(torch.tensor(idx, device=H.device))  # [C]
                else:
                    # 假设是 [K,C] 的 tensor/list
                    scale_token = scale_emb[idx]
                    if scale_token.dim() == 1:
                        pass
                    else:
                        scale_token = scale_token.view(-1)

                scale_token = scale_token.unsqueeze(0).expand(B, -1)       # [B,C]

                if c_sum is not None:
                    film_in = torch.cat([c_sum, scale_token], dim=-1)      # [B, Cc+C]
                else:
                    film_in = scale_token                                   # [B, C]

                gb = film_head(film_in)                                     # [B, 2C]
                gamma_raw, beta_raw = gb[:, :C], gb[:, C:]

                if idx == 0:  # 低频
                    gamma = 1.0 + torch.tanh(gamma_raw) * low_g_range
                    beta  = torch.tanh(beta_raw) * low_b_range
                else:
                    gamma = 1.0 + torch.tanh(gamma_raw) * high_g_range
                    beta  = torch.tanh(beta_raw) * high_b_range

            # 应用 FiLM
            U = gamma.unsqueeze(1) * U + beta.unsqueeze(1)                  # [B,T,C]

            # ---- Gate（可选带 context）----
            Ud = self.dropout(U)
            if self.gate_timewise:
                # g_in: [B,T,2C] 或 [B,T,3C]（若使用 context 且有投影）
                if gate_use_ctx and (c_sum is not None) and (gate_ctx_proj is not None):
                    c_proj = gate_ctx_proj(c_sum).unsqueeze(1).expand(-1, T, -1)  # [B,T,C]
                    g_in = torch.cat([Hn, Ud, c_proj], dim=-1)                    # [B,T,3C]
                else:
                    g_in = torch.cat([Hn, Ud], dim=-1)                            # [B,T,2C]
                g = torch.sigmoid(self.gate_mlp(g_in))                            # [B,T,C]
            else:
                h_pool = Hn.mean(dim=1, keepdim=True)                             # [B,1,C]
                u_pool = Ud.mean(dim=1, keepdim=True)                             # [B,1,C]
                if gate_use_ctx and (c_sum is not None) and (gate_ctx_proj is not None):
                    c_pool = gate_ctx_proj(c_sum).unsqueeze(1)                    # [B,1,C]
                    g_in = torch.cat([h_pool, u_pool, c_pool], dim=-1)            # [B,1,3C]
                else:
                    g_in = torch.cat([h_pool, u_pool], dim=-1)                    # [B,1,2C]
                g = torch.sigmoid(self.gate_mlp(g_in)).expand(-1, T, -1)          # [B,T,C]

            if idx == 0 and self.init_lowfreq_bias != 0.0:
                g = torch.clamp(g + self.init_lowfreq_bias, 0.0, 1.0)

            # 叠加“层选择”门控
            w_level = level_gate_used[:, idx].view(B, 1, 1)                        # [B,1,1]
            g = g * w_level

            fused = fused + g * Ud

            # 统计
            stats[f"gate_scale_{idx}"] = g.mean(dim=(1, 2))                        # [B]
            # 仅当启用 FiLM 才记录 gamma/beta
            if (film_head is not None) and (scale_emb is not None):
                stats[f"gamma_scale_{idx}"] = gamma.mean(dim=-1)                   # [B]
                stats[f"beta_scale_{idx}"]  = beta.mean(dim=-1)                    # [B]

        fused = self.ln_out(fused)

        # -------- arch regularization (仅对参与层) --------
        def _entropy_categorical(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
            return -(p * (p + eps).log()).sum(dim=-1)

        def _entropy_bernoulli(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
            return -(p * (p + eps).log() + (1 - p) * (1 - p + eps).log())

        fam_H  = _entropy_categorical(family_probs).mean()
        lvl_H  = _entropy_bernoulli(level_gate_used).mean()
        lvl_L1 = level_gate_used[:, 1:].mean() if K_actual > 1 else torch.zeros((), device=H.device)

        arch_loss = (self.reg_family_entropy * fam_H
                    + self.reg_level_entropy  * lvl_H
                    + self.reg_level_budget   * lvl_L1)

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

class TransformerSceneFiLMModel_SWT_Ada_Uncertain_Cxt(nn.Module):
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
        super(TransformerSceneFiLMModel_SWT_Ada_Uncertain_Cxt, self).__init__()
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
                self.wavelet_block = WaveletSWTBlockAuto_Cxt(
                    hidden_dim=hidden_dim,
                    L_max=wavelet_L_max,
                    families=wavelet_families,             # None -> ["db2","db4","db6"]
                    dropout=wavelet_dropout,
                    gate_channels_timewise=wavelet_gate_timewise,
                    init_lowfreq_bias=wavelet_lowfreq_bias,
                    residual_scale=0.2,
                    context_dim=context_dim,
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
            x_bt = self.wavelet_block(x_bt, context=context)  # [B,T,C]
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

