# ============ loss_scale_separated_grouped.py ============
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# def _design_lowpass_fir(kernel_size: int, cutoff: float) -> torch.Tensor:
#     assert kernel_size % 2 == 1 and 0.0 < cutoff < 1.0
#     K = kernel_size
#     M = (K - 1) // 2
#     n = torch.arange(K, dtype=torch.float32) - M
#     h = torch.where(n == 0, 2 * cutoff * torch.ones_like(n),
#                     torch.sin(2 * torch.pi * cutoff * n) / (torch.pi * n))
#     w = 0.5 - 0.5 * torch.cos(2 * torch.pi * (torch.arange(K) / (K - 1)))  # Hann
#     h = (h * w).to(torch.float32)
#     h = h / h.sum()
#     return h.view(1, 1, K)  # [1,1,K]

# class _DepthwiseFIR(nn.Module):
#     def __init__(self, D: int, kernel: torch.Tensor, causal: bool = False):
#         super().__init__()
#         self.D = D
#         self.K = int(kernel.numel())
#         self.causal = causal
#         self.register_buffer("w", kernel.repeat(D, 1, 1))  # [D,1,K]

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B,T,D] or [B,D,T]
#         if x.dim() != 3: raise ValueError
#         if x.shape[1] < x.shape[2]:
#             x = x.permute(0, 2, 1)  # [B,D,T]
#             back = True
#         else:
#             back = False
#         B,D,T = x.shape
#         assert D == self.D
#         if self.causal:
#             pad = (self.K - 1, 0)
#         else:
#             L = (self.K - 1)//2
#             pad = (L, self.K-1-L)
#         y = F.conv1d(F.pad(x, pad, mode="reflect"), self.w, groups=D)
#         return y.permute(0, 2, 1) if back else y

# def _charb(x: torch.Tensor, eps: float=1e-3):  # Charbonnier
#     return torch.sqrt(x*x + eps*eps)

# def _diff(x: torch.Tensor, order: int) -> torch.Tensor:
#     for _ in range(order):
#         x = x[:,1:,:] - x[:,:-1,:]
#     return x

# class ScaleSeparatedLossGrouped(nn.Module):
#     """
#     对 pos 与 rot6d 分别做低通/高通分带损。
#     - pos 更强调低频稳定（低 cutoff、更大低频权重）
#     - rot6d 稍宽带，兼顾细节
#     """
#     def __init__(self,
#                  pos_dim: int = 3,
#                  rot_dim: int = 132,
#                  # pos 低通
#                  pos_kernel: int = 15, pos_cutoff: float = 0.20,
#                  # rot 低通
#                  rot_kernel: int = 11, rot_cutoff: float = 0.30,
#                  causal: bool = False,
#                  # 权重
#                  w_lf_pos: float = 1.0,
#                  w_hf_pos: float = 0.4,
#                  w_vel_pos: float = 0.2,
#                  w_lf_rot: float = 1.0,
#                  w_hf_rot: float = 0.6,
#                  w_vel_rot: float = 0.0,
#                  robust_eps: float = 1e-3):
#         super().__init__()
#         self.pos_dim, self.rot_dim = pos_dim, rot_dim
#         self.pos_lpf = _DepthwiseFIR(pos_dim, _design_lowpass_fir(pos_kernel, pos_cutoff), causal)
#         self.rot_lpf = _DepthwiseFIR(rot_dim, _design_lowpass_fir(rot_kernel, rot_cutoff), causal)
#         self.w_lf_pos, self.w_hf_pos, self.w_vel_pos = w_lf_pos, w_hf_pos, w_vel_pos
#         self.w_lf_rot, self.w_hf_rot, self.w_vel_rot = w_lf_rot, w_hf_rot, w_vel_rot
#         self.eps = robust_eps

#     def _one_group(self, yp, yt, lpf, w_lf, w_hf, w_vel):
#         lf_p, lf_t = lpf(yp), lpf(yt)
#         hf_p, hf_t = yp - lf_p, yt - lf_t
#         L_lf = _charb(lf_p - lf_t, self.eps).mean()
#         L_hf = _charb(hf_p - hf_t, self.eps).mean()
#         L_vel = torch.tensor(0., device=yp.device)
#         if w_vel > 0:
#             vp, vt = _diff(lf_p,1), _diff(lf_t,1)
#             L_vel = _charb(vp - vt, self.eps).mean()
#         return w_lf*L_lf + w_hf*L_hf + w_vel*L_vel, {"lf":L_lf.detach(),"hf":L_hf.detach(),"vel":L_vel.detach()}

#     def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
#         assert y_pred.shape == y_true.shape and y_pred.dim()==3
#         pos_p, rot_p = y_pred[...,:self.pos_dim], y_pred[...,self.pos_dim:]
#         pos_t, rot_t = y_true[...,:self.pos_dim], y_true[...,self.pos_dim:]

#         Lpos, auxp = self._one_group(pos_p, pos_t, self.pos_lpf, self.w_lf_pos, self.w_hf_pos, self.w_vel_pos)
#         Lrot, auxr = self._one_group(rot_p, rot_t, self.rot_lpf, self.w_lf_rot, self.w_hf_rot, self.w_vel_rot)
#         loss = Lpos + Lrot
#         aux = {"pos_lf":auxp["lf"],"pos_hf":auxp["hf"],"pos_vel":auxp["vel"],
#                "rot_lf":auxr["lf"],"rot_hf":auxr["hf"],"rot_vel":auxr["vel"]}
#         return loss, aux

# class STFTBandLoss(nn.Module):
#     """
#     对 [B,T,D] 做 STFT，分频带（低/高）比较幅度谱。
#     可以为 pos 与 rot6d 设不同的频带权重。
#     """
#     def __init__(self,
#                  n_fft:int=16, hop:int=4, win:int=16,
#                  # 频带划分：以频率bin索引为界（0 = DC，n_fft//2 = Nyquist）
#                  low_max_bin:int=3,   # 低频最高 bin（含）
#                  high_min_bin:int=5,  # 高频起始 bin（含）
#                  w_lf_pos:float=1.0, w_hf_pos:float=0.2,
#                  w_lf_rot:float=0.8, w_hf_rot:float=0.6):
#         super().__init__()
#         self.n_fft, self.hop, self.win = n_fft, hop, win
#         self.low_max_bin = low_max_bin
#         self.high_min_bin = high_min_bin
#         self.w_lf_pos, self.w_hf_pos = w_lf_pos, w_hf_pos
#         self.w_lf_rot, self.w_hf_rot = w_lf_rot, w_hf_rot
#         window = torch.hann_window(win)
#         self.register_buffer("window", window, persistent=False)

#     def _stft_mag(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B,T,D] → reshape → [B*D, T]
#         B,T,D = x.shape
#         xd = x.permute(0,2,1).reshape(B*D, T)
#         X = torch.stft(xd, n_fft=self.n_fft, hop_length=self.hop,
#                        win_length=self.win, window=self.window,
#                        return_complex=True, center=True, pad_mode="reflect")
#         mag = X.abs()  # [B*D, F, frames]
#         Freq, Frm = mag.shape[1], mag.shape[2]
#         # reshape back to [B, D, F, frames]
#         mag = mag.view(B, D, Freq, Frm)
#         return mag

#     def _band_loss(self, Mp:torch.Tensor, Mt:torch.Tensor,
#                    band:Tuple[int,int]) -> torch.Tensor:
#         # Mp/Mt: [B,D,F,Fr]; band=(lo,hi) inclusive
#         lo, hi = band
#         lo = max(0, lo); hi = min(Mp.shape[2]-1, hi)
#         if hi < lo: return torch.zeros((), device=Mp.device)
#         diff = (Mp[:,:,lo:hi+1,:] - Mt[:,:,lo:hi+1,:]).abs().mean()
#         return diff

#     def forward(self, y_pred:torch.Tensor, y_true:torch.Tensor) -> Tuple[torch.Tensor,dict]:
#         assert y_pred.shape == y_true.shape and y_pred.dim()==3
#         B,T,D = y_pred.shape
#         pos_dim = 3

#         Mp = self._stft_mag(y_pred)
#         Mt = self._stft_mag(y_true)

#         # 频带定义
#         low_band  = (0, self.low_max_bin)
#         high_band = (self.high_min_bin, self.n_fft//2)

#         Mp_pos, Mt_pos = Mp[:, :pos_dim], Mt[:, :pos_dim]
#         Mp_rot, Mt_rot = Mp[:, pos_dim:], Mt[:, pos_dim:]

#         L_lf_pos  = self._band_loss(Mp_pos, Mt_pos, low_band)
#         L_hf_pos  = self._band_loss(Mp_pos, Mt_pos, high_band)
#         L_lf_rot  = self._band_loss(Mp_rot, Mt_rot, low_band)
#         L_hf_rot  = self._band_loss(Mp_rot, Mt_rot, high_band)

#         loss = (self.w_lf_pos*L_lf_pos + self.w_hf_pos*L_hf_pos
#                 + self.w_lf_rot*L_lf_rot + self.w_hf_rot*L_hf_rot)
#         aux = {"lf_pos":L_lf_pos.detach(),"hf_pos":L_hf_pos.detach(),
#                "lf_rot":L_lf_rot.detach(),"hf_rot":L_hf_rot.detach()}
#         return loss, aux

def _design_lowpass_fir(kernel_size: int, cutoff: float) -> torch.Tensor:
    assert kernel_size % 2 == 1 and 0.0 < cutoff < 1.0
    K = kernel_size
    M = (K - 1) // 2
    n = torch.arange(K, dtype=torch.float32) - M
    h = torch.where(n == 0, 2*cutoff*torch.ones_like(n),
                    torch.sin(2*torch.pi*cutoff*n)/(torch.pi*n))
    w = 0.5 - 0.5*torch.cos(2*torch.pi*(torch.arange(K)/(K-1)))  # Hann
    h = (h*w).to(torch.float32)
    h = h / h.sum()
    return h.view(1,1,K)

class _DepthwiseFIR1D(nn.Module):
    def __init__(self, C: int, kernel: torch.Tensor, causal: bool=False):
        super().__init__()
        self.K = int(kernel.numel())
        self.causal = causal
        self.register_buffer("w", kernel.repeat(C,1,1))  # [C,1,K]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,C] → [B,C,T]
        x = x.permute(0,2,1)
        if self.causal:
            pad = (self.K-1, 0)
        else:
            L = (self.K-1)//2
            pad = (L, self.K-1-L)
        y = F.conv1d(F.pad(x, pad, mode="reflect"), self.w, groups=x.shape[1])
        return y.permute(0,2,1)  # [B,T,C]

class ScaleSeparated3D(nn.Module):
    """
    Scale-separated loss in 3D joint space
    在 3D 关节空间做分带损：低频对齐 + 高频残差 + (可选)低频速度一致性
    y_pred, y_true: [B, T, J, 3]
    """
    def __init__(self, J:int,
                 lp_kernel:int=15, lp_cutoff:float=0.25, causal:bool=False,
                 w_lf:float=1.0, w_hf:float=0.5, w_vel:float=0.2,
                 eps:float=1e-3):
        super().__init__()
        self.fir = _DepthwiseFIR1D(J*3, _design_lowpass_fir(lp_kernel, lp_cutoff), causal)
        self.w_lf, self.w_hf, self.w_vel = w_lf, w_hf, w_vel
        self.eps = eps
        self.J = J

    def _charb(self, x):  # Charbonnier
        return torch.sqrt(x*x + self.eps*self.eps)

    def _diff(self, x, order=1):  # x: [B,T,C]
        for _ in range(order):
            x = x[:,1:,:] - x[:,:-1,:]
        return x

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        B,T,J,_ = y_pred.shape
        yp = y_pred.reshape(B,T,J*3)
        yt = y_true.reshape(B,T,J*3)

        lf_p = self.fir(yp); lf_t = self.fir(yt)
        hf_p = yp - lf_p;    hf_t = yt - lf_t

        L_lf  = self._charb(lf_p - lf_t).mean()
        L_hf  = self._charb(hf_p - hf_t).mean()

        L_vel = torch.tensor(0., device=yp.device)
        if self.w_vel > 0:
            vp = self._diff(lf_p,1); vt = self._diff(lf_t,1)
            L_vel = self._charb(vp - vt).mean()

        loss = self.w_lf*L_lf + self.w_hf*L_hf + self.w_vel*L_vel
        return loss, {"lf":L_lf.detach(), "hf":L_hf.detach(), "vel":L_vel.detach()}

class RobustScaleSeparated3D(nn.Module):
    """
    分带损（稳健版）:
    - 在 "骨盆相对" 关节位置上做 (排除 root 位移)
    - 支持 per-joint std 归一化 + 关节权重
    - 时域低通 + 高频残差 + (可选)低频速度一致
    - 支持 warmup 系数
    y_pred, y_true: [B, T, J, 3] （全局）
    pelvis_idx: 骨盆关节索引（常用 0）
    x_std: [1, 1, J, 3] 或 [B,1,J,3]，用于归一化（可 None）
    joint_weight: [J] 或 [1,J,1]，可给脚/末端更高权重
    """
    def __init__(self, J:int, pelvis_idx:int=0,
                 lp_kernel:int=11, lp_cutoff:float=0.30, causal:bool=False,
                 w_lf:float=1.0, w_hf:float=0.4, w_vel:float=0.15,
                 eps:float=1e-3):
        super().__init__()
        self.J = J
        self.pelvis_idx = pelvis_idx
        self.fir = _DepthwiseFIR1D(J*3, _design_lowpass_fir(lp_kernel, lp_cutoff), causal)
        self.w_lf, self.w_hf, self.w_vel = w_lf, w_hf, w_vel
        self.eps = eps

    @staticmethod
    def _charb(x, eps): return torch.sqrt(x*x + eps*eps)
    @staticmethod
    def _diff(x, order=1):
        for _ in range(order):
            x = x[:,1:,:] - x[:,:-1,:]
        return x

    def forward(self, y_pred, y_true,
                x_std=None, joint_weight=None, warmup_alpha:float=1.0):
        # 1) 转为骨盆相对坐标，排除 root 平移的干扰
        pelv_p = y_pred[:,:,self.pelvis_idx:self.pelvis_idx+1,:]  # [B,T,1,3]
        pelv_t = y_true[:,:,self.pelvis_idx:self.pelvis_idx+1,:]
        yp_rel = y_pred - pelv_p
        yt_rel = y_true - pelv_t

        # 2) 归一化（与原 pos_diff 对齐）
        if x_std is not None:
            yp_rel = yp_rel / (x_std + 1e-8)
            yt_rel = yt_rel / (x_std + 1e-8)

        # 3) 展平到 [B,T,J*3]
        B,T,J,_ = yp_rel.shape
        yp = yp_rel.reshape(B, T, J*3)
        yt = yt_rel.reshape(B, T, J*3)

        # 4) 低通/高通
        lf_p = self.fir(yp); lf_t = self.fir(yt)
        hf_p = yp - lf_p;    hf_t = yt - lf_t

        # 5) 关节权重（可选，例如给脚/末端更高权重）
        if joint_weight is not None:
            # joint_weight: [J] 或 [1,J,1]，广播到 [B,T,J,3]
            w = joint_weight.view(1,1,J,1).to(yp.device)
            w = w.repeat(B,T,1,3).reshape(B,T,J*3)
        else:
            w = None

        def _mean_w(x):
            if w is None: return x.mean()
            return (x*w).sum()/(w.sum()+1e-8)

        # 6) 分带损 + 速度一致（都乘以 warmup_alpha）
        L_lf  = self._charb(lf_p - lf_t, self.eps)
        L_hf  = self._charb(hf_p - hf_t, self.eps)
        L_lf  = _mean_w(L_lf) * warmup_alpha
        L_hf  = _mean_w(L_hf) * warmup_alpha

        L_vel = torch.tensor(0., device=yp.device)
        if self.w_vel > 0:
            vp = self._diff(lf_p,1); vt = self._diff(lf_t,1)
            L_vel = self._charb(vp - vt, self.eps)
            L_vel = _mean_w(L_vel) * warmup_alpha

        loss = self.w_lf*L_lf + self.w_hf*L_hf + self.w_vel*L_vel
        aux = {"lf":L_lf.detach(), "hf":L_hf.detach(), "vel":L_vel.detach()}
        return loss, aux


class STFTBand3D(nn.Module):
    """
    Spectral loss in 3D joint space
    在 3D 关节空间做 STFT 分带损。
    y_pred, y_true: [B, T, J, 3]
    """
    def __init__(self, n_fft=16, hop=4, win=16,
                 low_max_bin=3, high_min_bin=5,
                 w_low=0.5, w_high=0.5):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.low_max_bin, self.high_min_bin = low_max_bin, high_min_bin
        window = torch.hann_window(win)
        self.register_buffer("window", window, persistent=False)
        self.w_low, self.w_high = w_low, w_high

    def _stft_mag(self, x):  # x: [B,T,J,3]
        B,T,J,_ = x.shape
        xd = x.reshape(B, T, J*3).permute(0,2,1).reshape(B*J*3, T)  # [B*J*3, T]
        X = torch.stft(xd, n_fft=self.n_fft, hop_length=self.hop,
                       win_length=self.win, window=self.window,
                       return_complex=True, center=True, pad_mode="reflect")
        mag = X.abs()  # [B*J*3, F, frames]
        F, Fr = mag.shape[1], mag.shape[2]
        mag = mag.view(B, J*3, F, Fr)
        return mag

    def _band_loss(self, Mp, Mt, lo, hi):
        lo = max(0, lo); hi = min(Mp.shape[2]-1, hi)
        if hi < lo: return torch.zeros((), device=Mp.device)
        return (Mp[:,:,lo:hi+1,:] - Mt[:,:,lo:hi+1,:]).abs().mean()

    def forward(self, y_pred, y_true):
        Mp = self._stft_mag(y_pred)
        Mt = self._stft_mag(y_true)
        lowL  = self._band_loss(Mp, Mt, 0, self.low_max_bin)
        highL = self._band_loss(Mp, Mt, self.high_min_bin, self.n_fft//2)
        loss = self.w_low*lowL + self.w_high*highL
        return loss, {"stft_low":lowL.detach(), "stft_high":highL.detach()}
