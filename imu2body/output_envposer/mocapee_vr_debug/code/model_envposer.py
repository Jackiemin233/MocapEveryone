# from __future__ import annotations

import sys, os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.append(parent_dir_path)
from imu2body.model_base import TransformerEncoderModel, TransformerSceneEncoderModel, TransformerSceneFiLMModel, \
                                TransformerEncoderModel_Uncertain, TransformerSceneFiLMModel_Uncertain, \
                                TransformerSceneFiLMModel_Uncertain_BiSmoother
# from imu2body.model_base import PointNet2SemSegSSGShape, PointNet, FPModule
# from imu2body.model_base import WaveletEmbedding
from imu2body.model_base_swt import WaveletSWTBlock, TransformerSceneFiLMModel_SWT, TransformerSceneFiLMModel_SWT_Uncertain, \
                                    TransformerSceneFiLMModel_Uncertain_SWT_BiSmoother
from imu2body.model_base_swt_new import TransformerSceneFiLMModel_WFiLM_Uncertain
from imu2body.model_base_swt_ada import TransformerSceneFiLMModel_SWT_Ada_Uncertain
from imu2body.model_base_swt_ada_cxt import TransformerSceneFiLMModel_SWT_Ada_Uncertain_Cxt
from imu2body.pointnet2 import PointNet2Encoder

import torch
import torch.nn as nn
import torch.nn.functional as F



# EnvPoser (minimal, reproducible PyTorch skeleton)
# -------------------------------------------------
# Two-stage model:
#   Stage I  : Uncertainty-aware initial motion estimation from sparse VR signals
#   Stage II : Environment-aware refinement with cross-attention + contact + geometry losses
#
# This file aims to mirror the key architectural and loss definitions in the paper
#   "EnvPoser: Environment-aware Realistic Human Motion Estimation from Sparse
#    Observations with Uncertainty Modeling" (Xia et al., 2025), with clean, modular code
#   that you can drop into your training loop. Some heavy parts (SMPL FK, COAP) are
#   implemented as interfaces or light-weight proxies so you can plug in your own.
#
# Key choices faithful to the paper:
# - Inputs: X ∈ R^{B×T×36} (head/hand position, rotation, linear velocity), VS ∈ R^{B×N×3}
# - Output: θ ∈ R^{B×T×132} (first 22 SMPL joints, 6D rotations), contact C ∈ R^{B×T×22}
# - Stage I: transformer encoder (8 heads), two-head MLP → mean θ~ and uncertainty δ
# - Stage II: vanilla PointNet/PointNet++ env encoder, cross-attention with spatial salience,
#             contact head (BCE), final motion decoder (L2 on θ̂_RM)
# - Losses: LM, Lδ (Stage I); LM', Lcontact, Lposi, LhAL, Lfc, Lgfh, Lgp, Lcoap (Stage II)
#
# NOTE:
# - PointNet++ is provided here as a simple PointNet-style fallback for easy run-ability.
#   If you have a PointNet++ encoder, replace EnvEncoder with your implementation.
# - SMPL forward kinematics (FK) and COAP collision require external components.
#   We expose callables/hooks so you can wire in your own modules.
# - X_new (T×40) = {X (36), h (1), θ_up (3)}. If you don't have h / θ_up yet, the helper
#   will fill zeros and you can plug in your actual values later.
#
# Author: ChatGPT (Minimal Reproducible Implementation)
#
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple
import math
import torch
import torch.nn as nn


# -----------------------------
# Utility: 6D → Rotation Matrix
# -----------------------------
# (Zhou et al. CVPR'19: On the Continuity of Rotation Representations in Neural Nets)

def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation (B,*,6) → rotation matrix (B,*,3,3).
    Args:
        x: [..., 6]
    Returns:
        R: [..., 3, 3]
    """
    x = x.view(*x.shape[:-1], 3, 2)  # [..., 3, 2]
    a1 = x[..., :, 0]
    a2 = x[..., :, 1]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    R = torch.stack([b1, b2, b3], dim=-1)  # [..., 3, 3]
    return R


# ------------------
# Positional Encoding
# ------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        T = x.shape[1]
        x = x + self.pe[:, :T]
        return self.dropout(x)


# -----------------------------
# Spatial Salience MLP (s_spatial)
# -----------------------------
class SpatialSalience(nn.Module):
    """Compute per-point spatial salience s_spatial ∈ R^{B×N} (later broadcast to T).
    Inputs are normalized point coords (relative to human center) with r and direction.
    """
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, pts_rel: torch.Tensor) -> torch.Tensor:
        # pts_rel: [B, N, 3], relative to human center
        r = torch.norm(pts_rel, dim=-1, keepdim=True)  # [B, N, 1]
        x = torch.cat([F.normalize(pts_rel, dim=-1), r], dim=-1)  # [B, N, 4]
        s = self.mlp(x).squeeze(-1)  # [B, N]
        return s


# -----------------------------
# Cross-Attention with s_spatial
# -----------------------------
class CrossAttention(nn.Module):
    def __init__(self, d_m: int = 256, d_e: int = 256, n_heads: int = 4):
        super().__init__()
        assert d_m % n_heads == 0
        self.d_m = d_m
        self.d_e = d_e
        self.n_heads = n_heads
        self.dk = d_m // n_heads
        self.Wq = nn.Linear(d_m, d_m)
        self.Wk = nn.Linear(d_e, d_m)
        self.Wv = nn.Linear(d_e, d_m)
        self.out = nn.Linear(d_m, d_m)

    def forward(self, ZM: torch.Tensor, Zenv_pts: torch.Tensor, s_spatial: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ZM        : [B, T, d_m]  motion embedding tokens (queries)
            Zenv_pts  : [B, N, d_e]  env per-point features (keys/values)
            s_spatial : [B, N]       per-point salience logits (added to attention scores)
        Returns:
            ZME       : [B, T, d_m]
        """
        B, T, _ = ZM.shape
        N = s_spatial.shape[1]
        Zenv_pts = Zenv_pts.unsqueeze(1).expand(-1, N, -1)
        Q = self.Wq(ZM)      # [B, T, d_m]
        K = self.Wk(Zenv_pts)  # [B, N, d_m]
        V = self.Wv(Zenv_pts)  # [B, N, d_m]

        # reshape to heads
        def split_heads(x):
            return x.view(B, -1, self.n_heads, self.dk).transpose(1, 2)  # [B, H, T/N, dk]

        Qh = split_heads(Q)  # [B, H, T, dk]
        Kh = split_heads(K)  # [B, H, N, dk]
        Vh = split_heads(V)  # [B, H, N, dk]

        # scaled dot-product attention with additive s_spatial
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(self.dk)  # [B, H, T, N]
        scores = scores + s_spatial.unsqueeze(1).unsqueeze(2)                 # broadcast over heads & time
        attn = torch.softmax(scores, dim=-1)                                   # [B, H, T, N]
        out = torch.matmul(attn, Vh)                                           # [B, H, T, dk]
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_m)           # [B, T, d_m]
        return self.out(out)


# -----------------------------------------
# Stage I: Uncertainty-aware Motion Encoder
# -----------------------------------------
class Stage1Uncertainty(nn.Module):
    def __init__(self,
                 input_dim: int = 36,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 dropout: float = 0.1,
                 out_dim: int = 132):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
        )
        self.pos = PositionalEncoding(d_model, dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                               dim_feedforward=d_model*4,
                                               dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.pose_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(True), nn.Linear(d_model, out_dim))
        self.unc_head  = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(True), nn.Linear(d_model, out_dim))

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Args: X [B, T, 36]  Returns: (Z_H, theta_mean, theta_logvar)
        - theta_logvar is unconstrained; convert to variance via softplus if needed
        """
        Zs = self.inp(X)
        Zs = self.pos(Zs)
        ZH = self.encoder(Zs)                        # [B, T, d]
        theta_mean = self.pose_head(ZH)              # [B, T, 132]
        theta_logvar = self.unc_head(ZH)             # [B, T, 132]
        return ZH, theta_mean, theta_logvar

    @staticmethod
    def sample(theta_mean: torch.Tensor, theta_logvar: torch.Tensor, num_samples: int = 1,
               clamp_logvar: Optional[Tuple[float, float]] = (-10.0, 5.0)) -> torch.Tensor:
        """Reparameterized sampling θ = θ~ + δ·ε.
        Returns: [B, S, T, 132]
        """
        if clamp_logvar is not None:
            theta_logvar = torch.clamp(theta_logvar, clamp_logvar[0], clamp_logvar[1])
        delta = F.softplus(theta_logvar)  # ensure positivity
        B, T, D = theta_mean.shape
        eps = torch.randn(B, num_samples, T, D, device=theta_mean.device, dtype=theta_mean.dtype)
        mean = theta_mean.unsqueeze(1).expand(-1, num_samples, -1, -1)
        delta = delta.unsqueeze(1).expand_as(mean)
        return mean + delta * eps


# -------------------------------------------------
# Stage II: Environment-aware Refinement Components
# -------------------------------------------------
class MotionEmbedding(nn.Module):
    """X_new = concat(X(36), h(1), up(3)) → 40-D per frame.
    Final concat: [θ(132), p_head(3), X_new(40)] = 175 → Linear → 256
    """
    def __init__(self, in_dim: int = 175, d_model: int = 256):
        super().__init__()
        self.fc = nn.Linear(in_dim, d_model)

    def forward(self, theta: torch.Tensor, p_head: torch.Tensor, X_new: torch.Tensor) -> torch.Tensor:
        x = torch.cat([theta, p_head, X_new], dim=-1)  # [B, T, 175]
        return F.relu(self.fc(x))                      # [B, T, 256]


def build_X_new(X: torch.Tensor,
                head_height: Optional[torch.Tensor] = None,
                head_up: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Ensure X_new has 40 dims per frame = {X(36), h(1), up(3)}.
    If h / up are missing, zeros are used (you can replace with your actual module).
    Args:
      X: [B, T, 36]
      head_height: [B, T, 1] or None
      head_up: [B, T, 3] or None
    Returns: X_new [B, T, 40]
    """
    B, T, _ = X.shape
    if head_height is None:
        head_height = torch.zeros(B, T, 1, device=X.device, dtype=X.dtype)
    if head_up is None:
        head_up = torch.zeros(B, T, 3, device=X.device, dtype=X.dtype)
    return torch.cat([X, head_height, head_up], dim=-1)


class RefinementBlock(nn.Module):
    def __init__(self, d_model: int = 256, d_env: int = 256):
        super().__init__()
        self.cross = CrossAttention(d_m=d_model, d_e=d_env, n_heads=4)
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(True),
            nn.Linear(d_model, d_model)
        )

    def forward(self, ZM: torch.Tensor, Zenv_pts: torch.Tensor, s_spatial: torch.Tensor) -> torch.Tensor:
        ZME = self.cross(ZM, Zenv_pts, s_spatial)  # [B, T, d]
        return self.mlp(torch.cat([ZME, ZM], dim=-1))


class ContactHead(nn.Module):
    def __init__(self, d_model: int = 256, out_joints: int = 22):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model + 35, d_model), nn.ReLU(True),
            nn.Linear(d_model, out_joints)
        )

    def forward(self, ZRM: torch.Tensor, X_new: torch.Tensor) -> torch.Tensor:
        logits = self.net(torch.cat([ZRM, X_new], dim=-1))
        return torch.sigmoid(logits)  # [B, T, 22]


class MotionDecoder(nn.Module):
    def __init__(self, d_model: int = 256, out_dim: int = 132, out_contact: int = 22):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model + out_contact, d_model), nn.ReLU(True),
            nn.Linear(d_model, out_dim)
        )

    def forward(self, ZRM: torch.Tensor, C_hat: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.cat([ZRM, C_hat], dim=-1))


# ---------------------
# Full EnvPoser Module
# ---------------------
@dataclass
class EnvPoserConfig:
    input_dim: int = 31
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    out_dim: int = 135
    out_contact: int = 22
    env_feat_dim: int = 256
    window_len: int = 40
    lambda_M: float = 1.0
    lambda_delta: float = 1e-3
    # Stage-II weights (set per your training script)
    lambda_posi: float = 1.0
    lambda_hAL: float = 1.0
    lambda_fc: float = 1.0
    lambda_contact: float = 1.0
    lambda_gfh: float = 1.0
    lambda_gp: float = 1.0
    lambda_coap: float = 1.0


class EnvPoser(nn.Module):
    def __init__(self, cfg: EnvPoserConfig):
        super().__init__()
        self.cfg = cfg
        # Stage I
        self.stage1 = Stage1Uncertainty(
            input_dim=cfg.input_dim,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            out_dim=cfg.out_dim,
        )
        # Stage II
        self.env_encoder = PointNet2Encoder(hidden_dim=cfg.env_feat_dim)
        self.spatial_salience = SpatialSalience()
        self.motion_embed = MotionEmbedding(in_dim=173, d_model=cfg.d_model)
        self.refine = RefinementBlock(d_model=cfg.d_model, d_env=cfg.env_feat_dim)
        self.contact_head = ContactHead(d_model=cfg.d_model, out_joints=cfg.out_contact)
        self.motion_dec = MotionDecoder(d_model=cfg.d_model, out_dim=cfg.out_dim, out_contact=cfg.out_contact)

    # -----------------
    # Stage I forward
    # -----------------
    def stage1_forward(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        ZH, theta_mean, theta_logvar = self.stage1(X)
        return {"Z_H": ZH, "theta_mean": theta_mean, "theta_logvar": theta_logvar}

    # -----------------
    # Stage II forward
    # -----------------
    def stage2_forward(self,
                        X: torch.Tensor,
                        theta_sample: torch.Tensor,
                        VS: torch.Tensor,
                        p_head: Optional[torch.Tensor] = None,
                        head_height: Optional[torch.Tensor] = None,
                        head_up: Optional[torch.Tensor] = None,
                        human_center: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, T, _ = X.shape
        if p_head is None:
            p_head = torch.zeros(B, T, 3, device=X.device, dtype=X.dtype)
        X_new = build_X_new(X, head_height, head_up)  # [B, T, 35]

        # Env encoding
        input_pc = VS - VS.mean(dim=1, keepdim=True)
        input_pc = input_pc / (input_pc.norm(dim=2, keepdim=True).amax(dim=1, keepdim=True) + 1e-8)
        Zenv = self.env_encoder(input_pc.permute(0, 2, 1))       # B,N,3 -> B,3,N

        # Spatial salience (relative to center). If not provided, use zeros.
        if human_center is None:
            human_center = p_head.mean(dim=1, keepdim=True)
        pts_rel = VS - human_center  # broadcast center over N
        s_spatial = self.spatial_salience(pts_rel)  # [B, N]

        # Motion embedding with sampled θ
        ZM = self.motion_embed(theta_sample, p_head, X_new)  # [B, T, d]

        # Cross-attention + fusion → Z_RM
        ZRM = self.refine(ZM, Zenv, s_spatial)              # B, T, d

        # Contact and final motion
        C_hat = self.contact_head(ZRM, X_new)              # [B, T, 22]
        theta_hat = self.motion_dec(ZRM, C_hat)            # [B, T, 132]
        return {"Z_RM": ZRM, "C_hat": C_hat, "theta_hat": theta_hat, "s_spatial": s_spatial}

    # -----------------
    # Full forward (both stages)
    # -----------------
    def forward(self,
                X: torch.Tensor,
                VS: Optional[torch.Tensor] = None,
                p_head: Optional[torch.Tensor] = None,
                head_height: Optional[torch.Tensor] = None,
                head_up: Optional[torch.Tensor] = None,
                human_center: Optional[torch.Tensor] = None,
                num_samples: int = 1,
                sample_from_mean: bool = False) -> Dict[str, torch.Tensor]:
        out1 = self.stage1_forward(X)
        theta_mean, theta_logvar = out1["theta_mean"], out1["theta_logvar"]
        if sample_from_mean:
            theta_sample = theta_mean
        else:
            # default: one sample for refinement; set num_samples>1 externally to explore hypotheses
            theta_sample = Stage1Uncertainty.sample(theta_mean, theta_logvar, num_samples=1).squeeze(1)
        out = {**out1, "theta_sample": theta_sample}

        if VS is not None:
            out2 = self.stage2_forward(X, theta_sample, VS, p_head, head_height, head_up, human_center)
            out.update(out2)        # {"Z_RM": ZRM, "C_hat": C_hat, "theta_hat": theta_hat, "s_spatial": s_spatial}
        return out



# --------------------------
# Example COAP proxy (toy!)
# --------------------------
class SimpleCOAPProxy(nn.Module):
    """A cheap proxy for COAP: penalize close vertex–point distances with a sigmoid.
    Replace with a proper COAP module for faithful reproduction.
    """
    def __init__(self, margin: float = 0.02):
        super().__init__()
        self.margin = margin

    def forward(self, verts_bt: torch.Tensor, VS: torch.Tensor) -> torch.Tensor:
        # verts_bt: [B, T, Nv, 3]; VS: [B, N, 3]
        B, T, Nv, _ = verts_bt.shape
        N = VS.shape[1]
        # compute min distance from each vertex to any scene point (vectorized approx)
        # flatten time: [B*T, Nv, 3]
        vt = verts_bt.reshape(B*T, Nv, 3)
        ps = VS.unsqueeze(1).expand(B, Nv, N, 3).reshape(B, Nv, N, 3)
        ps = ps.repeat(T, 1, 1, 1)  # [B*T, Nv, N, 3]
        d2 = torch.sum((vt.unsqueeze(2) - ps) ** 2, dim=-1)  # [B*T, Nv, N]
        dmin = torch.sqrt(torch.clamp(d2.min(dim=-1).values, min=1e-8))  # [B*T, Nv]
        # penalize if too close (< margin)
        logits = (self.margin - dmin) * 50.0
        loss = torch.sigmoid(logits).mean()
        return loss

    
def load_envposer_model(data_config, model_config):
    cfg = EnvPoserConfig()
    return EnvPoser(cfg)


# --------------------------
# Wiring / Usage (pseudo)
# --------------------------
if __name__ == "__main__":
    cfg = EnvPoserConfig()
    model = EnvPoser(cfg)

    B, T, N = 2, 40, 1000
    X = torch.randn(B, T, 36)
    VS = torch.randn(B, N, 3)
    p_head = torch.randn(B, T, 3)

    # Forward both stages (sampling from mean for determinism here)
    out = model(X, VS=VS, p_head=p_head, sample_from_mean=True)
    theta_mean = out["theta_mean"]       # [B, T, 132]
    theta_logvar = out["theta_logvar"]   # [B, T, 132]
    theta_hat = out.get("theta_hat")      # [B, T, 132]
    C_hat = out.get("C_hat")             # [B, T, 22]

    # Dummy GT
    theta_gt = torch.randn_like(theta_mean)

    # Stage I loss
    loss1 = model.loss_stage1(theta_mean, theta_logvar, theta_gt)

    # Stage II loss (without FK/COAP → only LM')
    loss2 = model.loss_stage2(theta_hat, theta_gt)

    print({k: float(v) for k, v in loss1.items()})
    print({k: float(v) for k, v in loss2.items()})