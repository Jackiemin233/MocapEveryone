import PIL
import numpy as np
import torch
import einops

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.image import ImgNorm
from mast3r.model import AsymmetricMASt3R
from mast3r_slam.retrieval_database import RetrievalDatabase
from mast3r_slam.config import config
import mast3r_slam.matching as matching

import einops
from einops import rearrange

from lib.vis.tools import vis_img

def load_mast3r(path=None, device="cuda"):
    weights_path = (
        "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        if path is None
        else path
    )
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    return model

def load_retriever(mast3r_model, retriever_path=None, device="cuda"):
    retriever_path = (
        "data/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
        if retriever_path is None
        else retriever_path
    )
    retriever = RetrievalDatabase(retriever_path, backbone=mast3r_model, device=device)
    return retriever

@torch.inference_mode
def decoder(model, feat1, feat2, pos1, pos2, shape1, shape2, mask1 = None, mask2 = None):
    dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2, mask1, mask2) # Feat2 is the reference image
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1, res2


def downsample(X, C, D, Q):
    downsample = config["dataset"]["img_downsample"]
    if downsample > 1:
        X = X[..., ::downsample, ::downsample, :].contiguous()
        C = C[..., ::downsample, ::downsample].contiguous()
        D = D[..., ::downsample, ::downsample, :].contiguous()
        Q = Q[..., ::downsample, ::downsample].contiguous()
    return X, C, D, Q

@torch.inference_mode
def mast3r_symmetric_inference(model, frame_i, frame_j):
    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = model._encode_image(
            frame_i.img, frame_i.img_true_shape
        )
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = model._encode_image(
            frame_j.img, frame_j.img_true_shape
        )

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape
    
    mask1, mask2 = frame_i.mask.unsqueeze(0), frame_j.mask.unsqueeze(0)
    mask_i = _resize_mask(mask1, mask1.shape) 
    mask_j = _resize_mask(mask2, mask2.shape)
    
    # res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2)
    # res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape2, shape1)
    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2, mask_i, mask_j)
    res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape2, shape1, mask_j, mask_i)
    res = [res11, res21, res22, res12]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


# NOTE: Assumes img shape the same
@torch.inference_mode
def mast3r_decode_symmetric_batch(
    model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j, mask_i = None, mask_j = None
):
    B = feat_i.shape[0]
    X, C, D, Q = [], [], [], []
    if mask_i != None and mask_j != None:
        mask1 = _resize_mask(mask_i, mask_i.shape)
        mask2 = _resize_mask(mask_j, mask_j.shape)
    
    for b in range(B):
        feat1 = feat_i[b][None]
        feat2 = feat_j[b][None]
        pos1 = pos_i[b][None]
        pos2 = pos_j[b][None]
        
        if mask1 != None and mask2 != None:
            mask_i = mask1[b][None]
            mask_j = mask2[b][None]
            
        res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape_i[b], shape_j[b], mask_i, mask_j)
        res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape_j[b], shape_i[b], mask_j, mask_i)
        res = [res11, res21, res22, res12]
        Xb, Cb, Db, Qb = zip(
            *[
                (r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0])
                for r in res
            ]
        )
        X.append(torch.stack(Xb, dim=0))
        C.append(torch.stack(Cb, dim=0))
        D.append(torch.stack(Db, dim=0))
        Q.append(torch.stack(Qb, dim=0))

    X, C, D, Q = (
        torch.stack(X, dim=1),
        torch.stack(C, dim=1),
        torch.stack(D, dim=1),
        torch.stack(Q, dim=1),
    )
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q

@torch.inference_mode
def mast3r_inference_mono(model, frame):
    if frame.feat is None:
        frame.feat, frame.pos, _ = model._encode_image(frame.img, frame.img_true_shape)

    feat = frame.feat
    pos = frame.pos
    shape = frame.img_true_shape
    
    mask = frame.mask.unsqueeze(0) # [1 512 384]
    mask = _resize_mask(mask, mask.shape) 
    #mask = None
    
    res11, res21 = decoder(model, feat, feat, pos, pos, shape, shape, mask, mask)
    res = [res11, res21]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)

    Xii, _ = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, _ = einops.rearrange(C, "b h w -> b (h w) 1")
    if frame.mask != None: #NOTE 4.13 : confidence Mask - NJ
        Xii = Xii * einops.rearrange((1-frame.mask), "h w -> (h w) 1")
        Cii = Cii * einops.rearrange((1-frame.mask), "h w -> (h w) 1")
       
    return Xii, Cii

def mast3r_match_symmetric(model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j, mask_i = None, mask_j = None):
    X, C, D, Q = mast3r_decode_symmetric_batch(
        model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j, mask_i, mask_j
    )

    # Ordering 4xbxhxwxc
    b = X.shape[1]

    Xii, Xji, Xjj, Xij = X[0], X[1], X[2], X[3]
    Dii, Dji, Djj, Dij = D[0], D[1], D[2], D[3]
    Qii, Qji, Qjj, Qij = Q[0], Q[1], Q[2], Q[3]
    
    Xii = mask_i.unsqueeze(-1) * Xii
    Dii = mask_i.unsqueeze(-1) * Dii
    Qii = mask_i * Qii
    
    Xjj = mask_j.unsqueeze(-1) * Xjj
    Djj = mask_j.unsqueeze(-1) * Djj
    Qjj = mask_j * Qjj

    # Always matching both
    X11 = torch.cat((Xii, Xjj), dim=0) # ii mask jj mask 动态区域
    X21 = torch.cat((Xji, Xij), dim=0) # J帧相机下 i帧点图的点图 #可视化 ji ij
    D11 = torch.cat((Dii, Djj), dim=0) 
    D21 = torch.cat((Dji, Dij), dim=0) 

    idx_1_to_2, valid_match_2 = matching.match(X11, X21, D11, D21)

    # TODO: Avoid this
    match_b = X11.shape[0] // 2
    idx_i2j = idx_1_to_2[:match_b]
    idx_j2i = idx_1_to_2[match_b:]
    valid_match_j = valid_match_2[:match_b]
    valid_match_i = valid_match_2[match_b:]

    return (
        idx_i2j,
        idx_j2i,
        valid_match_j,
        valid_match_i,
        Qii.view(b, -1, 1),
        Qjj.view(b, -1, 1),
        Qji.view(b, -1, 1),
        Qij.view(b, -1, 1),
    )

@torch.inference_mode
def mast3r_asymmetric_inference(model, frame_i, frame_j):
    if frame_i.feat is None:
        frame_i.feat, frame_i.pos, _ = model._encode_image(
            frame_i.img, frame_i.img_true_shape
        )
    if frame_j.feat is None:
        frame_j.feat, frame_j.pos, _ = model._encode_image(
            frame_j.img, frame_j.img_true_shape
        )

    feat1, feat2 = frame_i.feat, frame_j.feat
    pos1, pos2 = frame_i.pos, frame_j.pos
    shape1, shape2 = frame_i.img_true_shape, frame_j.img_true_shape
    
    mask1, mask2 = frame_i.mask.unsqueeze(0), frame_j.mask.unsqueeze(0)
    mask_i = _resize_mask(mask1, mask1.shape) 
    mask_j = _resize_mask(mask2, mask2.shape)

    res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape1, shape2, mask_i, mask_j)
    res = [res11, res21]
    X, C, D, Q = zip(
        *[(r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0]) for r in res]
    )
    # 4xhxwxc
    X, C, D, Q = torch.stack(X), torch.stack(C), torch.stack(D), torch.stack(Q)
    X, C, D, Q = downsample(X, C, D, Q)
    return X, C, D, Q


def mast3r_match_asymmetric(model, frame_i, frame_j, idx_i2j_init=None):
    '''
        frame_i: current frame
        frame_j: keyframe
    '''    
    X, C, D, Q = mast3r_asymmetric_inference(model, frame_i, frame_j)

    b, h, w = X.shape[:-1]
    # 2 outputs per inference
    b = b // 2

    Xii, Xji = X[:b], X[b:]
    Cii, Cji = C[:b], C[b:]
    Dii, Dji = D[:b], D[b:]
    Qii, Qji = Q[:b], Q[b:]
    
    if frame_i.mask != None and frame_j.mask != None:
        Xii = Xii * (1-frame_i.mask).unsqueeze(0).unsqueeze(-1)
        Dii = Dii * (1-frame_i.mask).unsqueeze(0).unsqueeze(-1)

    idx_i2j, valid_match_j = matching.match(
        Xii, Xji, Dii, Dji, idx_1_to_2_init=idx_i2j_init
    )

    # How rest of system expects it
    Xii, Xji = einops.rearrange(X, "b h w c -> b (h w) c")
    Cii, Cji = einops.rearrange(C, "b h w -> b (h w) 1")
    Dii, Dji = einops.rearrange(D, "b h w c -> b (h w) c")
    Qii, Qji = einops.rearrange(Q, "b h w -> b (h w) 1")
    
    if frame_i.mask != None and frame_j.mask != None:
        Cii = Cii * einops.rearrange(1-frame_i.mask, "h w -> (h w) 1")
        Qii = Qii * einops.rearrange(1-frame_i.mask, "h w -> (h w) 1")
        
    return idx_i2j, valid_match_j, Xii, Cii, Qii, Xji, Cji, Qji

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)

def resize_img(img, size, square_ok=False, return_transformation=False):
    assert size == 224 or size == 512
    # numpy to PIL format
    img = PIL.Image.fromarray(np.uint8(img * 255))
    W1, H1 = img.size
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
    else:
        # resize long side to 512
        img = _resize_pil_image(img, size)
    W, H = img.size
    cx, cy = W // 2, H // 2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx - half, cy - half, cx + half, cy + half))
    else:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not (square_ok) and W == H:
            halfh = 3 * halfw / 4
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    res = dict(
        img=ImgNorm(img)[None],
        true_shape=np.int32([img.size[::-1]]),
        unnormalized_img=np.asarray(img),
    )
    if return_transformation:
        scale_w = W1 / W
        scale_h = H1 / H
        half_crop_w = (W - img.size[0]) / 2
        half_crop_h = (H - img.size[1]) / 2
        return res, (scale_w, scale_h, half_crop_w, half_crop_h)

    return res

def resize_img_mask(img, size, square_ok=False, return_transformation=False):
    assert size == 224 or size == 512
    # numpy to PIL format
    img = PIL.Image.fromarray(np.uint8(img * 255))
    W1, H1 = img.size
    if size == 224:
        # resize short side to 224 (then crop)
        img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
    else:
        # resize long side to 512
        img = _resize_pil_image(img, size)
    W, H = img.size
    cx, cy = W // 2, H // 2
    if size == 224:
        half = min(cx, cy)
        img = img.crop((cx - half, cy - half, cx + half, cy + half))
    else:
        halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        if not (square_ok) and W == H:
            halfh = 3 * halfw / 4
        img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

    res = dict(
        img = (torch.from_numpy(np.array(img)).float() / 255),
        unnormalized_img=np.asarray(img),
    )
    if return_transformation:
        scale_w = W1 / W
        scale_h = H1 / H
        half_crop_w = (W - img.size[0]) / 2
        half_crop_h = (H - img.size[1]) / 2
        return res, (scale_w, scale_h, half_crop_w, half_crop_h)

    return res

def _resize_mask(mask, shape):
    H, W = shape[1], shape[2]
    N_H = H // 16
    N_W = W // 16
    # # visualize the mask
    # import torchvision
    # torchvision.utils.save_image(mask.float(), 'dynamic_mask1.png')
    # downsample the mask to the shape of N_H x N_W
    max_pool = torch.nn.MaxPool2d(kernel_size=16, stride=16)
    mask = max_pool(mask.float().unsqueeze(1)).squeeze(1)
    #torchvision.utils.save_image(mask.float(), 'dynamic_mask_downsampled.png')
    mask = rearrange(mask, 'b nh nw -> b (nh nw) 1', nh=N_H, nw=N_W)
    return mask
