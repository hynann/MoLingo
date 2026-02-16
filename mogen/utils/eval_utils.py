import sys
sys.path.append('.')

import os
import torch

import torch.nn.functional as F

from mogen.models.eval.ms_motionencoder import ActorAgnosticEncoder
from mogen.models.eval.distillbert_actor import DistilbertActorAgnosticEncoder

def load_ms_evaluators(device):
    modelpath = 'distilbert-base-uncased'
    motionencoder = ActorAgnosticEncoder(nfeats=272, vae=True, num_layers=4, latent_dim=256, max_len=300)
    textencoder = DistilbertActorAgnosticEncoder(modelpath, num_layers=4, latent_dim=256)

    os.chdir('mogen/checkpoints/ms')
    sys.path.append(os.getcwd())
    ckpt_path = 'epoch=99.ckpt'
    print(f'Loading evaluator checkpoint from {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    os.chdir('../../../')

    textencoder_ckpt = {}
    for k, v in ckpt['state_dict'].items():
        if k.split(".")[0] == "textencoder":
            name = k.replace("textencoder.", "")
            textencoder_ckpt[name] = v
    textencoder.load_state_dict(textencoder_ckpt, strict=True)
    textencoder.eval()
    textencoder.to(device)

    # load motionencoder
    motionencoder_ckpt = {}
    for k, v in ckpt['state_dict'].items():
        if k.split(".")[0] == "motionencoder":
            name = k.replace("motionencoder.", "")
            motionencoder_ckpt[name] = v
    motionencoder.load_state_dict(motionencoder_ckpt, strict=True)
    motionencoder.eval()
    motionencoder.to(device)

    return textencoder, motionencoder


def recover_from_local_position_batched(final_x: torch.Tensor, njoint: int) -> torch.Tensor:
    squeeze_back = False
    if final_x.ndim == 2:              # [T,D] -> [1,T,D]
        final_x = final_x.unsqueeze(0)
        squeeze_back = True

    B, T, D = final_x.shape
    device, dtype = final_x.device, final_x.dtype

    pos_no_head = final_x[..., 8:8+3*njoint].view(B, T, njoint, 3)  # [B,T,J,3]
    vel_root_xy = final_x[..., :2]                                  # [B,T,2]
    head6d_rel  = final_x[..., 2:8]                                 # [B,T,6]

    R_rel = rotation_6d_to_matrix(head6d_rel)                       # [B,T,3,3]
    R_abs = cumulative_matprod_time(R_rel, use_identity_init=False) # [B,T,3,3]
    R_inv = R_abs.transpose(-1, -2).contiguous()

    pos_with_head = torch.matmul(
        R_inv.unsqueeze(2),              # [B,T,1,3,3]
        pos_no_head.unsqueeze(-1)        # [B,T,J,3,1]
    ).squeeze(-1)                        # [B,T,J,3]

    vx, vz = vel_root_xy.unbind(dim=-1)                              # [B,T], [B,T]
    vel_root_xyz = torch.stack([vx, torch.zeros_like(vx), vz], dim=-1)  # [B,T,3]

    if T > 1:
        rotated_tail = torch.matmul(
            R_inv[:, :-1],                                 # [B,T-1,3,3]
            vel_root_xyz[:, 1:, None, :].transpose(-1, -2) # [B,T-1,3,1]
        ).squeeze(-1)                                      # [B,T-1,3]
        vel_root_xyz = torch.cat([vel_root_xyz[:, :1, :], rotated_tail], dim=1)

    root_tr = vel_root_xyz.cumsum(dim=1)                  # [B,T,3]

    trans = torch.zeros(B, T, 1, 3, device=device, dtype=dtype)
    trans[..., 0] = root_tr[..., 0].unsqueeze(-1)         # x
    trans[..., 2] = root_tr[..., 2].unsqueeze(-1)         # z
    pos_with_head = pos_with_head + trans.expand(-1, -1, njoint, -1)

    return pos_with_head.squeeze(0) if squeeze_back else pos_with_head


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def cumulative_matprod_time(R_rel: torch.Tensor, use_identity_init: bool = False) -> torch.Tensor:
    B, T = R_rel.shape[:2]
    I = torch.eye(3, device=R_rel.device, dtype=R_rel.dtype).expand(B, 3, 3)
    Rs = []
    prev = I if use_identity_init else R_rel[:, 0]
    Rs.append(prev)
    for t in range(1, T):
        prev = R_rel[:, t].matmul(prev)
        Rs.append(prev)
    return torch.stack(Rs, dim=1)  # [B,T,3,3]