import sys
sys.path.append('')

import torch

from lib.utils.data_utils import quaternion_hamilton_product

def exp_map(q, rgrad, device='cuda'):
    alpha = torch.norm(rgrad[:, :, 1:], dim=-1)
    tmp = torch.sin(alpha) / alpha
    beta = torch.ones_like(alpha).to(device)
    mask = (alpha > 0.)
    beta[mask] = tmp[mask]

    exp_v = torch.ones_like(rgrad).to(device)
    exp_v[:, :, 0] = torch.cos(alpha)
    exp_v[:, :, 1:] = torch.einsum('...d, ...->...d', rgrad[:, :, 1:], beta)

    prod = quaternion_hamilton_product(q, exp_v)
    # prod[:,:,0] = prod[:,:,0].clamp(0.)
    prod = prod / torch.norm(prod, dim=-1).unsqueeze(-1)
    # prod = tr.sign(prod[:,:,0]).unsqueeze(-1)*prod
    mask = (prod[:, :, 0] < 0)
    prod[mask] *= -1
    return prod

def egrad2rgrad(egrad, q, device='cuda', norm=True):
    bs, n_joints, _ = q.shape

    Id = torch.eye(4).to(device)
    Id = Id.expand(bs, n_joints, 4, 4)
    P = Id - torch.einsum('...ij, ...jk -> ...ik', q.unsqueeze(-1), q.unsqueeze(2)) # (bs, nj, 4, 4)
    
    # project egrad to the tangent of q -> v
    v = torch.einsum('...ij, ...jk -> ...ik', P, egrad.unsqueeze(-1)) # (bs, nj, 4, 1)
    if norm:
        v = torch.nn.functional.normalize(v, dim=2) # unit gradient length

    # unit quaternion constraint
    rmat = torch.eye(4) # (bs, nj, 4, 4)
    rmat = rmat.expand(bs, n_joints, 4, 4).to(device)

    rmat[:, :, 0, 0] = 0.
    rmat[:, :, 1, 0] = -q[:, :, 1] / (1 + q[:, :, 0])
    rmat[:, :, 2, 0] = -q[:, :, 2] / (1 + q[:, :, 0])
    rmat[:, :, 3, 0] = -q[:, :, 3] / (1 + q[:, :, 0])
    
    rgrad = torch.einsum('...ij, ...jk -> ...ik', rmat, v)
    rgrad = rgrad.squeeze(-1) # [bs, nj, 4]

    return rgrad


def rdfgrad(egrad, q, dist, step_size, device='cuda', norm=True):

    rgrad = egrad2rgrad(egrad, q, device=device, norm=norm) # [bs, nj, 4]
    rgrad = -step_size * dist * rgrad
   
    res = exp_map(q, rgrad, device=device)

    return res
