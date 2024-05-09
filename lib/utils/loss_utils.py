import torch
import smplx

from torch.autograd import grad

from lib.utils.data_utils import vertice_idx, quat_to_global
from lib.utils.transforms import axis_angle_to_quaternion


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad


def cal_dist_geo(nonman_pose, man_pose):
    dis = torch.mean(1 - torch.abs(torch.sum(man_pose*nonman_pose, dim=2)), dim=1).to('cuda')
    return dis


def v2v_err(pose_src, pose_dst, bm_path, device='cuda'):
    body_model = smplx.create(bm_path, model_type='smplh', num_betas=10, gender='neutral',
                                  batch_size=len(pose_src)).to(device=device)
    params_src, params_dst = {}, {}
    params_src['body_pose'], params_dst['body_pose'] = torch.from_numpy(pose_src).type(torch.float32).to(
        device=device), torch.from_numpy(pose_dst).type(torch.float32).to(device=device)
    params_src['betas'], params_dst['betas'] = torch.zeros((len(pose_src), 10)).to(device=device), torch.zeros(
        (len(pose_dst), 10)).to(device=device)
    body_out_src, body_out_dst = body_model(return_verts=True, **params_src), body_model(return_verts=True,
                                                                                        **params_dst)
    verts_src, verts_dst = body_out_src.vertices, body_out_dst.vertices

    markers_src, markers_dst = verts_src[:, vertice_idx, :], verts_dst[:, vertice_idx, :]  # (N, nm, 3)
    
    m2m_dist = markers_src - markers_dst
    m2m_dist = torch.mean(torch.sqrt(torch.sum(m2m_dist * m2m_dist, dim=-1)), dim=-1).detach().cpu().numpy()  # (N, )
    
    return m2m_dist


def quat_geo_global(pose_src, pose_dst, device='cuda'):
    n_joints = pose_src.shape[-1] // 3
    quat_src = axis_angle_to_quaternion(torch.from_numpy(pose_src.reshape(-1, n_joints, 3)).to(device))
    quat_dst = axis_angle_to_quaternion(torch.from_numpy(pose_dst.reshape(-1, n_joints, 3)).to(device))

    quat_src_glob, quat_dst_glob = quat_to_global(quat_src), quat_to_global(quat_dst)
    geo_dist_glob = torch.mean(cal_dist_geo(quat_src_glob, quat_dst_glob), dim=-1).detach().cpu().numpy()
    
    return geo_dist_glob


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count