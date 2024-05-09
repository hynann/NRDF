import os.path as osp

import faiss
import torch
import numpy as np

amass_splits = {
    'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BioMotionLab_NTroje' ,'BMLhandball',
              'BMLmovi', 'EKUT', 'TCD_handMocap', 'ACCAD'],
    'test': ['Transitions_mocap', 'SSM_synced'],
    'valid': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
}

vertice_idx = [1666, 4503, 4288, 3346, 3298, 6217, 2346, 4154, 2660, 1170, 5494, 6366, 1634,
        544, 6836, 4409, 5350, 1510, 1321, 5135, 1017, 720, 6747, 6698, 2756, 5764,
        666, 6121, 4655, 2066, 2906, 5101, 4024, 3435, 923, 1888, 4981, 4801, 1306,
        446, 1515, 3052, 3158, 3077, 3148, 3163, 409, 3532] # selected vertex indices on SMPL layout

kin_tree = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21] # kinematic structure for SMPL+H 21-joint layout


class geo():
    def __init__(self, weighted=False):
        self.weighted = weighted
        joint_rank = torch.from_numpy(np.array([7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 4, 4, 3, 3, 3, 2, 2, 1, 1])).float()
        self.joint_weights = torch.nn.functional.normalize(joint_rank, dim=0)
        self.l2_loss = torch.nn.MSELoss()

    def dist_calc(self, noisy_quats, valid_quats, k_faiss, k_dist):
        bs = len(noisy_quats)
        noisy_quats = noisy_quats.view(bs, -1, 4)
        valid_quats = valid_quats.view(bs, k_faiss, -1, 4)
        noisy_quats = noisy_quats.unsqueeze(1).repeat(1, k_faiss, 1, 1)

        geo_dis = torch.mean(cal_intrinsic_geo(valid_quats, noisy_quats), dim=2)  # (k_faiss, )
        geo_val, geo_idx = torch.topk(geo_dis, k=k_dist, largest=False)

        return geo_val, geo_idx


def cal_intrinsic_geo(a, b):
    ndim = len(a.shape)
    a_conj = torch.cat((a[..., 0].unsqueeze(-1), -a[..., 1].unsqueeze(-1), -a[..., 2].unsqueeze(-1), -a[..., 3].unsqueeze(-1)), dim=ndim-1)
    w = quaternion_hamilton_product(a_conj, b)[..., 0]
    prod = torch.abs(w)
    prod = torch.clamp(prod, max=1)
    res = torch.acos(prod)

    return res


def quaternion_hamilton_product(a, b):

    ndim = len(a.shape)

    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    res = torch.concat((
        (aw * bw - ax * bx - ay * by - az * bz).unsqueeze(-1),
        (aw * bx + ax * bw + ay * bz - az * by).unsqueeze(-1),
        (aw * by - ax * bz + ay * bw + az * bx).unsqueeze(-1),
        (aw * bz + ax * by - ay * bx + az * bw).unsqueeze(-1)
    ), dim=ndim-1)

    return res


def quat_to_global(quat):
    '''
    :param quat: torch (bs, 21, 4)
    :return: res: torch (bs, 21, 4)
    '''
    bs = len(quat)
    unit_rot = torch.zeros(bs, 4).to('cuda')
    unit_rot[:, 0] = 1.
    transform_chain = [unit_rot]

    for i in range(quat.shape[1]):
        idx = i + 1
        pidx = kin_tree[idx]

        child_quat = quat[:, i] # (bs, 1, 4)
        cur_res = quaternion_hamilton_product(transform_chain[pidx], child_quat)
        transform_chain.append(cur_res)

    res = torch.stack(transform_chain, dim=1)[:, 1:, :]
    return res


def load_faiss(faiss_model_dir):
    faiss_model_path, all_data_path = osp.join(faiss_model_dir, 'faiss.index'), osp.join(faiss_model_dir, 'all_data.npz')
    index = faiss.read_index(faiss_model_path)
    all_data = np.load(all_data_path)
    all_poses_aa, all_poses_quat = all_data['all_poses_aa'], all_data['all_poses_quat']
    return index, all_poses_aa, all_poses_quat