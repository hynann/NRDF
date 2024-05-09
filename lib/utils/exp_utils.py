'''some part of code is borrowed from VPoser:
https://github.com/nghorbani/human_body_prior/blob/master/tutorials/ik_example_mocap.py'''

import sys
sys.path.append('')

import numpy as np
import smplx
import torch
import math

from torch import nn

from lib.utils.transforms import quaternion_to_axis_angle
from lib.utils.marker_vids import marker_type_labels, all_marker_vids

NUM_SMPL_VERTICES = 6890

body_parts = {
    'end_eff': ['left_wrist', 'right_wrist', 'left_foot', 'right_foot'],
    'legs_occ': ['left_wrist', 'right_wrist', 'head', 'torso', 'left_arm', 'right_arm'],
    'right_arm_occ': ['left_wrist', 'head', 'torso', 'left_arm', 'legs', 'left_foot', 'right_foot'],
    'left_arm_occ': ['right_wrist', 'head', 'torso', 'right_arm', 'legs', 'left_foot', 'right_foot'],
    'arm_leg_occ': ['left_wrist', 'head', 'torso', 'left_arm', 'right_leg', 'right_foot']
}


partial_joint_index = {
    'end_eff': [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    'legs_occ': [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'left_arm_occ': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1],
    'right_arm_occ': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0],
    'all': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
}


def get_marker_vids(obs_type, occ_type):
    if obs_type == 'marker':
        res = []
        for part in body_parts[occ_type]:
            all_parts = marker_type_labels[part]
            for joint_n in all_parts:
                res.append(all_marker_vids['smplh'][joint_n])
        mask = np.zeros((NUM_SMPL_VERTICES,)).astype(np.int64)
        mask[res] = 1
    else:
        mask = partial_joint_index[occ_type]
        mask = np.array(mask)
    return mask


def makepath(*args, **kwargs):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    isfile = kwargs.get('isfile', False)
    import os
    desired_path = os.path.join(*args)
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)): os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path


def c2c(tensor):
    if isinstance(tensor, np.ndarray): return tensor
    return tensor.detach().cpu().numpy()


def visualize(verbosity=2, logger=None):
    if logger is None: logger = log2file()

    def view(opt_objs, opt_it):
        if verbosity <= 0: return
        opt_objs_cpu = {k: c2c(v) for k, v in opt_objs.items()}

        total_loss = np.sum([np.sum(v) for k, v in opt_objs_cpu.items()])
        message = 'it {} -- [total loss = {:.2e}] - {}'.format(opt_it, total_loss, ' | '.join(
            ['%s = %2.2e' % (k, np.sum(v)) for k, v in opt_objs_cpu.items()]))
        logger(message)

    return view


class log2file():
    def __init__(self, logpath=None, prefix='', auto_newline=True, write2file_only=False):
        if logpath is not None:
            makepath(logpath, isfile=True)
            self.fhandle = open(logpath, 'a+')
        else:
            self.fhandle = None

        self.prefix = prefix
        self.auto_newline = auto_newline
        self.write2file_only = write2file_only

    def __call__(self, text):
        if text is None: return
        if self.prefix != '': text = '{} -- '.format(self.prefix) + text
        # breakpoint()
        if self.auto_newline:
            if not text.endswith('\n'):
                text = text + '\n'
        if not self.write2file_only: sys.stderr.write(text)
        if self.fhandle is not None:
            self.fhandle.write(text)
            self.fhandle.flush()


class SourceKeyPoints(nn.Module):
    def __init__(self,
                 bm_path,
                 visible_idx,
                 obs_type,
                 n_joints: int=22,
                 batch_size=4,
                 device='cuda',
                 ):
        super(SourceKeyPoints, self).__init__()

        self.bm = smplx.create(bm_path, model_type='smplh', num_betas=10, gender='neutral',
                              batch_size=batch_size).to(device=device)
        self.n_joints = n_joints
        self.obs_type = obs_type
        self.visible_idx = visible_idx

    def forward(self, body_params):

        smpl_params = {k: v for k, v in body_params.items() if k != 'body_pose'}
        smpl_params['body_pose'] = quaternion_to_axis_angle(body_params['body_pose']).reshape(-1, 63)
        body_new = self.bm(**smpl_params)

        if self.obs_type == 'marker':
            source_kpts = body_new.vertices[:, self.visible_idx]
        else:
            source_kpts = body_new.joints[:, :self.n_joints][:, self.visible_idx]

        return {'source_kpts': source_kpts, 'body': body_new}


def euc_err(pt1, pt2):
    diff = pt1 - pt2
    err = torch.sqrt(torch.sum(diff**2, dim=-1)).mean().detach().cpu().numpy()
    return err


def projection(cam_transform, cam_intrinsics, points_3d, batch_size, device='cuda'):
    """project similar to smplify-x"""
    center = torch.zeros([batch_size, 2], device=points_3d.device)
    center[:, 0] = cam_intrinsics[0, 2]
    center[:, 1] = cam_intrinsics[1, 2]
    camera_transform = cam_transform
    camera_mat = cam_intrinsics[:2, :2].unsqueeze(0).repeat(batch_size, 1, 1)
    homog_coord = torch.ones(list(points_3d.shape)[:-1] + [1], dtype=points_3d.dtype, device=device)
    # Convert the points to homogeneous coordinates
    points_h = torch.cat([points_3d, homog_coord], dim=-1)

    projected_points = torch.einsum('bki,bji->bjk',  [camera_transform, points_h])

    img_points = torch.div(projected_points[:, :, :2], projected_points[:, :, 2].unsqueeze(dim=-1))
    img_points = torch.einsum('bki,bji->bjk', [camera_mat, img_points]) + center.unsqueeze(dim=1)

    return img_points


def joints_coco(smpl_verts, smpl_joints):
    VERT_NOSE = 331
    VERT_EAR_L = 3485
    VERT_EAR_R = 6880
    VERT_EYE_L = 2802
    VERT_EYE_R = 6262

    nose = smpl_verts[:, None, VERT_NOSE]
    ear_l = smpl_verts[:, None, VERT_EAR_L]
    ear_r = smpl_verts[:, None, VERT_EAR_R]
    eye_l = smpl_verts[:, None, VERT_EYE_L]
    eye_r = smpl_verts[:, None, VERT_EYE_R]

    shoulders_m = torch.sum(torch.hstack((smpl_joints[:, None, 14],  smpl_joints[:, None, 13])), axis=1) / 2.
    shoulders_m = shoulders_m[:, None, :]
    neck = smpl_joints[:, None,12] - 0.55 * (smpl_joints[:,None, 12] - shoulders_m)

    coco_jts = torch.hstack((nose,neck,
        2.1 * (smpl_joints[:, None, 14] - shoulders_m) + neck,
        smpl_joints[:, None, 19],
        smpl_joints[:, None, 21],
        2.1 * (smpl_joints[:, None, 13] - shoulders_m) + neck,
        smpl_joints[:, None, 18],
        smpl_joints[:, None, 20],
        smpl_joints[:, None, 2] + 0.38 * (smpl_joints[:, None, 2] - smpl_joints[:, None, 1]),
        smpl_joints[:, None, 5],
        smpl_joints[:, None, 8],
        smpl_joints[:, None, 1] + 0.38 * (smpl_joints[:, None,1] - smpl_joints[:, None, 2]),
        smpl_joints[:, None, 4],
        smpl_joints[:, None, 7],
        eye_r,
        eye_l,
        ear_r,
        ear_l))
    return coco_jts


class GMoF(nn.Module):
    def __init__(self, rho=100.0):
        super(GMoF, self).__init__()
        self.rho = rho

    def extra_repr(self):
        return 'rho = {}'.format(self.rho)

    def forward(self, residual):
        squared_res = residual ** 2
        dist = torch.div(squared_res, squared_res + self.rho ** 2)
        return self.rho ** 2 * dist
