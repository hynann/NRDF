'''some part of code is borrorwed from VPoser:
https://github.com/nghorbani/human_body_prior/blob/master/tutorials/ik_example_mocap.py'''

import sys
sys.path.append('')

import argparse
import os
import os.path as osp
import numpy as np
import smplx
import torch
import datetime
import random

from torch import nn

from configs.config import load_config

from lib.model.nrdf import NRDF
from lib.core.optimizers import QuaternionSGD, HybridSGD
from lib.data.dataloaders import nrdf_sampling
from lib.utils.transforms import quaternion_to_axis_angle
from lib.utils.exp_utils import SourceKeyPoints, log2file, get_marker_vids, visualize

n_joints = 22
prior_thred = 5e-4
batch_size = 4


def prepare_data(args):
    pose = np.load(args.input_path)['pose']
    rnd_frame_ids = random.sample(range(len(pose)), batch_size)
    pose = pose[rnd_frame_ids, :63]
    device = args.device
    mask = get_marker_vids(args.obs_type, args.occ_type)
    visible_idx = np.where(mask == 1)[0]

    body_model = smplx.create(args.bm_path, model_type='smplh', num_betas=10, gender='neutral',
                              batch_size=len(pose)).to(device)
    body_params = {}
    body_params['body_pose'] = torch.tensor(pose).type(torch.float32).to(device)
    body_params['betas'] = torch.zeros((len(pose), 10)).to(device)
    body_out = body_model(return_verts=True, **body_params)

    source_pts = SourceKeyPoints(bm_path=args.bm_path,
                                 visible_idx=visible_idx,
                                 obs_type=args.obs_type,
                                 n_joints=n_joints,
                                 batch_size=len(pose),
                                 device=device)

    target_pts = body_out.vertices[:, visible_idx].detach().to(device) if args.obs_type == 'marker' else body_out.joints[:, :n_joints][:,visible_idx].detach().to(device)

    return source_pts, target_pts


def prior_fit(optimizer, nrdf_model, on_step=None, gstep=0):
    def fit(weights, free_vars):
        fit.gstep += 1
        optimizer.zero_grad()

        opt_objs = {}

        dist_pred = nrdf_model(free_vars['body_pose'], train=False)['dist_pred']
        opt_objs['dis'] = torch.mean(dist_pred)

        opt_objs = {k: opt_objs[k] * v for k, v in weights.items() if k in opt_objs.keys()}
        loss_total = torch.sum(torch.stack(list(opt_objs.values())))
        loss_total.backward()

        if on_step is not None:
            on_step(opt_objs, fit.gstep)

        fit.free_vars = {k: v for k, v in free_vars.items()}
        fit.final_loss = loss_total

        return loss_total

    fit.gstep = gstep
    fit.final_loss = None
    fit.free_vars = {}
    return fit


def ik_fit(optimizer, source_kpts_model, static_vars, nrdf_model, extra_params={}, on_step=None, gstep=0):
    data_loss = extra_params.get('data_loss', torch.nn.SmoothL1Loss(reduction='mean'))

    def fit(weights, free_vars):
        fit.gstep += 1
        optimizer.zero_grad()

        opt_objs = {}
        res = source_kpts_model(free_vars)

        opt_objs['data'] = data_loss(res['source_kpts'], static_vars['target_kpts'])
        data_err = opt_objs['data']

        opt_objs['betas'] = torch.pow(free_vars['betas'], 2).sum()

        dist_pred = nrdf_model(free_vars['body_pose'], train=False)['dist_pred']
        opt_objs['dis'] = torch.mean(dist_pred)
        prior_err = opt_objs['dis']

        opt_objs = {k: opt_objs[k] * v for k, v in weights.items() if k in opt_objs.keys()}
        loss_total = torch.sum(torch.stack(list(opt_objs.values())))
        loss_total.backward()

        if on_step is not None:
            on_step(opt_objs, fit.gstep)

        fit.free_vars = {k: v for k, v in free_vars.items()}  # if k in IK_Engine.fields_to_optimize}
        fit.final_loss = loss_total

        return loss_total, data_err, prior_err

    fit.gstep = gstep
    fit.final_loss = None
    fit.free_vars = {}

    return fit


class IK_Engine(nn.Module):
    def __init__(self,
                 model_dir: str,
                 data_loss,
                 device: str,
                 optimizer_args: dict = {'type': 'ADAM'},
                 stepwise_weights: list[dict] = [{'data': 10., 'poZ_body': .01, 'betas': .5}],
                 verbosity: int = 1,
                 num_betas: int = 10,
                 thred: float = 0.03,
                 logger=None):
        '''
        :param model_dir: The nrdf directory that holds the model checkpoint
        :param data_loss: should be a pytorch callable (source, target) that returns the accumulated loss
        :param optimizer_args: arguments for optimizers
        :param stepwise_weights: list of dictionaries. each list element defines weights for one full step of optimization
                                 if a weight value is left out, its respective object item will be removed as well. imagine optimizing without data term!
        :param verbosity: 0: silent, 1: text, 2: text/visual. running 2 over ssh would need extra work
        :param logger: an instance of human_body_prior.tools.omni_tools.log2file
        '''
        super(IK_Engine, self).__init__()

        self.data_loss = torch.nn.SmoothL1Loss(reduction='mean') if data_loss is None else data_loss
        self.num_betas = num_betas
        self.stepwise_weights = stepwise_weights
        self.verbosity = verbosity
        self.optimizer_args = optimizer_args
        self.data_thred = thred
        self.device = device

        self.logger = log2file() if logger is None else logger
        self.model = self._load_model(model_dir)

    def _load_model(self, model_dir):
        checkpoint_path = osp.join(model_dir, 'checkpoints', 'checkpoint_epoch_best.tar')
        config_file = osp.join(model_dir, 'config.yaml')

        self.model = NRDF(load_config(config_file))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')['model_state_dict']
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model = self.model.to(self.device)
        return self.model

    def forward(self, source_kpts, target_kpts, initial_body_params={}):
        bs = target_kpts.shape[0]
        on_step = visualize(verbosity=self.verbosity, logger=self.logger)
        comp_device = target_kpts.device

        if 'body_pose' not in initial_body_params:
            dist = np.random.normal(scale=0.1, size=bs) # feel free to change the variance, higher var -> higher diversity in results
            noisy_quat = nrdf_sampling(dist=dist, k=n_joints-1)
            noisy_quat = torch.from_numpy(noisy_quat).to(device=comp_device).to(torch.float)
            noisy_quat.requires_grad = False
            initial_body_params['body_pose'] = noisy_quat

        if 'trans' not in initial_body_params:
            initial_body_params['transl'] = torch.zeros([bs, 3], device=comp_device, dtype=torch.float, requires_grad=False)

        if 'root_orient' not in initial_body_params:
            initial_body_params['global_orient'] = torch.zeros([bs, 3], device=comp_device, dtype=torch.float, requires_grad=False)

        initial_body_params['betas'] = torch.zeros([bs, self.num_betas], device=comp_device, dtype=torch.float, requires_grad=False)

        free_vars = {k: torch.nn.Parameter(v.detach(), requires_grad=True) for k, v in initial_body_params.items()
                     if k in ['betas', 'transl', 'body_pose', 'global_orient']}

        static_vars = {
            'target_kpts': target_kpts,
        }

        prior_optimizer = QuaternionSGD(list(free_vars.values()),
                                        lr=self.optimizer_args.get('lr', 1))

        hybrid_optimizer = HybridSGD(list(free_vars.values()),
                                     lr=self.optimizer_args.get('lr', 1))

        gstep = 0

        prior_closure = prior_fit(optimizer=prior_optimizer, nrdf_model=self.model, on_step=on_step, gstep=gstep)
        hybrid_closure = ik_fit(optimizer=hybrid_optimizer,
                                source_kpts_model=source_kpts,
                                static_vars=static_vars,
                                nrdf_model=self.model,
                                extra_params={'data_loss': self.data_loss},
                                on_step=on_step,
                                gstep=gstep)

        for wts in self.stepwise_weights:
            # stage 1: optimize prior term only
            prior_optimizer.step(lambda: prior_closure(wts, free_vars))
            free_vars = prior_closure.free_vars

            # stage 2: optimize all terms
            prev_data_err = float('inf')
            for k in range(self.optimizer_args['max_iter']):
                _, data_err, prior_err = hybrid_optimizer.step(lambda: hybrid_closure(wts, free_vars))
                if data_err <= prev_data_err:
                    prev_data_err = data_err
                if data_err <= self.data_thred and prior_err < prior_thred:
                    break
            free_vars = hybrid_closure.free_vars

        return free_vars


def partial_ik(args):
    # create output dir
    now = datetime.datetime.now()
    op_dir = osp.join('outputs', now.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(op_dir, exist_ok=True)
    print(f'Created output dir: {op_dir}')

    device = args.device
    source_pts, target_pts = prepare_data(args)

    np.savez(osp.join(op_dir, 'target_pts.npz'), target_pts=target_pts.detach().cpu().numpy())

    data_loss = torch.nn.MSELoss(reduction='sum')
    optimizer_args = {'type': 'ADAM', 'max_iter': 10000, 'lr': 0.01, 'tolerance_change': 1e-15, 'history_size': 200}
    stepwise_weights = [
        {'data': 0.2, 'betas': .5, 'dis': 1.},
    ]

    thred = 0.03

    ik_engine = IK_Engine(model_dir=args.model_dir,
                          data_loss=data_loss,
                          optimizer_args=optimizer_args,
                          stepwise_weights=stepwise_weights,
                          verbosity=2,
                          num_betas=10,
                          thred=thred,
                          device=device)

    ik_res = ik_engine(source_pts, target_pts)
    print('Done.')

    ik_res_detached = {k: v.detach() for k, v in ik_res.items()}
    nan_mask = torch.isnan(ik_res_detached['transl']).sum(-1) != 0
    if nan_mask.sum() != 0:
        raise ValueError('Sum results were NaN!')

    res_pose = quaternion_to_axis_angle(
        ik_res['body_pose'].type(torch.float).to(device=device)).detach().cpu().numpy().reshape(-1, 63)
    res_ori = ik_res['global_orient'].type(torch.float).detach().cpu().numpy()
    res_transl = ik_res['transl'].type(torch.float).detach().cpu().numpy()
    res_betas = ik_res['betas'].detach().cpu().numpy()

    np.savez(osp.join(op_dir, 'res.npz'), pose=res_pose, root_ori=res_ori, transl=res_transl, betas=res_betas)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configurations for the pose denoising task by using NRDF")

    parser.add_argument("-ip", "--input_path", type=str, default="examples/ik_pose.npz",
                        help="Path to the input poses")
    parser.add_argument("-md", "--model_dir", type=str,
                        default="checkpoints/amass_softplus_l1_0.0001_10000_dist0.5_eik0.0_man0.1",
                        help="Path to the pretrained NRDF model")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="CUDA or CPU")
    parser.add_argument("-bp", "--bm_path", type=str, default="/home/ubuntu/data/smplx_models",
                        help="SMPL body model path")
    parser.add_argument("-occ", "--occ_type", type=str, default='left_arm_occ', help="occlusion type: all/left_arm_occ/right_arm_occ/end_eff/legs_occ")
    parser.add_argument("-obs", "--obs_type", type=str, default='marker', help="observation type: marker/joint")

    args = parser.parse_args()

    partial_ik(args)


