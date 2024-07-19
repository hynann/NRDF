import sys
sys.path.append('')

import torch
import numpy as np
import argparse
import os.path as osp
import smplx

from lib.data.dataloaders import nrdf_sampling
from lib.utils.transforms import quaternion_to_axis_angle
from lib.exp.pose_den import Projector
from lib.utils.exp_utils import calculate_frechet_distance

from scipy.spatial import distance

n_joints = 21


def calculate_apd(pts):
    # pts: (n, j, 3)
    bs, nj, _ = pts.shape
    res = distance.pdist(pts.reshape(-1, nj*3), lambda u, v: np.sqrt(np.sum(np.power((u.reshape(nj, 3)-v.reshape(nj, 3)), 2), axis=1)).mean())  # Pairwise distances
    res = res.mean()
    return res


def eval_poses(pose):
    device = args.device
    bs, _ = pose.shape
    body_model = smplx.create(args.bm_path, model_type='smplh', num_betas=10, gender='neutral',
                              batch_size=bs).to(device=device)

    cdata = np.load(osp.join('examples', 'gt_mean_var.npz'))
    mu, cov, muj, covj = cdata['mu'], cdata['cov'], cdata['muj'], cdata['covj']

    body_params = {}
    body_params['body_pose'] = torch.tensor(pose).type(torch.float).to(
        device=device)
    body_params['betas'] = torch.zeros((bs, 10)).to(device=device)
    body_out = body_model(return_verts=True, **body_params)
    pts = body_out.joints[:, :22].detach().cpu().numpy()  # numpy (bs, 22, 3)
    pts_r = pts.reshape(-1, 66)

    apd = calculate_apd(pts)
    mu_wj, cov_wj = np.mean(pts_r, axis=0), np.cov(pts_r, rowvar=False)

    fid = calculate_frechet_distance(muj, covj, mu_wj, cov_wj)

    print(f'APD: {apd*100} CM')
    print(f'FID: {fid}')


def sample_pose():
    device = args.device

    # initialize poses
    dist = np.random.normal(scale=1., size=args.num_samples)  # [bs,]
    noisy_quat = nrdf_sampling(dist=dist, k=n_joints) # numpy [bs, 21, 4]
    noisy_pose = quaternion_to_axis_angle(torch.from_numpy(noisy_quat)).reshape(args.num_samples, -1)
    projector = Projector(model_dir=args.model_dir,
                          noisy_pose=noisy_pose,
                          device=device)

    res_aa, _ = projector.project(step_size=args.step_size, iterations=args.iterations, save_all_step=False)
    return res_aa[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configurations for generating diverse poses by using NRDF")

    parser.add_argument("-ns", "--num_samples", type=int, default=500, help="Number of generated samples")
    parser.add_argument("-md", "--model_dir", type=str,
                        default="checkpoints/flip_exp_softplus_l1_0.0001_10000_dist0.5_eik0.0_man0.1",
                        help="Path to the pretrained NRDF model")
    parser.add_argument("-bp", "--bm_path", type=str, default="/home/ubuntu/data/smplx_models",
                        help="SMPL body model path")
    parser.add_argument("-s", "--step_size", type=float, default=0.01, help="Step size (alpha) during projection")
    parser.add_argument("-it", "--iterations", type=int, default=200, help="Number of iterations during projection")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="CUDA or CPU")

    args = parser.parse_args()

    sampled_poses = sample_pose()
    eval_poses(sampled_poses)