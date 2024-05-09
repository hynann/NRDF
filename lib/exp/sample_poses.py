import sys
sys.path.append('')

import torch
import numpy as np
import argparse

from lib.data.dataloaders import nrdf_sampling
from lib.utils.transforms import quaternion_to_axis_angle
from lib.exp.pose_den import Projector

n_joints = 21


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configurations for generating diverse poses by using NRDF")

    parser.add_argument("-ns", "--num_samples", type=int, default=500, help="Number of generated samples")
    parser.add_argument("-md", "--model_dir", type=str,
                        default="checkpoints/flip_exp_softplus_l1_0.0001_10000_dist0.5_eik0.0_man0.1",
                        help="Path to the pretrained NRDF model")
    parser.add_argument("-s", "--step_size", type=float, default=0.01, help="Step size (alpha) during projection")
    parser.add_argument("-it", "--iterations", type=int, default=200, help="Number of iterations during projection")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="CUDA or CPU")

    args = parser.parse_args()

    sample_pose()

