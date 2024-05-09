import sys
sys.path.append('')

import os
import numpy as np
import torch
import glob

from torch.utils.data import Dataset

from lib.utils.transforms import axis_angle_to_quaternion
from lib.utils.data_utils import quaternion_hamilton_product, amass_splits

n_joints = 21


def nrdf_sampling(dist, k):
    """

    Args:
        dist: numpy [bs,] Desired distance user specify
        k: Number of articulated joints (without root)

    Returns: delta_quat: numpy [bs, k, 4] Quaternions for perturbing the clean poses

    """
    bs = len(dist)
    # sample uniformed directions in the tangent space of SO(3)^nj
    dirs = np.random.randn(bs, k, 3)
    dirs = dirs / np.linalg.norm(dirs, 2, -1, keepdims=True)  # unit vecs

    # sample weights distributed to different body joints
    weights = np.random.rand(bs, k)  # sample from uniform distribution
    weights = weights / np.sum(weights, axis=-1, keepdims=True)  # [bs, nj]

    dist = dist[:, np.newaxis] * weights * float(
        n_joints)  # [bs, nj]: dis values distributed to multiple body joints

    delta_quat = np.concatenate(
        (np.cos(0.5 * dist)[:, :, np.newaxis], np.sin(0.5 * dist)[:, :, np.newaxis] * dirs),
        axis=2)  # [bs, nj, 4]

    return delta_quat


class PerturbData(Dataset):
    def __init__(self, data_path, 
                       mode='gaussian',
                       batch_size=1, 
                       num_workers=12, 
                       num_samples=1000, 
                       runs=10,
                       var=0.785):

        self.data_path = data_path
        self.mode = mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_samples, self.runs = num_samples, runs

        self.var = var
    
    def __len__(self):
        return self.runs
    
    def __getitem__(self, idx):
        bdata = np.load(self.data_path)
        pose_aa = bdata['pose_body'][:, :n_joints*3]

        pose_quat = axis_angle_to_quaternion(torch.from_numpy(pose_aa.reshape(len(pose_aa), -1, 3))).detach().numpy()

        indices = np.random.randint(0, len(pose_quat), self.num_samples)
        pose_quat = pose_quat[indices] # [bs, nj, 4]
        org_pose_aa = pose_aa[indices] # [bs, nj*3]

        # generate dist values according to the desired distribution users define
        geo_dis = None
        if self.mode == 'gaussian':
            geo_dis = np.random.normal(scale=self.var, size=len(pose_quat)) # [bs,]
        elif self.mode == 'exp':
            geo_dis = np.random.exponential(scale=1./2.5, size=len(pose_quat))
        elif self.mode == 'uni':
            geo_dis = (np.random.rand(len(pose_quat)) - 0.5) * 2 * np.pi * 0.5

        delta_quat = nrdf_sampling(geo_dis, n_joints)
        perturbed_quat = quaternion_hamilton_product(torch.from_numpy(pose_quat), torch.from_numpy(delta_quat)).detach().cpu().numpy() # [bs, nj, 4]

        return {'perturbed_quat': perturbed_quat, 'org_pose_aa': org_pose_aa}
    

    def get_loader(self, shuffle=False):
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle, drop_last=True)


class FaissData(Dataset):

    def __init__(self,  data_path, batch_size=256, num_workers=3):

        self.path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        bdata = np.load(self.path)
        self.poses = bdata['pose_body']

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        poses = self.poses[idx]
        return {'pose': poses}

    def get_loader(self, shuffle=False, drop_last=False):
        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn, drop_last=drop_last)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)


class PoseData(Dataset):
    def __init__(self, mode, noisy_dir, clean_dir, batch_size=4, num_workers=6, num_pts=5000, stage=1, flip=False,
                 random_poses=False):
        self.mode = mode
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.num_pts = num_pts

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.flip = flip
        self.stage = stage

        self.data_splits = amass_splits[self.mode]

        self.noisy_seqs = sorted(glob.glob(self.noisy_dir + '/*/*.npz'))
        self.noisy_seqs = [ds for ds in self.noisy_seqs if ds.split('/')[-2] in self.data_splits]

        self.clean_seqs = sorted(glob.glob(self.clean_dir + '/*/*.npz'))
        self.clean_seqs = [ds for ds in self.clean_seqs if ds.split('/')[-2] in self.data_splits]
    
    def __len__(self):
        return len(self.clean_seqs)

    def __getitem__(self, idx):
        bdata = np.load(self.noisy_seqs[idx])
        dist = bdata['dist'][:, 0].astype(np.float32)

        subsample_indices = np.random.randint(0, len(bdata['noisy_quats']), self.num_pts)

        # noisy data
        poses = bdata['noisy_quats'][subsample_indices].astype(np.float32)
        
        dist = dist[subsample_indices]

        # clean data
        cdata = np.load(self.clean_seqs[idx])
        subsample_indices_clean = np.random.randint(0, len(cdata['pose_body']), self.num_pts)
        clean_poses = cdata['pose_body'][:, :63][subsample_indices_clean].astype(np.float32)
        clean_poses = axis_angle_to_quaternion(
            torch.Tensor(clean_poses.reshape(-1, 21, 3))).detach().cpu().numpy()
        
        return {'pose': poses,
                'dist': dist,
                'man_poses': clean_poses,
                }
    
    def get_loader(self, shuffle=True):
        return torch.utils.data.DataLoader(
            self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
            worker_init_fn=self.worker_init_fn, drop_last=True)
    
    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)
    

class TestData(Dataset):

    def __init__(self,  data_path, batch_size=100, num_workers=3, device='cuda:0'):

        self.path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        bdata = np.load(self.path)
        self.noisy_quats = bdata['noisy_quats']
        self.noisy_pose_aa = bdata['noisy_pose_aa']

    def __len__(self):
        return len(self.noisy_quats)

    def __getitem__(self, idx):

        noisy_quats = self.noisy_quats[idx]
        noisy_pose_aa = self.noisy_pose_aa[idx]
        return {'noisy_quats': noisy_quats, 'noisy_pose_aa': noisy_pose_aa}

    def get_loader(self, shuffle =False, drop_last=False):
        return torch.utils.data.DataLoader(
                self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle,
                worker_init_fn=self.worker_init_fn, drop_last=drop_last)

    def worker_init_fn(self, worker_id):
        random_data = os.urandom(4)
        base_seed = int.from_bytes(random_data, byteorder="big")
        np.random.seed(base_seed + worker_id)

        