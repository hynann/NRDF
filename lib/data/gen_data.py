import sys
sys.path.append('')

import os
import os.path as osp
import numpy as np
import argparse
import torch

from configs.config import load_config

from lib.utils.data_utils import amass_splits, geo, load_faiss
from lib.utils.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle
from lib.data.dataloaders import PerturbData


def nn_search(quat, index, dist_cal, all_poses_aa, all_poses_quat, k_faiss=1000, k_dist=1, org_poses=None):
    """

    Args:
        quat: numpy [bs, nj, 4], input noisy quaternion for querying nearest neighbor
        index: Pretrained faiss model
        dist_cal: Distance calculation function
        all_poses_aa: numpy [N, nj*3], all clean poses from AMASS training set, axis-angle format
        all_poses_quat: numpy [N, nj*4], all clean poses from AMASS training set, quaternion format
        k_faiss: Number of candidates selected by the kd tree
        k_dist: Number of final nearest neighbor
        org_poses: Corresponding clean pose

    Returns: k_quats, k_poses_aa, dist

    """
    
    # stage I: search K candidates
    key_faiss = quat.reshape(len(quat), -1).astype(np.float32)
    _, neighbors = index.search(key_faiss, k_faiss)

    nn_quats = all_poses_quat[neighbors].reshape(len(quat), k_faiss, -1, 4)
    nn_poses_aa = all_poses_aa[neighbors].reshape(len(quat), k_faiss, -1)

    if org_poses is not None:
        org_quat = axis_angle_to_quaternion(torch.from_numpy(org_poses.reshape(len(org_poses), -1, 3))).detach().numpy() # [N, 21, 4]
        
        nn_quats = np.concatenate((nn_quats, org_quat[:, np.newaxis, :]), axis=1)
        nn_poses_aa = np.concatenate((nn_poses_aa, org_poses[:, np.newaxis, :]), axis=1)
        
        k_faiss += 1

    # stage II: NN search within K candidates
    dist, nn_id = dist_cal.dist_calc(
        noisy_quats=torch.from_numpy(quat),
        valid_quats=torch.from_numpy(nn_quats), 
        k_faiss=k_faiss, 
        k_dist=k_dist)

    nn_id = nn_id.detach().cpu().numpy()

    k_quats = []
    k_poses_aa = []

    for idx in range(len(quat)):
        k_quats.append(nn_quats[idx][nn_id[idx]])
        k_poses_aa.append(nn_poses_aa[idx][nn_id[idx]])
    
    k_quats, k_poses_aa = np.array(k_quats), np.array(k_poses_aa)
    dist = dist.detach().cpu().numpy()

    return k_quats, k_poses_aa, dist


class DataGenerator(object):
    def __init__(self, opt, parallel=False, seq_file=None):
        self.parallel = parallel
        self.seq_file = seq_file
        self.mode = opt['mode']
        self.num_samples = opt['num_samples']
        self.runs = opt['runs']
        self.var = opt['variance']
        self.k_faiss = opt['k_faiss']
        self.k_dist = opt['k_dist']

        faiss_model_dir = osp.join(opt['data_root'], 'FAISS_MODEL')
        self.index, self.all_poses_aa, self.all_poses_quat = load_faiss(faiss_model_dir)
        print(f'Loaded {len(self.all_poses_aa)} clean poses...')

        self.data_splits = sorted(amass_splits[opt['split']])

        self.raw_data_dir = osp.join(opt['data_root'], 'SAMPLED_POSES')
        self.out_dir = osp.join(opt['data_root'], 'NOISY_POSES')

        self.out_dir = osp.join(self.out_dir, opt['mode'])
        if opt['mode'] == 'gaussian':
            self.out_dir = self.out_dir + '_' + str(opt['variance'])
        os.makedirs(self.out_dir, exist_ok=True)

        self.dist_cal = geo()

    def gen_data(self):
        for split in self.data_splits:
            print(split)

            if self.parallel:
                split = self.seq_file.split('/')[0]

            ds_dir = osp.join(self.raw_data_dir, split)
            out_ds_dir = osp.join(self.out_dir, split)
            os.makedirs(out_ds_dir, exist_ok=True)
            seqs = sorted(os.listdir(ds_dir))

            for seq in seqs:
                if self.parallel:
                    seq = self.seq_file.split('/')[1]

                if osp.exists(osp.join(out_ds_dir, seq)):
                    print('Done...', split, seq)
                    continue

                if not 'npz' in seq:
                    continue

                # read pose data
                seq_path = osp.join(ds_dir, seq)
                if not osp.exists(seq_path):
                    print('Missing sequence file...', seq_path)
                    continue

                res_dict = self.gen_single_seq(seq_path=seq_path)

                np.savez(osp.join(out_ds_dir, seq), dist=res_dict['dist_all'], noisy_quats=res_dict['perturbed_quats'],
                         noisy_pose_aa=res_dict['perturbed_poses_aa'], nn_quats=res_dict['nn_quats'], nn_poses_aa=res_dict['nn_poses_aa'])
                print("Done for ...{}, pose shape... {}".format(split + '/' + seq, len(res_dict['nn_quats'])))

                if self.parallel:
                    break

            if self.parallel:
                break

    def gen_single_seq(self, seq_path):
        # create dataloader
        perturbed_data = PerturbData(data_path=seq_path,
                                     mode=self.mode,
                                     num_samples=self.num_samples,
                                     runs=self.runs,
                                     var=self.var)

        perturbed_dataloader = perturbed_data.get_loader()

        perturbed_quats, nn_quats = [], []
        perturbed_poses_aa, nn_poses_aa = [], []
        dist_all = []

        for data_batch in perturbed_dataloader:
            quat = data_batch.get('perturbed_quat')[0].numpy()
            aa_clean = data_batch.get('org_pose_aa')[0].numpy()

            k_quats, k_poses_aa, dist = nn_search(quat=quat,
                                                  index=self.index,
                                                  dist_cal=self.dist_cal,
                                                  all_poses_aa=self.all_poses_aa,
                                                  all_poses_quat=self.all_poses_quat,
                                                  k_faiss=self.k_faiss,
                                                  k_dist=self.k_dist,
                                                  org_poses=aa_clean)

            nn_quats.extend(k_quats)
            nn_poses_aa.extend(k_poses_aa)

            perturbed_quats.extend(quat)
            perturbed_poses_aa.extend(quaternion_to_axis_angle(torch.from_numpy(quat)).reshape(len(quat), -1))

            dist_all.extend(dist)

        nn_quats, nn_poses_aa = np.array(nn_quats), np.array(
            nn_poses_aa)  # [N, k_dist, 21, 4], [N, k_dist, 63]
        perturbed_quats, perturbed_poses_aa = np.array(perturbed_quats), np.array(
            perturbed_poses_aa)  # [N, 21, 4], [N, 63]
        dist_all = np.array(dist_all)  # [N, 1]

        res_dict = {
            'nn_quats': nn_quats,
            'nn_poses_aa': nn_poses_aa,
            'perturbed_quats': perturbed_quats,
            'perturbed_poses_aa': perturbed_poses_aa,
            'dist_all': dist_all
        }

        return res_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preparing pose and distance paired data for training NRDF")
    parser.add_argument('--config', '-c', default='configs/data_gen.yaml', type=str, help='Path to config file.')
    parser.add_argument("--parallel", "-p", action="store_true", help="executing in parallel")
    parser.add_argument("--seq_file", "-f", default='ACCAD/s009.npz', type=str, help='Path to seq file.')

    args = parser.parse_args()
    
    opt_args = load_config(args.config)

    data_generator = DataGenerator(opt=opt_args, parallel=args.parallel, seq_file=args.seq_file)
    data_generator.gen_data()

    print("Done...")
