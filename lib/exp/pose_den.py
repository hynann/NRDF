import sys
sys.path.append('')

import os
import os.path as osp
import argparse
import numpy as np
import torch
import datetime

from tqdm import tqdm as tqdm

from configs.config import load_config

from lib.model.nrdf import NRDF
from lib.data.gen_data import nn_search
from lib.utils.data_utils import geo, load_faiss
from lib.utils.loss_utils import gradient, v2v_err, quat_geo_global
from lib.utils.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle
from lib.core.rdfgrad import rdfgrad


class Projector(object):
    def __init__(self, model_dir, noisy_pose_path=None, noisy_pose=None, device='cuda:0'):
        """

        Args:
            model_dir: Path to  the pretrained NRDF checkpoint
            noisy_pose_path: [optional] Path to the input noisy pose file, npz format
            noisy_pose: [optional] Input noisy pose: numpy [bs, nj*3]
            device: cuda or cpu

        """
        self.device = device
        
        # create output dir
        now = datetime.datetime.now()
        self.op_dir = osp.join('outputs', now.strftime("%Y-%m-%d-%H-%M-%S"))
        os.makedirs(self.op_dir, exist_ok=True)
        print(f'Created output dir: {self.op_dir}')

        # load pretrained NRDF model
        self._load_model(model_dir)

        # initialize input noisy pose
        self.noisy_pose = None
        if noisy_pose_path is not None:
            self.noisy_pose = self._load_noisy_pose(noisy_pose_path)  # torch [bs, nj*3]
        else:
            # self.noisy_pose = torch.from_numpy(noisy_pose).to(torch.float32).to(self.device) # torch [bs, nj*3]
            self.noisy_pose = noisy_pose # torch [bs, nj*3]
            self.noisy_pose = self.noisy_pose.to(torch.float32).to(self.device)

    def project(self, step_size=1.0, iterations=200, save_all_step=True):
        """

        Args:
            step_size: alpha in Eq.(8)
            iterations: Max iteration during projection
            save_all_step: if true: save intermediate poses in all projection steps, else save the converged poses only

        Returns: step_aa: numpy [iterations+1, bs, nj*3], result poses in all projection steps, axis-angle format
                 step_dist: numpy [iterations, bs], predicted distance in all projection steps

        """
        bs, ndim = self.noisy_pose.shape # [bs, 63]
        n_joints = ndim // 3
        step_aa, step_dist = [], []
        
        step_aa.append(self.noisy_pose.detach().cpu().numpy())
        
        noisy_pose_quat = axis_angle_to_quaternion(self.noisy_pose.reshape(-1, n_joints, 3)) # [bs, nj, 4]
        noisy_pose_quat.requires_grad = True

        for _ in tqdm(range(iterations)):
            dist_pred = self.model(noisy_pose_quat, train=False)['dist_pred']

            # Euclidean gradient returned by network backpropagation
            grad_val = gradient(noisy_pose_quat, dist_pred).reshape(-1, n_joints, 4)

            dist = dist_pred.reshape(-1, 1, 1)

            # Riemannian gradient descent
            noisy_pose_quat = rdfgrad(egrad=grad_val, q=noisy_pose_quat, dist=dist, step_size=step_size, device=self.device)

            noisy_pose_quat = noisy_pose_quat.detach()
            noisy_pose_quat.requires_grad = True
            
            noisy_aa = quaternion_to_axis_angle(noisy_pose_quat).detach().cpu().numpy()
            step_dist.append(dist_pred.detach().cpu().numpy().reshape(bs, ))
            step_aa.append(noisy_aa.reshape(-1, n_joints*3))

        step_aa = np.array(step_aa)
        step_dist = np.array(step_dist)
        
        # save results
        output_path = osp.join(self.op_dir, 'res.npz')
        if save_all_step:
            np.savez(output_path, all_step_pose=step_aa, all_step_dist=step_dist)
        else:
            np.savez(output_path, pose=step_aa[-1], dist=step_dist[-1])

        print(f'Projection done, results saved in {output_path}')

        return step_aa, step_dist

    def _load_noisy_pose(self, path, subsample=True):
        noisy_pose = np.load(path)['noisy_pose_aa']
        
        if subsample:
            # randomly sample 100 poses
            subsample_indices = np.random.randint(0, len(noisy_pose), 100)
            noisy_pose = noisy_pose[subsample_indices]
        
        np.savez(osp.join(self.op_dir, 'input.npz'), pose=noisy_pose)
        noisy_pose = torch.from_numpy(noisy_pose).to(torch.float32).to(self.device)

        return noisy_pose
    
    def _load_model(self, model_dir):
        checkpoint_path = osp.join(model_dir, 'checkpoints', 'checkpoint_epoch_best.tar')
        config_file = osp.join(model_dir, 'config.yaml')
        
        self.model = NRDF(load_config(config_file))
        checkpoint = torch.load(checkpoint_path, map_location='cpu')['model_state_dict']
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        self.model = self.model.to(self.device)
    
    def cal_error(self, pose_aa,
                        bm_path,
                        faiss_model_dir,
                        k_faiss=1000,
                        k_dist=1):
        """

        Args:
            pose_aa: numpy [iterations+1, bs, nj*3], result pose in all projection steps, axis-angle format
            bm_path: SMPL body model path
            faiss_model_dir: Pretrained faiss model path
            k_faiss: Number of candidates selected by the kd tree
            k_dist: Number of final nearest neighbor

        """

        bs, n_steps, n_dim = pose_aa.shape
        n_joints = n_dim // 3

        # load faiss related terms
        index, all_poses_aa, all_poses_quat = load_faiss(faiss_model_dir)
        input_pose = pose_aa[0] # [bs, 63]

        pose_quat = axis_angle_to_quaternion(torch.from_numpy(input_pose.reshape(-1, n_joints, 3))).detach().cpu().numpy() # [bs, nj, 4]
        dist_cal = geo()

        # search nearest neighbors
        k_quats, k_poses_aa, dist_gt = nn_search(quat=pose_quat, 
                                              index=index, 
                                              dist_cal=dist_cal, 
                                              all_poses_aa=all_poses_aa, 
                                              all_poses_quat=all_poses_quat, 
                                              k_faiss=k_faiss, 
                                              k_dist=k_dist)
        
        nn_pose = k_poses_aa[:, 0] # nearest neighbor [bs, 63]
        converged_pose = pose_aa[-1]

        # caculate v2v
        m2m_dist = v2v_err(converged_pose, nn_pose, bm_path=bm_path, device=self.device)
        geo_dist_glob = quat_geo_global(converged_pose, nn_pose, device=self.device)

        geo_m2m = 0.5 * geo_dist_glob + m2m_dist
        geo_m2m, m2m_dist = geo_m2m.mean(), m2m_dist.mean()

        print(f'delta_q+m2m error: {geo_m2m}')
        print(f'marker2marker error: {m2m_dist}')


def project_poses(args):
    projector = Projector(model_dir=args.model_dir,
                          noisy_pose_path=args.noisy_data_path,
                          device=args.device)
    res_aa, _ = projector.project(step_size=args.step_size, iterations=args.iterations)
    
    if args.eval:
        print('Caculating metrics...')
        projector.cal_error(res_aa,
                            bm_path=args.bm_path,
                            faiss_model_dir=args.faiss_model_dir,
                            k_faiss=args.k_faiss,
                            k_dist=args.k_dist)
    
    print('Done.')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Configurations for the pose denoising task by using NRDF")
    
    # projection related terms
    parser.add_argument("-nd", "--noisy_data_path", type=str, default="examples/noisy_pose.npz",
                        help="Path to the input noisy poses")
    parser.add_argument("-md", "--model_dir", type=str, default="checkpoints/amass_softplus_l1_0.0001_10000_dist0.5_eik0.0_man0.1",
                        help="Path to the pretrained NRDF model")
    parser.add_argument("-s", "--step_size", type=float, default=0.01, help="Step size (alpha) during projection")
    parser.add_argument("-it", "--iterations", type=int, default=200, help="Number of iterations during projection")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="CUDA or CPU")

    # evaluation related terms
    parser.add_argument("-e", "--eval", action="store_true", help="if calculate error")
    parser.add_argument("-bp", "--bm_path", type=str, default="/home/ubuntu/data/smplx_models",
                        help="SMPL body model path")     
    parser.add_argument("-fd", "--faiss_model_dir", type=str, default="/home/ubuntu/data/faiss_nrdf",
                        help="Path to the pretrained faiss model")
    parser.add_argument("-kf", "--k_faiss", type=int, default=1000, help="Number of candidates selected by the kd tree")
    parser.add_argument("-kd", "--k_dist", type=int, default=1, help="Number of final nearest neighbor")
    
    args = parser.parse_args()
    
    project_poses(args)
