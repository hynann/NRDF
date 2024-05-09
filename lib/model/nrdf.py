import sys
sys.path.append('')

import torch

from lib.utils.loss_utils import gradient
from lib.model.net_modules import StructureEncoder, DFNet, StructureDecoder


class NRDF(torch.nn.Module):
    
    def __init__(self, opt):
        super(NRDF, self).__init__()

        self.device = opt['train']['device']
        self.njoints = opt['model']['DFNet']['num_parts']
        self.enc = StructureEncoder(opt['model']['StrEnc']).to(self.device)

        self.dfnet = DFNet(opt['model']['DFNet']).to(self.device)
        self.dec = StructureDecoder(opt['model']['StrEnc']).to(self.device)

        self.exp_name = opt['experiment']['exp_name']

        self.loss = opt['train']['loss_type']
        self.batch_size = opt['train']['batch_size']

        if self.loss == 'l1':
            self.loss_l1 = torch.nn.L1Loss()
        elif self.loss == 'l2':
            self.loss_l1 = torch.nn.MSELoss()

        self.loss_l2 = torch.nn.MSELoss()

    def train(self, mode=True):
        super().train(mode)

    def forward(self, pose_in, dist_gt=None, man_poses=None, train=True, eikonal=0.0):

        if train and eikonal > 0.0:
            pose_in.requires_grad = True

        if dist_gt is not None:
            dist_gt = dist_gt.reshape(-1)
        
        pose_latent = self.enc(pose_in)

        if train:
            dist_pred = self.dfnet(pose_latent)
            loss = self.loss_l1(dist_pred[:, 0], dist_gt)

            man_pose_latent = self.enc(man_poses)
            dist_man = self.dfnet(man_pose_latent)
            loss_man = (dist_man.abs()).mean()

            if eikonal > 0.0:
                grad_val = gradient(pose_in, dist_pred)
                eikonal_loss = ((grad_val.norm(2, dim=-1) - 1) ** 2).mean()
                return loss, {'dist': loss, 'man_loss': loss_man, 'eikonal': eikonal_loss}

            return loss, {'dist': loss, 'man_loss': loss_man}

        else:
            dist_pred = self.dfnet(pose_latent)
            return {'dist_pred': dist_pred}