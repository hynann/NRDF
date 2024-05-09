import sys
sys.path.append('')

import os
import torch
import shutil
import argparse
from glob import glob
from torch.utils.tensorboard import SummaryWriter

from configs.config import load_config

from lib.model.nrdf import NRDF
from lib.data.dataloaders import PoseData
from lib.utils.loss_utils import AverageMeter


def train(opt, config_file):
    trainer = NRDFTrainer(opt)
    
    # copy the config file
    copy_config = '{}/{}/{}'.format(opt['experiment']['root_dir'], trainer.exp_name, 'config.yaml')
    shutil.copyfile(config_file,copy_config)

    for i in range(trainer.ep, opt['train']['max_epoch']):
        loss, epoch_loss = trainer.train_model(i)


class NRDFTrainer(object):

    def __init__(self, opt):
        self.device = opt['train']['device']
        self.enc_name = opt['model']['StrEnc']['name']

        self.train_dataset = PoseData(mode='train', noisy_dir=opt['data']['data_dir'],
                                        clean_dir=opt['data']['amass_dir'],
                                        batch_size=opt['train']['batch_size'],
                                        num_workers=opt['train']['num_worker'],
                                        num_pts=opt['data']['num_pts'])

        self.train_dataset = self.train_dataset.get_loader()
        self.njoint = opt['model']['StrEnc']['num_part']

        self.learning_rate = opt['train']['optimizer_param']
        self.model = NRDF(opt).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        self.batch_size = opt['train']['batch_size']
        self.ep = 0
        self.val_min = 10000.

        self.init_net(opt)

        if opt['train']['continue_train']:
            self.ep = self.load_checkpoint()

    def init_net(self, opt):
        self.iter_nums = 0

        # create exp name based on experiment params
        self.loss_weight = {'man_loss': opt['train']['man_loss'], 'dist': opt['train']['dist'],
                            'eikonal': opt['train']['eikonal']}

        self.exp_name = opt['experiment']['exp_name']
        self.loss = opt['train']['loss_type']

        self.exp_name = '{}_{}_{}_{}_{}_dist{}_eik{}_man{}'.format(self.exp_name, opt['model']['DFNet']['act'],
                                                                    self.loss, opt['train']['optimizer_param'],
                                                                    opt['data']['num_pts'], opt['train']['dist'],
                                                                    opt['train']['eikonal'], opt['train']['man_loss'])

        self.exp_path = '{}/{}/'.format(opt['experiment']['root_dir'], self.exp_name)
        self.checkpoint_path = self.exp_path + 'checkpoints/'
        if not os.path.exists(self.checkpoint_path):
            print(self.checkpoint_path)
            os.makedirs(self.checkpoint_path)
        self.writer = SummaryWriter(self.exp_path + 'summary')
        self.loss = opt['train']['loss_type']
        self.n_part = opt['experiment']['num_part']
        self.loss_mse = torch.nn.MSELoss()

        self.batch_size = opt['train']['batch_size']

        if self.loss == 'l1':
            self.loss_l1 = torch.nn.L1Loss()
        elif self.loss == 'l2':
            self.loss_l1 = torch.nn.MSELoss()
    
    def train_model(self, ep=None):
        self.model.train()
        epoch_loss = AverageMeter()
        individual_loss_epoch = {}
        for loss_term in self.loss_weight.keys():
            if self.loss_weight[loss_term]:
                individual_loss_epoch[loss_term] = AverageMeter()

        for i, inputs in enumerate(self.train_dataset):
            self.optimizer.zero_grad()

            dist = inputs['dist'].to(device=self.device)
            quat = inputs['pose'].reshape(-1, self.njoint, 4).to(device=self.device)
            man_pose = inputs['man_poses'].reshape(-1, self.njoint, 4).to(device=self.device)

            _, loss_dict = self.model(quat, dist, man_pose,
                                      eikonal=self.loss_weight['eikonal'])
            loss = 0.0
            for k in loss_dict.keys():
                loss += self.loss_weight[k] * loss_dict[k]
            loss.backward()
            self.optimizer.step()

            epoch_loss.update(loss, self.batch_size)
            for loss_term in self.loss_weight.keys():
                if loss_term in loss_dict.keys():
                    individual_loss_epoch[loss_term].update(loss_dict[loss_term], self.batch_size)

            self.iter_nums += 1
            # logger and summary writer
            for k in loss_dict.keys():
                self.writer.add_scalar("train/Iter_{}".format(k), loss_dict[k].item(), self.iter_nums)
        self.writer.add_scalar("train/epoch", epoch_loss.avg, ep)
        for k in loss_dict.keys():
            self.writer.add_scalar("train/epoch_{}".format(k), individual_loss_epoch[k].avg, ep)
        print("train/epoch", epoch_loss.avg, ep)
        log_loss = 'train/epoch '
        for k in loss_dict.keys():
            log_loss += str(loss_dict[k])
            log_loss += '  '
        print(log_loss)

        self.save_checkpoint(ep)
        return loss.item(), epoch_loss.avg
    
    def save_checkpoint(self, epoch):
        path = self.checkpoint_path + 'checkpoint_epoch_best.tar'

        if not os.path.exists(path):
            torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, path,
                       _use_new_zipfile_serialization=False)
        else:
            shutil.copyfile(path, path.replace('best', 'previous'))
            torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()}, path,
                       _use_new_zipfile_serialization=False)
    
    def load_checkpoint(self):
        checkpoints = glob(self.checkpoint_path + '/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(self.checkpoint_path))
            return 0

        path = self.checkpoint_path + 'checkpoint_epoch_best.tar'

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Loaded checkpoint from: {}'.format(path))

        epoch = checkpoint['epoch']
        return epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train NRDF.'
    )
    parser.add_argument('--config', '-c', default='configs/train.yaml', type=str, help='Path to config file.')
    parser.add_argument('--test', '-t', action="store_true")
    args = parser.parse_args()

    opt = load_config(args.config)

    train(opt, args.config)
