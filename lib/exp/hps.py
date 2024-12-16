"""image pose estimation from images"""
import argparse
# General config

import shutil
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import os
import cv2
import pickle
import smplx
from body_model import BodyModel
from rdf_opt.model.posendf import PoseNDF
from configs.config import load_config

device = 'cuda'

from pytorch3d.transforms import matrix_to_axis_angle, axis_angle_to_quaternion, quaternion_to_axis_angle
from render_smplerx import joints_coco, rotate_mesh, visualise_mesh
from exp_utils import renderer, quat_flip, rel_change
from rdf_opt.exp.optimizers import quaternion_SGD, hybrid_SGD
from hps_utils import euc_err, visualise_pts, projection, visualise_mesh, joints_coco, rotate_mesh, GMoF


class ImagePose(object):
    def __init__(self, pose_prior, bm_dir_path, debug=False, device='cuda:0', batch_size=1, gender='male', render=False,
                 save_video=True, rho=0.0):
        self.debug = debug
        self.device = device
        self.pose_prior = pose_prior
        self.body_model = BodyModel(bm_path=bm_dir_path, model_type='smpl', batch_size=batch_size, num_betas=10).to(
            device=device)

        self.render = render
        self.save_video = save_video
        self.ftol = 0.0001  # relative error tolerance for loss
        self.batch_size = batch_size
        self.quat_lr = 0.0005
        self.hybrid_lr = 0.005
        self.quat_step = 0
        self.hybrid_step = 10
        self.rho = rho

        if self.rho > 0:
            self.robustifier = GMoF(self.rho)

    def init_opt(self, cam_transform, cam_intrinsics, gt_2djoints, gt_pose, gt_betas, gt_trans, gt_global_orient):
        """"initialise the optimization variables and observation model"""

        # create GT
        self.gt_smpl = {'pose': gt_pose, 'betas': gt_betas, 'trans': gt_trans, '2djoints': gt_2djoints}

        # create optimization variables
        self.pose = gt_pose
        self.betas = gt_betas
        self.trans = gt_trans
        self.global_orient = gt_global_orient

        # create the camera model  for projection
        self.center = torch.zeros([batch_size, 2], device=device)
        self.center[:, 0] = cam_intrinsics[0, 2]
        self.center[:, 1] = cam_intrinsics[1, 2]
        self.camera_mat = cam_intrinsics[:2, :2].unsqueeze(0).repeat(batch_size, 1, 1)

        self.camera_transform = cam_transform

    def projection(self, points_3d, batch_size):
        """project similar to smplify-x"""

        homog_coord = torch.ones(list(points_3d.shape)[:-1] + [1], dtype=points_3d.dtype, device=device)
        # Convert the points to homogeneous coordinates
        points_h = torch.cat([points_3d, homog_coord], dim=-1)

        projected_points = torch.einsum('bki,bji->bjk', [self.camera_transform, points_h])

        img_points = torch.div(projected_points[:, :, :2], projected_points[:, :, 2].unsqueeze(dim=-1))
        img_points = torch.einsum('bki,bji->bjk', [self.camera_mat, img_points]) + self.center.unsqueeze(dim=1)

        return img_points

    def get_loss_weights(self):
        """Set loss weights"""
        loss_weight = {'j2j': lambda cst, it: 10. ** 2 * cst / ((1 + it)),
                       'betas': lambda cst, it: 10. ** 1 * cst,
                       'pose_pr': lambda cst, it: 10. ** 4 * cst * cst * (1 + it)
                       }
        return loss_weight

    @staticmethod
    def backward_step(loss_dict, weight_dict, it):
        w_loss = dict()
        for k in loss_dict:
            w_loss[k] = weight_dict[k](loss_dict[k], it)

        tot_loss = list(w_loss.values())
        tot_loss = torch.stack(tot_loss).sum()
        return tot_loss

    def optimize(self, pose_init, betas_init, joints_gt, cam_transform, cam_intrinsics, trans, coco_gt=None,
                 coco_conf=None, itr=2, trans_init=None, global_orient_init=None, use_prior=True):
        """optimize pose and betas"""
        smpl_init = self.body_model(betas=betas_init, pose_body=pose_init, root_orient=self.global_orient)

        # Optimizer
        pose_init = axis_angle_to_quaternion(smpl_init.body_pose.reshape(-1, 23, 3)[:, :21, :])
        pose_hands = axis_angle_to_quaternion(smpl_init.body_pose.reshape(-1, 23, 3)[:, 21:, :])
        betas_init = smpl_init.betas
        pose_init.requires_grad = True
        pose_hands.requires_grad = False
        betas_init.requires_grad = True

        hybrid_optimizer = hybrid_SGD([pose_init, betas_init], lr=self.hybrid_lr)

        loop = tqdm(range(itr))
        loop.set_description('Optimizing prior and data term with betas prior')

        # forward body model
        for i in loop:

            loss_dict = dict()

            hybrid_optimizer.zero_grad()
            dis_val = self.pose_prior(pose_init, train=False)['dist_pred']
            loss_dict['pose_pr'] = torch.mean(dis_val)

            # smpl_init = self.body_model.forward(betas= smpl_init.betas, pose_body=smpl_init.body_pose
            full_body_pose = torch.cat([pose_init, pose_hands], dim=1)
            smpl_init = self.body_model(betas=betas_init,
                                        pose_body=quaternion_to_axis_angle(full_body_pose).reshape(-1, 69),
                                        root_orient=self.global_orient)
            joints3d = smpl_init.Jtr + self.trans.unsqueeze(dim=1)
            projected_points = self.projection(joints3d, len(joints3d))

            # calculate the data loss
            if coco_gt is not None:
                coc_jts = joints_coco(smpl_init.vertices + self.trans.unsqueeze(dim=1), joints3d)
                projected_points_coco = projection(cam_transform, cam_intrinsics, coc_jts, len(pose_body))
                # calculate the loss
                loss_dict['j2j'] = projected_points_coco - coco_gt
                self.rho = 0.0
                if self.rho > 0:
                    joint_diff = self.robustifier(loss_dict['j2j'])
                    loss_dict['j2j'] = torch.sum(coco_conf.unsqueeze(dim=-1) ** 2 * joint_diff)
                else:
                    loss_dict['j2j'] = (torch.sqrt(torch.sum(loss_dict['j2j'] * loss_dict['j2j'],
                                                             dim=-1)) * coco_conf).mean()  # Todo: change to weighted loss?
            else:
                # calculate the loss
                loss_dict['j2j'] = projected_points - self.gt_smpl['2djoints']
                loss_dict['j2j'] = torch.sqrt(
                    torch.sum(loss_dict['j2j'] * loss_dict['j2j'], dim=-1)).mean()  # Todo: change to weighted loss?

            loss_dict['betas'] = torch.sqrt(torch.sum(self.betas * self.betas, dim=-1)).mean()
            tot_loss = self.backward_step(loss_dict, self.get_loss_weights(), i)
            # print('loss: ', tot_loss.item())

            tot_loss.backward(retain_graph=True)
            # hybrid_optimizer.step()
            hybrid_optimizer.step(tot_loss, loss_dict['j2j'], loss_dict['pose_pr'])
            # hybrid_optimizer.step()

            l_str = 'Iter: {}'.format(i)

            for k in loss_dict:
                l_str += ', {}: {:0.8f}'.format(k, loss_dict[k].mean().item())
                loop.set_description(l_str)

        print('optimization done')
        return smpl_init.body_pose, smpl_init.betas


def overlay_points_on_image(image_path, points, pts2d):
    # Load the image
    image = cv2.imread(image_path)

    # Draw a circle at each point
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

    for point in pts2d:
        cv2.circle(image, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

    return image


def visualise_pts(im_dir, im_names, projected_pts, pose2d_18=None, out_dir=None, prefix='image'):
    for idx, im_name in enumerate(im_names):
        im_path = os.path.join(im_dir, 'image_{:05}.jpg'.format(im_name))
        if os.path.exists(im_path):
            image = overlay_points_on_image(im_path, projected_pts[idx], pose2d_18[idx])
            cv2.imwrite(os.path.join(out_dir, prefix + '_{:05}.jpg'.format(im_name)), image)
            print(os.path.join(out_dir, prefix + '_{:05}.jpg'.format(im_name)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image fitting using pytorch3d projection'
    )

    # parser for loading image and pose file
    parser.add_argument('--seq_file', type=str,
                        default='/BS/garvita3/static00/image_pose_estimation/images_data/img_datasets/pw3d/sequenceFiles/test',
                        help='path to 2d pose ')
    parser.add_argument('--seq_name', type=str, default='flat_packBags_00_smpler_x_h32', help='path to 2d pose ')
    parser.add_argument('--image_folder', type=str,
                        default='/BS/garvita3/static00/image_pose_estimation/images_data/img_datasets/pw3d/imageFiles',
                        help='path to image  ')
    parser.add_argument('--outpath_folder', type=str,
                        default='/BS/garvita3/static00/image_pose_estimation/images_data/img_datasets/pw3d/out_rdf_prior_coco',
                        help='path to save results')
    parser.add_argument('--bm_dir_path', type=str, default='/BS/garvita/work/SMPL_data/models',
                        help='path to body model')
    parser.add_argument('--smplerx_folder', type=str,
                        default='/BS/garvita3/static00/image_pose_estimation/images_data/img_datasets/pw3d/smplerx_test_smpl/results_crop',
                        help='path to save results')
    parser.add_argument('--rho', type=float, default=100.0, help='path to save results')

    args = parser.parse_args()

    test_folder_root = args.seq_file
    image_folder_root = args.image_folder
    outpath_folder_root = args.outpath_folder
    os.makedirs(outpath_folder_root, exist_ok=True)
    bm_dir_path = args.bm_dir_path
    # seq_name = args.seq_name

    # load prior model
    ### load the model
    exp_name = 'flip_ours_softplus_l1_0.0001__10000_dist0.5_eik0.0_man0.1'
    config_file = '/BS/humanpose/static00/yannan/checkpoints_new/{}/config.yaml'.format(exp_name)
    ckpt = '/BS/humanpose/static00/yannan/checkpoints_new/{}/checkpoints/checkpoint_epoch_best.tar'.format(exp_name)

    opt = load_config(config_file)
    net = PoseNDF(opt)
    device = 'cuda:0'
    ckpt = torch.load(ckpt, map_location='cpu')['model_state_dict']
    net.load_state_dict(ckpt)
    net.eval()
    net = net.to(device)

    smplerx_folder_root = args.smplerx_folder
    all_err_j2j = []
    all_err_v2v = []
    all_err_coco = []
    all_pck3d = []
    all_pck = []
    all_samples = 0
    all_seqs = sorted(os.listdir(smplerx_folder_root))[:-1]
    side_view = False
    save_overlay = True
    col = (200 / 255, 100 / 255, 205 / 255, 1.0)
    # col_opt = (20/255, 177/255, 20/255, 1.0)
    col_opt = (200 / 255, 100 / 255, 205 / 255, 1.0)
    print(all_seqs)
    for seq in all_seqs:
        seq_name = seq[:-4]
        if 's32' in seq:
            continue

        seq_file = os.path.join(test_folder_root, seq[:-17] + '.pkl')
        image_folder = os.path.join(image_folder_root, seq[:-17])
        smplerx_file = os.path.join(smplerx_folder_root, seq)
        outpath_folder = os.path.join(outpath_folder_root, seq[:-4])
        os.makedirs(outpath_folder, exist_ok=True)

        with open(seq_file, 'rb') as f:
            gt_data = pickle.load(f, encoding='latin1')

        # if len(data['genders']) !=1:
        #     return
        im_out = os.path.join(outpath_folder, 'img_out')
        os.makedirs(im_out, exist_ok=True)
        res_out = os.path.join(outpath_folder, 'smpl_out_final2')
        os.makedirs(res_out, exist_ok=True)
        gender = 'male'
        if gt_data['genders'] == 'f':
            gender = 'female'

        # load GT values
        # batch_size = len(data['poses'][0])
        # load prediction values
        smplerx_file = os.path.join(smplerx_folder_root, seq)
        smplerx_data = np.load(smplerx_file)

        select_id = smplerx_data['valid_idx']
        pose_body_pred = torch.from_numpy(smplerx_data['pose_body']).to(device=device)
        betas_pred = torch.from_numpy(smplerx_data['betas']).to(device=device)
        globalo_pred = torch.from_numpy(smplerx_data['global_orient']).to(device=device)
        batch_size = len(select_id)

        # load GT values
        # select_id = np.where(data['img_frame_ids'] == pred_id)[0][0]
        pose_body = torch.from_numpy(
            np.array(gt_data['poses'][0][select_id, 3:72].astype(np.float32).reshape(-1, 69))).to(device=device)
        global_orient = torch.from_numpy(
            np.array(gt_data['poses'][0][select_id, :3].astype(np.float32).reshape(-1, 3))).to(device=device)
        trans = torch.from_numpy(np.array(gt_data['trans'][0][select_id, :3].astype(np.float32).reshape(-1, 3))).to(
            device=device)
        betas = torch.from_numpy(np.array(gt_data['betas'][0][:10].astype(np.float32))).to(device=device).unsqueeze(
            0).repeat(len(pose_body), 1)
        pose2d_18 = torch.from_numpy(np.array(gt_data['poses2d'][0][select_id].astype(np.float32))).to(
            device=device).permute(0, 2, 1)[:, :, :2]
        cam_transform = torch.from_numpy(np.array(gt_data['cam_poses'][select_id].astype(np.float32))).to(device=device)
        cam_intrinsics = torch.from_numpy(np.array(gt_data['cam_intrinsics'].astype(np.float32))).to(device=device)
        pose_conf = torch.from_numpy(np.array(gt_data['poses2d'][0][select_id].astype(np.float32))).to(
            device=device).permute(0, 2, 1)[:, :, 2]

        # for visualisation
        focal = np.array([gt_data['cam_intrinsics'][0, 0], gt_data['cam_intrinsics'][1, 1]])
        princpt = np.array([gt_data['cam_intrinsics'][0, 2], gt_data['cam_intrinsics'][1, 2]])
        pose_prior = None

        # create initial body model
        net.eval()
        net = net.to(device)
        pose_prior = net
        impose_opt = ImagePose(pose_prior, bm_dir_path + '/smpl', gender=gender, debug=False, device=device,
                               batch_size=batch_size, rho=args.rho)
        body_model_output_init = impose_opt.body_model.forward(betas=betas_pred, pose_body=pose_body_pred,
                                                               root_orient=global_orient)
        joints3d_init = (body_model_output_init.Jtr + trans.unsqueeze(dim=1))
        mesh_cam_init = rotate_mesh(cam_transform, cam_intrinsics,
                                    body_model_output_init.vertices + trans.unsqueeze(dim=1), len(joints3d_init),
                                    side_view).detach().cpu().numpy()

        # if save_overlay:
        #     visualise_mesh(image_folder, select_id, mesh_cam_init, body_model_output_init.faces.detach().cpu().numpy(), focal,  princpt, res_out, prefix='in',col=col)
        if side_view:
            visualise_mesh(image_folder, select_id, mesh_cam_init, body_model_output_init.faces.detach().cpu().numpy(),
                           focal, princpt, res_out, prefix='in_side', rewrite=True, col=col)

        # init projected pts
        projected_points_init = projection(cam_transform, cam_intrinsics, joints3d_init, len(pose_body))
        # create coco joints
        coc_jts_init = joints_coco(body_model_output_init.vertices + trans.unsqueeze(dim=1), joints3d_init)
        projected_points_coco_init = projection(cam_transform, cam_intrinsics, coc_jts_init, len(pose_body))
        # if save_overlay:
        #     visualise_pts(image_folder,select_id, projected_points, projected_points_coco, im_out, prefix='jts')

        # create gt body model
        body_model_output_gt = impose_opt.body_model.forward(betas=betas, pose_body=pose_body,
                                                             root_orient=global_orient)
        joints3d_gt = (body_model_output_gt.Jtr + trans.unsqueeze(dim=1))
        mesh_cam_gt = rotate_mesh(cam_transform, cam_intrinsics, body_model_output_gt.vertices + trans.unsqueeze(dim=1),
                                  len(joints3d_gt)).detach().cpu().numpy()

        # if save_overlay:
        #     visualise_mesh(image_folder, select_id, mesh_cam_gt, body_model_output_gt.faces.detach().cpu().numpy(), focal,  princpt, res_out, prefix='gt')

        # gt projected pts
        projected_points_gt = projection(cam_transform, cam_intrinsics, joints3d_gt, len(pose_body))
        # create coco joints
        coc_jts_gt = joints_coco(body_model_output_gt.vertices + trans.unsqueeze(dim=1), joints3d_gt)
        projected_points_coco_gt = projection(cam_transform, cam_intrinsics, coc_jts_gt, len(pose_body))
        # if save_overlay:
        #     visualise_pts(image_folder,select_id, projected_points, projected_points_coco, im_out, prefix='jts')

        err_jts_init = euc_err(joints3d_init, joints3d_gt)
        err_v2v_init = euc_err(body_model_output_init.vertices, body_model_output_gt.vertices)

        # start optimization

        # initialization of variables
        pose_body_pred = pose_body_pred.detach()
        betas_pred = betas_pred.detach()
        impose_opt.init_opt(cam_transform, cam_intrinsics, projected_points_gt, pose_body, betas, trans, global_orient)
        pose_res, betas_res = impose_opt.optimize(pose_body_pred, betas_pred, joints3d_gt, cam_transform,
                                                  cam_intrinsics, trans, pose2d_18, coco_conf=pose_conf, itr=100)

        body_model_output_opt = impose_opt.body_model.forward(betas=betas_res, pose_body=pose_res,
                                                              root_orient=global_orient)
        # create gt pose

        joints3d_opt = body_model_output_opt.Jtr + trans.unsqueeze(dim=1)
        projected_points_opt = projection(cam_transform, cam_intrinsics, joints3d_opt, len(joints3d_opt))

        # if save_overlay:
        #     visualise_pts(image_folder,select_id, projected_points_opt, projected_points_opt, im_out, prefix='opt')

        mesh_cam_opt = rotate_mesh(cam_transform, cam_intrinsics,
                                   body_model_output_opt.vertices + trans.unsqueeze(dim=1), len(joints3d_opt),
                                   side_view).detach().cpu().numpy()
        if save_overlay:
            visualise_mesh(image_folder, select_id, mesh_cam_opt, body_model_output_opt.faces.detach().cpu().numpy(),
                           focal, princpt, res_out, prefix='opt', col=col_opt)

        if side_view:
            visualise_mesh(image_folder, select_id, mesh_cam_opt, body_model_output_opt.faces.detach().cpu().numpy(),
                           focal, princpt, res_out, prefix='opt_side', rewrite=True, col=col_opt)

        err_jts = euc_err(joints3d_opt, joints3d_gt)
        err_v2v = euc_err(body_model_output_opt.vertices, body_model_output_gt.vertices)
        # err_jts_coco = euc_err(coc_jts_pred, coc_jts)

        print('For seq: {}/{} Error in mm j2j: {:.4f},  v2v:{:.4f}'.format(seq, len(select_id), err_jts * 1000.,
                                                                           err_v2v * 1000.))
        print('For initial seq: {}/{} Error in mm j2j: {:.4f},  v2v:{:.4f}'.format(seq, len(select_id),
                                                                                   err_jts_init * 1000.,
                                                                                   err_v2v_init * 1000.))

        coc_jts_opt = joints_coco(body_model_output_opt.vertices + trans.unsqueeze(dim=1), joints3d_opt)
        projected_points_coco_opt = projection(cam_transform, cam_intrinsics, coc_jts_opt, len(pose_body))

        # calculate pck@0.2: PCK@0.2 == Distance between predicted and true joint < 0.2 * torso diameter

        torso_diameter = pose2d_18[:, 8] - pose2d_18[:, 11]
        torso_diameter = torch.sqrt(torch.sum(torso_diameter * torso_diameter, dim=-1))

        diff_2d = pose2d_18 - projected_points_coco_opt
        diff_2d = torch.sqrt(torch.sum(diff_2d * diff_2d, dim=-1))
        pck = diff_2d < 0.2 * torso_diameter.unsqueeze(1).repeat(1, 18)
        pck = torch.mean(torch.sum(pck, dim=-1) / 18).detach().cpu().numpy()

        # calculate pck@50mm

        diff_3d = joints3d_gt - joints3d_opt
        diff_3d = torch.sqrt(torch.sum(diff_3d * diff_3d, dim=-1))
        pck_3d = diff_3d < (50 / 1000.)
        pck_3d = torch.mean(torch.sum(pck_3d, dim=-1) / 21).detach().cpu().numpy()
        all_pck.append(pck)
        all_pck3d.append(pck_3d)

        all_err_j2j.append(err_jts)
        all_err_v2v.append(err_v2v)
        all_samples += len(select_id)

        np.savez(os.path.join(outpath_folder, seq), pose_body=pose_res.detach().cpu().numpy(),
                 betas=betas_res.detach().cpu().numpy(), j2j=err_jts, v2v=err_v2v, valid_idx=select_id)

    print('Error in j2j mm:', np.mean(np.array(all_err_j2j)) * 1000., all_samples)
    print('Error in v2v mm:', np.mean(np.array(all_err_v2v)) * 1000., all_samples)
    print('PCK:', np.mean(np.array(all_pck)) * 100., all_samples)
    print('PCK50:', np.mean(np.array(all_pck3d)) * 100., all_samples)