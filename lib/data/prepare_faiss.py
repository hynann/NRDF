import sys 
sys.path.append('')

import os
import os.path as osp
import argparse
import torch
import faiss
import numpy as np

from lib.data.dataloaders import FaissData

from lib.utils.data_utils import amass_splits
from lib.utils.transforms import axis_angle_to_quaternion


def aa2faisskey(aa):
    bs = len(aa)

    # prepare quaternions relative to the parent
    quat_local = axis_angle_to_quaternion(torch.from_numpy(aa.reshape(bs, -1, 3))).detach().numpy()
    key_faiss = quat_local.reshape(bs, -1)

    return key_faiss


def build_faiss(data_splits, sampled_pose_dir, batch_size=256):
    all_poses_quat = []
    all_poses_aa = []

    for split in data_splits:
        ds_dir = osp.join(sampled_pose_dir, split)
        seqs = sorted(os.listdir(ds_dir))
        print(split, len(seqs))

        for seq in seqs:
            if not 'npz' in seq:
                continue

            amass_seq = FaissData(osp.join(ds_dir, seq), batch_size=batch_size, num_workers=3)
            amass_loader = amass_seq.get_loader(shuffle=False, drop_last=True)

            for query_batch in amass_loader:
                pose_aa = query_batch.get('pose')[:, :63].numpy()  # numpy: (bs, 63)

                # key_faiss = aa2faisskey(pose) # numpy: [bs, 84]
                pose_quat = axis_angle_to_quaternion(torch.from_numpy(pose_aa.reshape(len(pose_aa), -1, 3))).detach().numpy()
                pose_quat = pose_quat.reshape(len(pose_aa), -1)
                
                all_poses_quat.extend(pose_quat)
                all_poses_aa.extend(pose_aa)

    all_faiss = np.array(all_poses_quat).astype(np.float32)
    faiss_dim = all_faiss.shape[-1]
    index = faiss.index_factory(faiss_dim, "Flat")
    index.train(all_faiss)
    index.add(all_faiss)

    all_poses_quat = np.array(all_poses_quat) # [N, 84]
    all_poses_aa = np.array(all_poses_aa) # [N, 63]

    return index, all_poses_quat, all_poses_aa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preparing faiss data for training PoseNDF")

    parser.add_argument('--data_root', '-d', type=str, default='/home/ubuntu/data/nrdf_data',
                        help='Data path')
    parser.add_argument("-m", "--mode", type=str, default='train', help="AMASS subset")
    parser.add_argument("-bs", "--batch_size", type=int, default=256, help="Batch size of faiss dataloader")

    args = parser.parse_args()

    data_splits = sorted(amass_splits[args.mode])
    sampled_pose_dir = osp.join(args.data_root, 'SAMPLED_POSES')
    out_dir = osp.join(args.data_root, 'FAISS_MODEL')
    os.makedirs(out_dir, exist_ok=True)

    index, all_poses_quat, all_poses_aa = build_faiss(data_splits=data_splits,
                                                      sampled_pose_dir=sampled_pose_dir,
                                                      batch_size=args.batch_size)

    # save faiss model
    model_name = 'faiss.index'
    faiss.write_index(index, os.path.join(out_dir, model_name))

    # save all pose data
    all_data_name = 'all_data.npz'
    np.savez(os.path.join(out_dir, all_data_name), all_poses_aa=all_poses_aa, all_poses_quat=all_poses_quat)

    print('Done.')
