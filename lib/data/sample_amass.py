import sys
sys.path.append('')

import os
import os.path as osp
import argparse
import numpy as np

from lib.utils.data_utils import amass_splits


def sample_amass_poses(data_dir, keep_rate=0.3, mode='train'):
    in_dir = osp.join(data_dir, 'RAW_DATA')
    out_dir = osp.join(data_dir, 'SAMPLED_POSES')
    os.makedirs(out_dir, exist_ok=True)

    data_splits = sorted(amass_splits[mode])
    for split in data_splits:
        ds_dir = osp.join(in_dir, split)
        seqs = sorted(os.listdir(ds_dir))

        if not osp.exists(osp.join(out_dir, split)):
            os.makedirs(osp.join(out_dir, split), exist_ok=True)

        for seq in seqs:
            pose_body = []

            if 'LICENSE' in seq:
                continue

            out_path = osp.join(out_dir, split, f'{seq}.npz')
            if osp.exists(out_path):
                print('Done. ', out_path)
                continue

            npz_fnames = sorted(os.listdir(osp.join(ds_dir, seq)))
            for npz_fname in npz_fnames:
                if 'female' in npz_fname or 'male' in npz_fname or 'neutral' in npz_fname or 'shape' in npz_fname:
                    continue
                cdata = np.load(osp.join(ds_dir, seq, npz_fname))
                pose_len = len(cdata['poses'])

                # skip first and last frames to avoid initial standard poses, e.g. T pose
                cdata_ids = np.random.choice(list(range(int(0.1 * pose_len), int(0.9 * pose_len), 1)),
                                             int(keep_rate * 0.8 * pose_len),
                                             replace=False)
                if len(cdata_ids) < 1:
                    continue

                full_pose = cdata['poses'][cdata_ids].astype(np.float64)
                pose_body.extend(full_pose[:, 3:72])  # [keep_rate*0.8*N, 69]

            pose_body = np.array(pose_body)

            np.savez(out_path, pose_body=pose_body)
            print(mode, split, seq, len(pose_body))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Sample distinct poses from the raw AMASS dataset.'
    )

    parser.add_argument('--data_root', '-d', type=str, default='/home/ubuntu/data/nrdf_data',
                        help='Data path')
    parser.add_argument('--mode', '-m', type=str, default='train', help='AMASS train/test/valid subset')
    parser.add_argument('--sample_rate', '-r', type=float, default=0.3, help='Sampling rate')

    args = parser.parse_args()

    sample_amass_poses(data_dir=args.data_root,
                       keep_rate=args.sample_rate,
                       mode=args.mode)

    print('Done.')
