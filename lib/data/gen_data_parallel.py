"""Generate script for creating training data from AMASS
 Please change L23:L38, if you are using slurm """

import sys
sys.path.append('')

import os
import os.path as osp
import argparse

from lib.utils.data_utils import amass_splits


def main():
    bash_path = args.bash_file
    amass_datas = sorted(amass_splits['train'])

    with open(bash_path, 'w+') as fp:

        count = 0
        if args.use_slurm:
            # ToDo: change these paths accordingly
            fp.write("#!/bin/bash" + "\n")
            fp.write("#SBATCH -p gpu-2080ti" + "\n")
            fp.write("#SBATCH --signal=B:SIGTERM@120" + "\n")
            fp.write("#SBATCH -c 5" + "\n")
            fp.write("#SBATCH --mem-per-cpu=6144" + "\n")
            fp.write("#SBATCH --gres gpu:1" + "\n")
            fp.write("#SBATCH -t 0-12:00:00" + "\n")
            fp.write("#SBATCH -a 1-409%409" + "\n")
            fp.write('#SBATCH -o "logs/job_%j.out"' + "\n")
            fp.write('#SBATCH -e "logs/job_%j.err"' + "\n")
            fp.write("#SBATCH --gres gpu:1" + "\n")
            fp.write("source $HOME/.bashrc" + "\n")
            fp.write("conda activate nrdf" + "\n")
            fp.write("cd /home/ponsmoll/pba406/NRDF" + "\n") # project folder
            fp.write("export PATH=/mnt/qb/work/ponsmoll/pba406/anaconda/envs/nrdf/bin:$PATH" + "\n") # optional, specify the conda env
            fp.write("case $SLURM_ARRAY_TASK_ID in" + "\n")

        out_dir = osp.join(args.data_root, 'NOISY_POSES')
        sampled_pose_dir = osp.join(args.data_root, 'SAMPLED_POSES')

        for amass_data in amass_datas:
            ds_dir = os.path.join(sampled_pose_dir, amass_data)
            seqs = sorted(os.listdir(ds_dir))
            for seq in seqs:
                if os.path.exists(os.path.join(out_dir, amass_data, seq)):
                    print('done....', amass_data,  seq)
                    continue
                if 'npz' not in seq:
                    continue
                if args.use_slurm:
                    fp.write("\t {})".format(count) + "\n")
                    fp.write("\t\t\t")

                fp.write(
                    "python lib/data/gen_data.py --config configs/data_gen.yaml --seq_file {}/{} --parallel".format(amass_data, seq))
                count += 1

                if args.use_slurm:
                    fp.write("& \n")
                    fp.write("\t\t\t;;\n")
                else:
                    fp.write("\n")

        if args.use_slurm:
            fp.write("esac" + "\n\n" + "wait")

        print("Total sequences to be processed....", count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for creating PoseNDF dataset")

    # Paths to output files
    parser.add_argument( '-d', '--data_root', type=str, default='/home/ubuntu/data/nrdf_data',
                        help='Data path')
    parser.add_argument("-bf", "--bash_file", type=str, default="./data/gen_data.sh",
                        help="Path to the bash script file")
    parser.add_argument('-sl', '--use_slurm',  action="store_true", help="Using slurm for creating dataset")

    args = parser.parse_args()
    main()
