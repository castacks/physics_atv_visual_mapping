import os
import tqdm
import argparse
import subprocess

from tartandriver_utils.os_utils import is_kitti_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, required=True, help='base dir of run dirs to proc')
    parser.add_argument('--not_shallow', action='store_true', help='set this flag to look through everything in dir for kitti dirs instead of one level (much slower)')
    args = parser.parse_args()

    run_dirs = []

    if args.not_shallow:
        for root, dirs, files in os.walk(args.src_dir):
            if is_kitti_dir(root):
                run_dirs.append(root)
    else:
        for rdir in os.listdir(args.src_dir):
            run_dir = os.path.join(args.src_dir, rdir)
            if is_kitti_dir(run_dir):
                run_dirs.append(run_dir)

    print('found the following run dirs ({} total):'.format(len(run_dirs)))
    for rdir in sorted(run_dirs):
        print('\t' + rdir)

    base_cmd = "python3 get_voxel_inpainting_supervision.py --run_dir {}"

    success_dirs = []
    fail_dirs = []

    for ri, run_dir in enumerate(run_dirs):
        print("Proc {} ({}/{})".format(run_dir, ri+1, len(run_dirs)))

        # cmd = base_cmd.format(args.config_fp, run_dir)
        cmd = base_cmd.format(run_dir)

        res = subprocess.run(cmd.split(" "))

        if res.returncode == 0:
            success_dirs.append(run_dir)
        else:
            fail_dirs.append(run_dir)

    print('successfully processed {}/{} run dirs'.format(len(success_dirs), len(run_dirs)))

    if len(fail_dirs) > 0:
        print('should manually check:')
        for fp in fail_dirs:
            print('    ' + fp)