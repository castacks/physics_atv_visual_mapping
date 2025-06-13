import os
import argparse

import cv2
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_dir', type=str, required=True, help='path to orig mask dir')
    args = parser.parse_args()

    mask_fps = [x for x in os.listdir(args.mask_dir) if ('.png' in x) and not ('.mask') in x]

    print('proc these files:')
    for mfp in mask_fps:
        print(mfp)

    for mask_fp in mask_fps:
        src_fp = os.path.join(args.mask_dir, mask_fp)
        dst_fp = os.path.join(args.mask_dir, mask_fp[:-4] + '.mask.png')

        img = cv2.imread(src_fp)
        
        mask = img.mean(axis=-1) > 253.

        mask_img = np.stack([mask] * 3, axis=-1) * 255

        cv2.imwrite(dst_fp, mask_img)
