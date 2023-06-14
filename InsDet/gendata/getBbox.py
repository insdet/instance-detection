import argparse
import glob
import os
import re
import numpy as np
from PIL import Image
from data_utils import getbbox


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, help='source data path')
    parser.add_argument('--destdir', type=str, help='target data path')
    parser.add_argument('--exponent', type=int, default=1, help='multiple of factor for height/width')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    args.datadir = "/home/SQQ/svid/segment-anything/RealObject/079_polaroid_film"
    args.destdir = "/home/SQQ/svid/segment-anything/RealObject/079_polaroid_film"
    args.exponent = 1
    mask_paths = sorted([p for p in glob.glob(os.path.join(args.datadir, 'masks', '*'))
                         if re.search('/*\.(jpg|jpeg|png|gif|bmp)', str(p))])
    bbox = []
    for _, mask_path in enumerate(mask_paths):
        mask = np.array(Image.open(mask_path))

        bbox.append(getbbox(mask, args.exponent))

    np.save(os.path.join(args.destdir, 'bbox.npy'), np.array(bbox))
