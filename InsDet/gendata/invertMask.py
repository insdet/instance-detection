import argparse
import glob
import os
import re
import numpy as np
from PIL import Image
from data_utils import invertmask


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, help='source data path')
    parser.add_argument('--destdir', type=str, help='target data path')
    args = parser.parse_args()

    args.datadir = "/home/SQQ/svid/gendata/objects_downsize/079_polaroid_film"
    args.destdir = "/home/SQQ/svid/gendata/objects_bk/079_polaroid_film"

    return args


if __name__ == '__main__':
    args = get_args()

    image_paths = sorted([p for p in glob.glob(os.path.join(args.datadir, 'images', '*'))
                          if re.search('/*\.(jpg|jpeg|png|gif|bmp)', str(p))])
    mask_paths = sorted([p for p in glob.glob(os.path.join(args.datadir, 'masks', '*'))
                         if re.search('/*\.(jpg|jpeg|png|gif|bmp)', str(p))])

    if not os.path.exists(args.destdir):
        os.makedirs(os.path.join(args.destdir, 'images'))
        os.makedirs(os.path.join(args.destdir, 'masks'))

    for _, file_path in enumerate(zip(image_paths, mask_paths)):
        img = np.array(Image.open(file_path[0]))
        mask = np.array(Image.open(file_path[1]))
        base_filename = os.path.splitext(os.path.basename(file_path[0]))[0]

        new_img, new_mask = invertmask(img, mask)
        new_img = Image.fromarray(new_img.astype(np.uint8), mode="RGB")
        new_mask = Image.fromarray(new_mask.astype(np.uint8))

        new_img.save(os.path.join(args.destdir,'images', base_filename + '.png'))
        new_mask.save(os.path.join(args.destdir, 'masks', base_filename + '.png'))
