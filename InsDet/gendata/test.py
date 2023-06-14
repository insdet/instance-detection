import cv2
import glob
import os
import re
import sys
import numpy as np
from PIL import Image


def get_palette():
    palette = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
               [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
               [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
               [128, 64, 128]]
    return np.asarray(palette)


color_palette = get_palette().flatten()
color_palette = color_palette.tolist()
datadir = "/home/SQQ/svid/RealWorld/RealObject"
destdir = "/home/SQQ/svid/RealWorld/Mask"

source_list = sorted(glob.glob(os.path.join(datadir, '*')))
for _, source_dir in enumerate(source_list):
    object_name = source_dir.split('/')[-1]

    if object_name != '034_thermos_flask_muji':
        continue

    target_dir = os.path.join(datadir, object_name)
    mask_paths = sorted([p for p in glob.glob(os.path.join(source_dir, 'masks', '*'))
                         if re.search('/*\.(jpg|jpeg|png|gif|bmp)', str(p))])

    if not os.path.exists(os.path.join(destdir, object_name, 'masks')):
        os.makedirs(os.path.join(destdir, object_name, 'masks'))

    for _, mask_path in enumerate(mask_paths):
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = np.where(mask != 0, 1, 0).astype('uint8')
        base_name = os.path.basename(mask_path)

        with Image.fromarray(mask, mode="P") as png_image:
            png_image.putpalette(color_palette)
            png_image.save(os.path.join(destdir, object_name, 'masks', base_name))

    print(object_name + ' Done !')
