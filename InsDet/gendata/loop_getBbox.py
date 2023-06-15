import glob
import os
import re
import numpy as np
from PIL import Image
from data_utils import getbbox

datadir = "../RealWorld/RealObject"
destdir = "../RealWorld/RealObject"
savefile = "bbox.npy"
source_list = sorted(glob.glob(os.path.join(datadir, '*')))

for _, source_dir in enumerate(source_list):
    object_name = source_dir.split('/')[-1]
    target_dir = os.path.join(destdir, object_name)


    mask_paths = sorted([p for p in glob.glob(os.path.join(source_dir, 'masks', '*'))
                         if re.search('/*\.(jpg|jpeg|png|gif|bmp)', str(p))])

    bbox = []
    for _, mask_path in enumerate(mask_paths):
        mask = Image.open(mask_path)
        if mask.mode != 'L':
            mask = mask.convert('L')
            mask.save(mask_path)
        mask = np.array(mask)

        bbox.append(getbbox(mask, 1))
    bbox.append([0, mask.shape[0], 0, mask.shape[1]])  # the last array represent full bbox of the mask

    np.save(os.path.join(target_dir, savefile), np.array(bbox))
    print('BBox: ' + object_name + ' Done!')
