import glob
import os
import re
import sys
import numpy as np
from PIL import Image
from data_utils import centercrop

datadir = "../InsDet/Objects"
destdir = "../gendata/Object_centercrop"

cropsize = []

if len(cropsize) == 0:
    use_bbox = True
else:
    use_bbox = False

source_list = sorted(glob.glob(os.path.join(datadir, '*')))

for _, source_dir in enumerate(source_list):
    object_name = source_dir.split('/')[-1]
    target_dir = os.path.join(destdir, object_name)

    image_paths = sorted([p for p in glob.glob(os.path.join(source_dir, 'images', '*'))
                          if re.search('/*\.(jpg|jpeg|png|gif|bmp)', str(p))])
    mask_paths = sorted([p for p in glob.glob(os.path.join(source_dir, 'masks', '*'))
                         if re.search('/*\.(jpg|jpeg|png|gif|bmp)', str(p))])

    if not os.path.exists(os.path.join(target_dir, 'images')):
        os.makedirs(os.path.join(target_dir, 'images'))
    if not os.path.exists(os.path.join(target_dir, 'masks')):
        os.makedirs(os.path.join(target_dir, 'masks'))

    if use_bbox:
        bbox_file = os.path.join(source_dir, "bbox.npy")
        if os.path.exists(bbox_file):
            bbox = np.load(bbox_file)
        else:
            print('ERROR: Please provide crop size or "bbox.npy". Aborting')
            sys.exit()

        img_size = bbox[-1]
        crop_size = bbox[:-1]
        new_x = np.max(crop_size[:, 1]) - np.min(crop_size[:, 0])
        new_y = np.max(crop_size[:, 3]) - np.min(crop_size[:, 2])

        cropsize = [np.max([new_x, new_y]), np.max([new_x, new_y])]

    for i, file_path in enumerate(zip(image_paths, mask_paths)):
        img = Image.open(file_path[0])
        mask = Image.open(file_path[1])

        new_img = centercrop(img, cropsize)
        new_mask = centercrop(mask, cropsize)

        # save
        filename = os.path.splitext(os.path.basename(file_path[0]))[0]
        # new_img.save(os.path.join(target_dir, 'images', filename + '.jpg'))
        new_img.save(os.path.join(target_dir, 'images', filename + '.png'))
        new_mask.save(os.path.join(target_dir, 'masks', filename + '.png'))

    print('Center Crop: ' + object_name + ' Done!')
