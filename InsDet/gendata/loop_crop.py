import glob
import os
import re
import sys
import numpy as np
from PIL import Image

datadir = "../segment-anything/Objects"
destdir = "../gendata/objects_crop"
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

    for i, file_path in enumerate(zip(image_paths, mask_paths)):
        img = Image.open(file_path[0])
        mask = Image.open(file_path[1])

        new_img = img.crop((crop_size[i, 2], crop_size[i, 0], crop_size[i, 3], crop_size[i, 1]))
        new_mask = mask.crop((crop_size[i, 2], crop_size[i, 0], crop_size[i, 3], crop_size[i, 1]))

        # save
        filename = os.path.splitext(os.path.basename(file_path[0]))[0]
        new_img.save(os.path.join(target_dir, 'images', filename + '.jpg'))
        new_mask.save(os.path.join(target_dir, 'masks', filename + '.png'))

    print('Crop: ' + object_name + ' Done!')
