import glob
import os
from data_utils import resizemask

datadir = "/home/SQQ/svid/gendata/objects_centercrop/"
destdir = "/home/SQQ/svid/gendata/objects_downsize/"
factors = []
resolutions = [256, 256]

source_list = sorted(glob.glob(os.path.join(datadir, '*')))

for _, source_dir in enumerate(source_list):
    object_name = source_dir.split('/')[-1]
    target_dir = os.path.join(destdir, object_name)

    resizemask(os.path.join(source_dir, 'masks'), os.path.join(target_dir, 'masks'),
               factors=factors, resolutions=[resolutions])
