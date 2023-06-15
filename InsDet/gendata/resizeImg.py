import argparse
import glob
import os
import sys
from data_utils import minify


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, help='source data path')
    parser.add_argument('--destdir', type=str, help='target data path')
    parser.add_argument('--factors', type=int, default=[], help='resize factor for target image')
    parser.add_argument('--resolutions', nargs='+', type=int, default=[], help='resolution for target image')
    parser.add_argument('--extend', type=str, help='extend for images')
    args = parser.parse_args()

    args.datadir = "../gendata/objects_centercrop/079_polaroid_film"
    args.destdir = "../gendata/objects_downsize"
    args.factors = []
    args.resolutions = [256, 256]
    args.extend = "png"

    if len(args.factors) == 0 and len(args.resolutions) == 0:
        print("ERROR: Either 'factors' or 'resolutions' should be given. Aborting.")
        sys.exit()
    else:
        return args


if __name__ == '__main__':
    args = get_args()
    minify(args.datadir, args.destdir, args.factors, [args.resolutions], args.extend)
