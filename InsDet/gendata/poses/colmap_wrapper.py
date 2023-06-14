import os
import subprocess


# $ DATASET_PATH=/path/to/dataset

# $ colmap feature_extractor \
#    --database_path $DATASET_PATH/database.db \
#    --image_path $DATASET_PATH/images

# $ colmap exhaustive_matcher \
#    --database_path $DATASET_PATH/database.db

# $ mkdir $DATASET_PATH/sparse

# $ colmap mapper \
#     --database_path $DATASET_PATH/database.db \
#     --image_path $DATASET_PATH/images \
#     --output_path $DATASET_PATH/sparse

# $ mkdir $DATASET_PATH/dense
def run_colmap(basedir, match_type):
    posedir = os.path.join(basedir, 'poses')
    if not os.path.exists(posedir):
        os.makedirs(posedir)
    logfile_name = os.path.join(posedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')

    feature_extractor_args = [
        'colmap', 'feature_extractor',
        '--database_path', os.path.join(posedir, 'database.db'),
        '--image_path', os.path.join(basedir, 'images'),
        '--ImageReader.mask_path', os.path.join(basedir, 'masks'),
        '--ImageReader.single_camera', '1',
        '--ImageReader.camera_model', 'SIMPLE_RADIAL',
        '--SiftExtraction.use_gpu', '1',
        '--SiftExtraction.max_image_size', '3840',
    ]
    feat_output = (subprocess.check_output(feature_extractor_args, universal_newlines=True))
    logfile.write(feat_output)
    print('Features extracted')

    matcher_args = [
        'colmap', match_type,
        '--database_path', os.path.join(posedir, 'database.db'),
    ]

    match_output = (subprocess.check_output(matcher_args, universal_newlines=True))
    logfile.write(match_output)
    print('Features matched')

    p = os.path.join(posedir, 'sparse')
    if not os.path.exists(p):
        os.makedirs(p)

    # mapper_args = [
    #     'colmap', 'mapper',
    #         '--database_path', os.path.join(posedir, 'database.db'),
    #         '--image_path', os.path.join(basedir, 'images'),
    #         '--output_path', os.path.join(posedir, 'sparse'),
    #         '--Mapper.num_threads', '16',
    #         '--Mapper.init_min_tri_angle', '4',
    # ]
    mapper_args = [
        'colmap', 'mapper',
        '--database_path', os.path.join(posedir, 'database.db'),
        '--image_path', os.path.join(basedir, 'images'),
        '--output_path', os.path.join(posedir, 'sparse'),  # --export_path changed to --output_path in colmap 3.6
        '--Mapper.num_threads', '16',
        '--Mapper.init_min_tri_angle', '4',
        '--Mapper.multiple_models', '0',
        '--Mapper.extract_colors', '0',
    ]

    map_output = (subprocess.check_output(mapper_args, universal_newlines=True))
    logfile.write(map_output)
    logfile.close()
    print('Sparse map created')

    print('Finished running COLMAP, see {} for logs'.format(logfile_name))