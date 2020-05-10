
import os
import pathlib
import pandas as pd
import numpy as np
from tqdm import tqdm

import inputfuncs
import ccprocess

import segmfuncs
import parcelfuncs

# GENERAL DEFINITIONS -------------------------------------------------------------------------

segmentation_methods_dict = {'ROQS': segmfuncs.segm_roqs,
                             'Watershed': segmfuncs.segm_roqs}

parcellation_methods_dict = {'Witelson': parcelfuncs.parc_witelson,
                             'Hofer': parcelfuncs.parc_hofer,
                             'Chao': parcelfuncs.parc_chao,
                             'Cover': parcelfuncs.parc_cover,
                             'Freesurfer': parcelfuncs.parc_freesurfer}

# DATA IMPORTING -----------------------------------------------------------------------------

# Arg parser
opts = inputfuncs.get_parser().parse_args()

# Get paths and check if files exist
path_dict = {}
if opts.dirs is not None:
    for directory in opts.dirs:
        if directory is not None:
            if inputfuncs.check_directory(directory, opts.basename):
                path_dict[directory.rsplit('/', 2)[1]] = directory

group_dict = {}
if opts.parents is not None:
    for parent in opts.parents:
        if parent is not None:
            
            directory_dict = inputfuncs.import_parent(parent, opts.basename)
            
            path_dict.update(directory_dict)
            group_dict[parent.rsplit('/', 2)[1]] = list(directory_dict.keys())


# DATA PROCESSING -----------------------------------------------------------------------------

for subject_key, subject_path in tqdm(path_dict.items()):

    for segmentation_method in segmentation_methods_dict.keys():
    
        ccprocess.segment(subject_path, segmentation_method, segmentation_methods_dict, opts.basename)
