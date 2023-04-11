import argparse
import glob
import os
import segmentation as sg

import warnings
warnings.filterwarnings('ignore') 

# Read input path
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--parent', nargs='*', dest='parents')

args = parser.parse_args()
    
# Read files
folder_mri = args.parents

all_subjects = []

for folder in folder_mri:
    subjects = glob.glob(os.path.join(folder, "*"))

    for subject in subjects:
        all_subjects.append(subject)

sg.get_segm(all_subjects)