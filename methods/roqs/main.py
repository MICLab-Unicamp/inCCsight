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

def is_subject_folder(path):
    """Return True if path directly contains DTI eigenvalue files."""
    for ext in ('.nii.gz', '.nii'):
        if os.path.isfile(os.path.join(path, f'dti_L1{ext}')):
            return True
    return False

all_subjects = []

for folder in folder_mri:
    if is_subject_folder(folder):
        # The folder itself is a single subject
        all_subjects.append(folder)
    else:
        # The folder is a parent containing subject subfolders
        for subject in glob.glob(os.path.join(folder, "*")):
            if os.path.isdir(subject):
                all_subjects.append(subject)

sg.get_segm(all_subjects)