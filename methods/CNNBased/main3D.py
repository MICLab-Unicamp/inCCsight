from unet_module import LightningMRICCv2
from predict3D import test_predict
import glob
import os
import time
from script import rename_files
import argparse
import warnings
warnings.filterwarnings('ignore') 

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--parent', nargs='*', dest='parents')

args = parser.parse_args()

path_Data = ""

pre_trained_model_path = os.path.join(path_Data, "peso/3DExperimentV2_ManualMask_FAepoch=362-val_loss=0.13.ckpt")

model = LightningMRICCv2.load_from_checkpoint(pre_trained_model_path).eval().cpu()

folder_mri = args.parents

for folder in folder_mri:
    rename_files(folder)

all_subjects = []

for folder in folder_mri:
    subjects = glob.glob(os.path.join(folder, "*"))
    for subject in subjects:
#        if not os.path.exists(os.path.join(subject, "cnnBased.nii.gz")) and os.path.exists(os.path.join(subject, "cnnBased_midsagittal.nii.gz")) and os.path.exists(os.path.join(subject, "cnnBased_FA_V2.nii.gz")):
        all_subjects.append(subject)


start_time = time.time()
vol_data, test_outputs, pos_process, vol_data_affine = test_predict(model, all_subjects)

end_time = time.time()

time = end_time - start_time

print(f"Tempo processado Total (CNN_based): {time:.2f} segundos.")
    