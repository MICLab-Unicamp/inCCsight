import os
import glob

def rename_files(path):
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file == "dti_FA.nii.gz":
                file_path = os.path.join(subdir, file)
                new_file_path = os.path.join(subdir, "iso_dti_FA_norm.nii.gz")
                os.rename(file_path, new_file_path)
            elif file == "FA.nii.gz":
                file_path = os.path.join(subdir, file)
                new_file_path = os.path.join(subdir, "iso_dti_FA_norm.nii.gz")
                os.rename(file_path, new_file_path)

#rename_files("/home/jovi/Dados/Dados_2")