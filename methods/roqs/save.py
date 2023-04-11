import os
import numpy as np
import nibabel as nib


def save_os(path, filename, content):

	save_path = os.path.join(path, 'inCCsight')

	# Create folder
	if not os.path.exists(save_path):
		os.mkdir(save_path)

	# Filename
	file_path = os.path.join(save_path, filename)

	# Save file
	print(file_path)
	np.save(file_path, content)

def save_nii(path, filename, content, affine):
	
	save_path = os.path.join(path, 'inCCsight')

	# Create folder
	if not os.path.exists(save_path):
		os.mkdir(save_path)

	# Filename
	nii_img = nib.Nifti1Image(content, affine)
	nib.save(nii_img, os.path.join(save_path, filename+'.nii.gz'))