import numpy as np
import nibabel as nib
import os
import libcc

shape_imports = libcc.shapeSignImports()


def segment(subject_path, segmentation_method, segmentation_methods_dict, parcellation_methods_dict, basename):

	name_dict = {'ROQS': 'roqs',
	             'Watershed': 'watershed'}

	folderpath = subject_path + 'CCLab/'
	filename = 'segm_' + name_dict[segmentation_method] + '_data.npy'


	# Check if segmentation has already been done
	if os.path.exists(folderpath + filename):

		# Load files
		data_tuple = np.load(folderpath+filename, allow_pickle=True)

	# If there is no data available, segment
	else:

		# Read data, get scalar maps and eigs.
		wFA_v, FA_v, MD_v, RD_v, AD_v, fissure, eigvals, eigvects, affine = libcc.run_analysis(subject_path, basename)
		
		wFA = wFA_v[fissure,:,:]
		FA = FA_v[fissure,:,:]
		MD = MD_v[fissure,:,:]
		RD = RD_v[fissure,:,:]
		AD = AD_v[fissure,:,:]
		eigvects_ms = abs(eigvects[0,:,fissure]) 

		if segmentation_method == 'STAPLE':
			segmentation = libcc.segm_staple()
		else:	
			segmentation = segmentation_methods_dict[segmentation_method](wFA, eigvects_ms)


		# Check segmentation errors (True/False)
		error_flag = False
		error_prob = None
		try:
			error_flag, error_prob = libcc.checkShapeSign(segmentation, shape_imports, threshold=0.6)
		except:
			error_flag = True

			
	   	# Get data (meanFA, stdFA, meanMD, stdMD, meanRD, stdRD, meanAD, stdAD)
		scalar_maps = (wFA, FA, MD, RD, AD)
		scalar_statistics = libcc.getScalars(segmentation, FA, MD, RD, AD)
		try:
			midline_fa = libcc.getFAmidline(segmentation, FA, n_points=200)
		except:
			midline_fa = None
		
		# Parcellation
		parcellations_dict = {}
		for parcellation_method, parcellation_function in parcellation_methods_dict.items():

			try:
				parcellations_dict[parcellation_method] = parcellation_function(segmentation)
			except:
				parcellations_dict[parcellation_method] = []

		# Save files
		data_tuple = (segmentation, scalar_maps, scalar_statistics, error_prob)

		# Assemble nifti mask
		if segmentation_method != 'S_MASK':
			canvas = np.zeros(wFA_v.shape, dtype = 'int32')
			canvas[fissure,:,:] = segmentation
			save_nii(subject_path, filename, canvas, affine)

		save_os(subject_path, filename, data_tuple)

	return data_tuple


def save_os(path, filename, content):

	save_path = path + 'CCLab'

	# Create folder
	if not os.path.exists(save_path):
		os.mkdir(save_path)

	# Filename
	file_path = save_path + '/' + filename

	# Save file
	np.save(file_path, content)

def save_nii(path, filename, content, affine):

	save_path = path + 'CCLab'

	# Create folder
	if not os.path.exists(save_path):
		os.mkdir(save_path)

	# Filename
	nii_img = nib.Nifti1Image(content, affine)
	nib.save(nii_img, save_path+'/'+filename+'.nii.gz') 
