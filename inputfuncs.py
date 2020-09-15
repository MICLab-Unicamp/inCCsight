import argparse
import os
import glob
import numpy as np
import libcc
import pandas as pd

def get_parser():
	parser = argparse.ArgumentParser(
	    description = 'This is a program for data processing and visualization of corpus callosum diffusion tensor images. '+
	                  ' In order to process the DTI data this software uses eigenvalues and eigenvectors in FSL format:'+
	                  '(dti_L1.nii.gz, dti_L2.nii.gz, dti_L3.nii.gz, dti_V1.nii.gz, dti_V2.nii.gz, dti_V3.nii.gz.', formatter_class=argparse.RawTextHelpFormatter)
	                

	parser.add_argument('-d', '--directory', help='Directory(ies) that contain the eigenvectors and values files.', nargs='*', dest='dirs')
	parser.add_argument('-p', '--parent', help='Directory(ies)that contain subdirectories with the eigenvectors and values files. Use this to easily import lots of data.', nargs='*', dest='parents')
	parser.add_argument('-b', '--basename', help='Basename for the FSL files (basename default is \'dti\', e.g. dti_L1.nii.gz', nargs='?', default='dti', dest='basename')

	parser.add_argument('-m', '--mask', help='String contained in the filename of the masks files located in the directories loaded using [-d] or [-p]', nargs='?', type=str, dest='mark_str')
	parser.add_argument('-s', '--segm', help='Segmentation method to be performed (ROQS and/or Watershed), default is both', nargs='+', dest='segm', default=['ROQS', 'Watershed'])
	parser.add_argument('--staple', help='Will create a segmentation consensus between the methods selected and the mask (if inputted). Only possible with multiple segmentation')

	return parser


def check_directory(path, basename):

	if (os.path.exists(path + '{}_L1.nii.gz'.format(basename)) and
		os.path.exists(path + '{}_L2.nii.gz'.format(basename)) and
		os.path.exists(path + '{}_L3.nii.gz'.format(basename)) and
		os.path.exists(path + '{}_V1.nii.gz'.format(basename)) and
		os.path.exists(path + '{}_V2.nii.gz'.format(basename)) and
		os.path.exists(path + '{}_V3.nii.gz'.format(basename))):
		
		return True
	else:
		print("Warning: Directory {} did not contain all files required to be imported".format(path))
		return False


def import_parent(parent_path, basename):

	directory_dict = {}
	
	if (os.path.exists(parent_path)):

		dirs = [directory for directory in glob.glob(parent_path+'/*/')]
		dirs.sort()		

		for directory_path in dirs:
			if check_directory(directory_path, basename):
				directory_dict[directory_path.rsplit('/', 2)[1]] = directory_path

	return directory_dict


def parcellations_dfs_dicts(scalar_maps_dict, parcellations_dict):

    method_dict = {}
    for method in ['Witelson', 'Hofer', 'Chao', 'Cover', 'Freesurfer']:
        
        scalar_dict = {}
        for scalar in ['FA', 'MD', 'RD', 'AD']:
            scalar_dict[scalar] = dict()
            
        for key in scalar_maps_dict.keys():
            _, FA, MD, RD, AD = scalar_maps_dict[key]
            data = np.array(libcc.getData(parcellations_dict[key][method], FA, MD, RD, AD))
                
            scalar_dict['FA'][key] = data[:,0]
            scalar_dict['MD'][key] = data[:,1]
            scalar_dict['RD'][key] = data[:,2]
            scalar_dict['AD'][key] = data[:,3]

        for scalar in ['FA', 'MD', 'RD', 'AD']:
            scalar_dict[scalar] = pd.DataFrame.from_dict(scalar_dict[scalar], orient='index', columns = ['P1', 'P2', 'P3', 'P4', 'P5']).reset_index().round(6)

        method_dict[method] = scalar_dict
        
    return method_dict


