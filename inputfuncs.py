import argparse
import os
import glob

def get_parser():
	parser = argparse.ArgumentParser(
	    description = 'This is a program for data processing and visualization of corpus callosum diffusion tensor images. '+
	                  ' In order to process the DTI data this software uses eigenvalues and eigenvectors in FSL format:'+
	                  '(dti_L1.nii.gz, dti_L2.nii.gz, dti_L3.nii.gz, dti_V1.nii.gz, dti_V2.nii.gz, dti_V3.nii.gz.' +
	                  '\n\n - You may import these files in two modes:'+
	                  '\n   [--directory]  Inform a list of directories which contain the FSL files;'+
	                  '\n   [--parent]     Inform the path for a single directory with subdirectories which contain the FSL files;'+
	                  '\n\n - You may also import your own segmentation masks:'+
	                  '\n   [--mask]       Inform a string contained in filename of the masks file (expected: .nii files)', formatter_class=argparse.RawTextHelpFormatter)
	                

	parser.add_argument('-d', '--directory', help='directory or list of directories that contain the eigenvectors and eigenvalues files.', nargs='*', dest='dirs')
	parser.add_argument('-p', '--parent', help='directory or list of directories that contain subdirectories that contain the eigenvectors and eigenvalues files. Use this to easily import lots of data.', nargs='*', dest='parents')
	parser.add_argument('-b', '--basename', help='basename for the FSL files (basename default is \'dti\', e.g. dti_L1.nii.gz', nargs='?', default='dti', dest='basename')
	parser.add_argument('-m', '--mask', help='string contained in the filename of the masks files located in the directories loaded using [-d] or [-p]', nargs='?', type=str, dest='mark_str')

	return parser
'''
# Arg parser
opts = get_parser().parse_args()

# Get paths and import data
for value in opts.dirs:
    if value is not None:
        print(opts.basename)

'''

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


def import_cclab(path):

	cclab_path = path + 'CCLab/'
	
	if(os.path.exists(cclab_path)):
		print('import here')