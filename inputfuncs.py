import argparse
import os
import glob
import numpy as np
import libcc
import pandas as pd
from pathlib import Path

def get_parser():
    
    parser = argparse.ArgumentParser(
        description = 'This is a program for data processing and visualization of corpus callosum diffusion tensor images. '+
                      ' In order to process the DTI data this software uses eigenvalues and eigenvectors in FSL format:'+
                      '(dti_L1.nii.gz, dti_L2.nii.gz, dti_L3.nii.gz, dti_V1.nii.gz, dti_V2.nii.gz, dti_V3.nii.gz.', formatter_class=argparse.RawTextHelpFormatter)
                    

    parser.add_argument('-f', '--folders', help='Directory(ies) that contain the eigenvectors and values files.', nargs='*', dest='folders')
    parser.add_argument('-p', '--parent', help='Directory(ies)that contain subdirectories with the eigenvectors and values files. Use this to easily import lots of data.', nargs='*', dest='parents')
    
    parser.add_argument('-d', '--extra-data', help='xlsm or csv file with extra data to be imported, must contain a column named Subjects with the same name as the folders that contain the eigenvectors', nargs='?', default=None, dest='ext_data')
    parser.add_argument('--port', help='Changes the localhost port the software will be available at. Default is 8000.', nargs='?', type=int, dest='port', default=8000)

    parser.add_argument('-b', '--basename', help='Basename for the FSL files (basename default is \'dti\', e.g. dti_L1.nii.gz', nargs='?', default='dti', dest='basename')

    parser.add_argument('-m', '--maskname', help='String contained in the filename of the masks files located in the directories loaded using [-d] or [-p]', nargs='?', type=str, dest='maskname')
    parser.add_argument('-s', '--segm', help='Segmentation methods to be performed (ROQS, Watershed), default is both', nargs='+', dest='segm', default=['ROQS', 'Watershed'])
    parser.add_argument('--staple', help='Will create a segmentation consensus between the methods selected and the mask (if inputted). Only possible with multiple segmentations', action='store_true')
    
    parser.add_argument('-3d', help='3D Segmentation methods to be performed (Watershed3d), default is None', dest='segm3d', default=[None])

    return parser


def check_directory(path, basename):

    if (os.path.exists(os.path.join(path, '{}_L1.nii.gz'.format(basename))) and
        os.path.exists(os.path.join(path, '{}_L2.nii.gz'.format(basename))) and
        os.path.exists(os.path.join(path, '{}_L3.nii.gz'.format(basename))) and
        os.path.exists(os.path.join(path, '{}_V1.nii.gz'.format(basename))) and
        os.path.exists(os.path.join(path, '{}_V2.nii.gz'.format(basename))) and
        os.path.exists(os.path.join(path, '{}_V3.nii.gz'.format(basename)))):
        
        return True
    else:
        print("Warning: Directory {} did not contain all files required to perform the segmentation".format(path))
        return False


def import_parent(parent_path, basename):

    directory_dict = {}
    group_dict = {}
    
    if (os.path.exists(parent_path)):

        dirs = [directory for directory in glob.glob(parent_path+'/**/*/', recursive=True) if os.path.basename(os.path.dirname(directory)) != 'inCCsight']
        dirs.sort()        

        for directory_path in dirs:
            if check_directory(directory_path, basename):
                subject_name = os.path.basename(os.path.dirname(directory_path))
                directory_dict[subject_name] = directory_path
                group_dict[subject_name] = os.path.basename(Path(directory_path).parent)

    return directory_dict, group_dict


def parcellations_dfs_dicts(scalar_maps_dict, parcellations_dict, segmentation_method):

    list_methods = ['Witelson', 'Hofer', 'Chao', 'Cover', 'Freesurfer']
    list_regions = ['P1', 'P2', 'P3', 'P4', 'P5']
    list_scalars = ['FA', 'FA StdDev', 'MD', 'MD StdDev', 'RD', 'RD StdDev', 'AD', 'AD StdDev']


    parcel_dict = {}
    for method in list_methods:
        parcel_dict[method] = {}

        for region in list_regions:
            parcel_dict[method][region] = {}

            for scalar in list_scalars:
                parcel_dict[method][region][scalar] = {}

        for key in scalar_maps_dict.keys():

            _, FA, MD, RD, AD = scalar_maps_dict[key]

            # Get dictionary
            data = libcc.getData(parcellations_dict[key][method], FA, MD, RD, AD)    
            
            for region in list_regions:
                for scalar in list_scalars:
                    
                    parcel_dict[method][region][scalar][key] = data[region][scalar]


        for region in ['P1', 'P2', 'P3', 'P4', 'P5']:
            parcel_dict[method][region] = pd.DataFrame.from_dict(parcel_dict[method][region])

    return parcel_dict
