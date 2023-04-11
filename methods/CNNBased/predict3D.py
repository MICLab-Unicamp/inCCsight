import torch
from postprocessed import get_post_processed_cc3d
from monai.inferers import sliding_window_inference
import nibabel as nib
import numpy as np
import os
import dipy
import dipy.io.peaks
import get_midsagittal as gm
import gets 
import pandas as pd
import time

def adjust_dict_parcellations_statistics(data):
    methods = list(data.keys())
    subjects = list(data[methods[0]].keys())
    methods_parc = list(data[methods[0]][subjects[0]])
    parts = list(data[methods[0]][subjects[0]][methods_parc[0]])
    scalars = list(data[methods[0]][subjects[0]][methods_parc[0]][parts[0]])
    
    for method in methods:
        subject_list = []
        for subject in subjects:
            subject_data = {"Name": subject}
            for method_p in methods_parc:
                for part in parts:
                    for scalar in scalars:
                        subject_data[f"{method_p}_{scalar}_{part}"] = data[method][subject][method_p][part][scalar]
            subject_list.append(subject_data)
        df = pd.DataFrame(subject_list)
        df.to_csv(f"./data/parcellation_statistics.csv", sep=";")

def test_predict(model, data_paths):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	vol_file = "iso_dti_FA_norm.nii.gz"

	with torch.no_grad():

		# Variables
		names = []
		meanFAList = []
		stdFAList = []
		meanMDList = []
		stdMDList = []
		meanRDList = [] 
		stdRDList = []
		meanADList = []
		stdADList = []
		parcellationsList = {"CNN": {}}
		times = []

		
		for data_path in data_paths:
			try:
				start = time.time()
				code = os.path.basename(data_path)
				sub = f'Subject_{code}'

				print(f"executando sujeito {data_path}", flush=True)

				vol_path = os.path.join(data_path, vol_file)
				vol_data = nib.load(vol_path).get_fdata().astype(np.float32)
				vol_data = (vol_data - vol_data.min()) / (vol_data.max() - vol_data.min())
				vol_data_affine = nib.load(vol_path).affine

				vol_data = torch.from_numpy(vol_data).unsqueeze(0).unsqueeze(0)

				test_outputs = sliding_window_inference(
					vol_data,
					roi_size=(112, 160, 56),
					sw_batch_size=1, 
					predictor=model,
					overlap=0.1,
					mode="gaussian",
					device=torch.device(device)
				)

				vol_data = vol_data.cpu()

				test_outputs = test_outputs.cpu().squeeze().detach()

				test_outputs = (test_outputs > 0.9).float()

				pos_process = get_post_processed_cc3d(test_outputs)
				pos_process_save = pos_process.numpy()

				dipy.io.peaks.save_nifti(os.path.join(data_path, "inCCsight/cnnBased.nii.gz"), pos_process_save, vol_data_affine, hdr = None )

				# 2D

				wFA_v, FA_v, MD_v, RD_v, AD_v, fissure, T3, output_mask, FA_mean, FAmeanJo, FAmeanJo_min = gm.run_analysis(data_path)

				wFA = wFA_v[fissure,:,:]
				FA = FA_v[fissure,:,:]
				MD = MD_v[fissure,:,:]
				RD = RD_v[fissure,:,:]
				AD = AD_v[fissure,:,:]

				scalar_maps = (FA, MD, RD, AD)

				midsagittal = output_mask[fissure, :, :]

				# values = gm.getParcellation(midsagittal, FA)
				# parcellation_dict = gm.parcellations_dfs_dicts(scalar_maps, values)
				# parcellationsList["CNN"][sub] = parcellation_dict

				midvolume = np.zeros(FA_v.shape)

				midvolume[fissure, :, :] = output_mask[fissure, :, :]

				dipy.io.peaks.save_nifti(os.path.join(data_path, "inCCsight/cnnBased_midsagittal.nii.gz"), midvolume, T3, hdr = None)
				dipy.io.peaks.save_nifti(os.path.join(data_path, "inCCsight/cnnBased_FA_V2.nii.gz"), FA_v, T3, hdr = None )

				scalar_statistics = gets.getScalars(midsagittal, FA, MD, RD, AD)
				names.append(sub)
				meanFAList.append(scalar_statistics[0])
				stdFAList.append(scalar_statistics[1])
				meanMDList.append(scalar_statistics[2])
				stdMDList.append(scalar_statistics[3])
				meanRDList.append(scalar_statistics[4])
				stdRDList.append(scalar_statistics[5])
				meanADList.append(scalar_statistics[6])
				stdADList.append(scalar_statistics[7])

				end = time.time()
	
				time_total = round(end - start, 2)
	
				times.append(time_total)
			except:
				print(f"Erro com: {data_path}")
				continue

		subjects = {"Names": names, "FA": meanFAList, "FA StdDev": stdFAList, "MD": meanMDList, "MD StdDev": stdMDList, 
              		"RD": meanRDList, "RD StdDev": stdRDList, "AD": meanADList, "AD StdDev": stdADList, 
					"Time": times}
		
		# adjust_dict_parcellations_statistics(parcellationsList)
		df = pd.DataFrame(subjects)
		df.to_csv("./data/cnn_based.csv", sep=";")

	return vol_data, test_outputs, pos_process, vol_data_affine
