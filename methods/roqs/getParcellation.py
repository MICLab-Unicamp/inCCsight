import parcellation as pcl
import pandas as pd

def adjust_dict_parcellations_statistics(data, subject_data, data_path):
    methods = list(data.keys())
    subjects = list(data[methods[0]].keys())
    methods_parc = list(data[methods[0]][subjects[0]])
    parts = list(data[methods[0]][subjects[0]][methods_parc[0]])
    scalars = list(data[methods[0]][subjects[0]][methods_parc[0]][parts[0]])
    
    for method in methods:
        subject_list = []
        for subject in subjects:
            for method_p in methods_parc:
                for part in parts:
                    for scalar in scalars:
                        subject_data[f"{method_p}_{scalar}_{part}"] = data[method][subject][method_p][part][scalar]
            subject_list.append(subject_data)
        
        df_sub = pd.DataFrame(subject_list)
        df_sub.to_csv(f"{data_path}/inCCsight/roqs_based.csv", sep=";")
        #df.to_csv(f"./data/parcellation_roqs_statistics.csv", sep=";")

def getParcellation(segment, wFA):
    
    parc_witelson = pcl.parc_witelson(segment, wFA)
    parc_hofer = pcl.parc_hofer(segment, wFA)
    parc_chao = pcl.parc_chao(segment, wFA)
    parc_cover = pcl.parc_cover(segment, wFA)
    parc_freesurfer = pcl.parc_freesurfer(segment, wFA)
    data = {"Witelson": parc_witelson, "Hofer": parc_hofer, "Chao": parc_chao, "Cover": parc_cover, "Freesurfer": parc_freesurfer}
    return data

# Auxiliar
def getData(parcel, FA, MD, RD, AD):
    
    import numpy as np

    data = {}

    # Initialize
    for region in ['P1', 'P2', 'P3', 'P4', 'P5']:
        data[region] = {}

    # Parcel values
    for i in range(2,7):
        data['P'+str(i-1)]['FA'] = np.mean(FA[parcel==i])
        data['P'+str(i-1)]['FA StdDev'] = np.std(FA[parcel==i])

        data['P'+str(i-1)]['MD'] = np.mean(MD[parcel==i])
        data['P'+str(i-1)]['MD StdDev'] = np.std(MD[parcel==i])

        data['P'+str(i-1)]['RD'] = np.mean(RD[parcel==i])
        data['P'+str(i-1)]['RD StdDev'] = np.std(RD[parcel==i])

        data['P'+str(i-1)]['AD'] = np.mean(AD[parcel==i])
        data['P'+str(i-1)]['AD StdDev'] = np.std(AD[parcel==i])

    return data

# Gerar os valores de parcelamento
def parcellations_dfs_dicts(scalar_maps, values):

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
            
        FA, MD, RD, AD = scalar_maps

        # Get dictionary
        data = getData(values[method], FA, MD, RD, AD)    
        for region in list_regions:
            for scalar in list_scalars:
                
                parcel_dict[method][region][scalar] = data[region][scalar]
            
    return parcel_dict