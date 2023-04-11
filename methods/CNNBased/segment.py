import libcc
import numpy

def segment(subject_path, segmentation_method, segmentation_methods_dict, parcellation_methods_dict, basename, mask_basename=None):

	name_dict = {'ROQS': 'roqs',
	             'Watershed': 'watershed',
	             'STAPLE':'staple',
	             'Imported Masks':'imported_mask'}

	folderpath = subject_path + 'inCCsight/'
	segmname = 'segm_' + name_dict[segmentation_method]
	filename = segmname + '_data.npy'

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

		# Check segmentation type and segment
		try:
			if segmentation_method == 'STAPLE':
				segmentation = segmentation_methods_dict[segmentation_method](subject_path, fissure, segm_import=None)
			elif segmentation_method == 'Imported Masks':
				segmentation, fissure, axis = segmentation_methods_dict[segmentation_method](get_mask_path(subject_path, mask_basename), threshold=0)
				if segmentation is None:
					raise TypeError()

				if axis == 0 :
					wFA = wFA_v[fissure,:,:]
					FA = FA_v[fissure,:,:]
					MD = MD_v[fissure,:,:]
					RD = RD_v[fissure,:,:]
					AD = AD_v[fissure,:,:]
				if axis == 1 :
					wFA = wFA_v[:,fissure,:]
					FA = FA_v[fissure,:,:]
					MD = MD_v[:,fissure,:]
					RD = RD_v[:,fissure,:]
					AD = AD_v[:,fissure,:]
				if axis == 2 :
					wFA = wFA_v[:,:,fissure]
					FA = FA_v[fissure,:,:]
					MD = MD_v[:,:,fissure]
					RD = RD_v[:,:,fissure]
					AD = AD_v[:,:,fissure]
			else:
				segmentation = segmentation_methods_dict[segmentation_method](wFA, eigvects_ms)
		except:
			print('> Segmentation failed for subject {} with segmentation method {}'.format(os.path.basename(os.path.dirname(subject_path)), segmentation_method))
			return None

		if len(np.array(segmentation).shape) < 2:
			print('> Segmentation failed for subject {} with segmentation method {}'.format(os.path.basename(os.path.dirname(subject_path)), segmentation_method))
			return None

		# Check segmentation errors (True/False)
		error_flag = False
		error_prob = []
		try:
			error_flag, error_prob = libcc.checkShapeSign(segmentation, shape_imports, threshold=0.6)
		except:
			error_flag = True

			
	   	# Get data (meanFA, stdFA, meanMD, stdMD, meanRD, stdRD, meanAD, stdAD)
		scalar_maps = (wFA, FA, MD, RD, AD)
		scalar_statistics = libcc.getScalars(segmentation, FA, MD, RD, AD)
		scalar_midlines = {}
		try:
			scalar_midlines['FA'] = libcc.getFAmidline(segmentation, FA, n_points=200)
			scalar_midlines['MD'] = libcc.getFAmidline(segmentation, MD, n_points=200)
			scalar_midlines['RD'] = libcc.getFAmidline(segmentation, RD, n_points=200)
			scalar_midlines['AD'] = libcc.getFAmidline(segmentation, AD, n_points=200)
		except:
			scalar_midlines = {'FA':[],'MD':[],'RD':[],'AD':[]}
		
		# Parcellation
		parcellations_dict = {}
		for parcellation_method, parcellation_function in parcellation_methods_dict.items():
			try:
				parcellations_dict[parcellation_method] = parcellation_function(segmentation, wFA)
			except:
				print("Parc. Error - Method: {}, Subj.: {}".format(parcellation_method, subject_path))
				parcellations_dict[parcellation_method] = []

		# Save files
		data_tuple = (segmentation, scalar_maps, scalar_statistics, scalar_midlines, error_prob, parcellations_dict)

		# Assemble nifti mask
		if segmentation_method != 'Imported Masks':
			canvas = np.zeros(wFA_v.shape, dtype = 'int32')
			canvas[fissure,:,:] = segmentation
			save_nii(subject_path, segmname, canvas, affine)

		save_os(subject_path, filename, data_tuple)
		
		del wFA_v, FA_v, MD_v, RD_v, AD_v, fissure, eigvals, eigvects, affine
		del wFA, FA, MD, RD, AD
		del segmentation, scalar_maps, scalar_statistics, scalar_midlines, error_prob, parcellations_dict

	return data_tuple