# coding: utf-8 
 
def segm_watershed(wFA_ms, gaussian_sigma=0.3):

    import numpy as np
    from scipy import ndimage 

    import siamxt
    from libcc.preprocess import run_analysis, grad_morf
    from libcc.gets import getTheCC
    from skimage.morphology import watershed, disk, square, erosion, dilation
    from skimage.measure import label, regionprops	


    ## MORPHOLOGICAL GRADIENT

    # Gaussian filter
    wFA_gauss = ndimage.gaussian_filter(wFA_ms, sigma=gaussian_sigma)

    # Structuring element
    se1 = np.zeros((3,3)).astype('bool')
    se1[1,:] = True
    se1[:,1] = True

    # Gradient
    grad_wFA = grad_morf(wFA_gauss, se1)



    ## MAX-TREE

    #Structuring element. connectivity-4
    se2 = se1.copy()

    # Computing Max Tree by volume
    mxt = siamxt.MaxTreeAlpha(((grad_wFA)*255).astype("uint8"), se2)
    attr = "volume"
    leaves_volume = mxt.computeExtinctionValues(mxt.computeVolume(),attr)

    # Create canvas
    segm_markers = np.zeros(grad_wFA.shape, dtype = np.int16)

    # Labeling canvas
    indexes = np.argsort(leaves_volume)[::-1]
    counter = 1
    for i in indexes[:85]:
        segm_markers = segm_markers + mxt.recConnectedComponent(i)*(counter)
        counter+=1
           


    ## SEGMENTING CC

    # Watershed    
    wc_wfa = watershed(grad_wFA, segm_markers)
        
    # Thresholding regions by FA
    seg_wFA = np.zeros((wFA_ms).shape).astype(bool)
    segs = seg_wFA
    listAll = np.unique(wc_wfa)
    for i in listAll:
        media = np.mean(wFA_ms[wc_wfa == i])
        if media > 0.2*wFA_ms.max():
            seg_wFA[wc_wfa == i] = 1

    # Getting the CC
    seg_wFA, ymed, xmed = getTheCC(seg_wFA)

    return seg_wFA, ymed, xmed


def segm_roqs(wFA_ms, eigvects_ms):
    
    import numpy as np 
    from scipy.ndimage.morphology import binary_fill_holes, binary_closing, binary_erosion, binary_opening
    from skimage.measure import label   
    from skimage import measure
    
    # Seed grid search - get highest FA seed within central area
    h,w = wFA_ms.shape
                
    # Define region to make search
    region = np.zeros((h,w))
    region[int(h/3):int(2*h/3), int(w/2):int(2*w/3)] = 1
    region = wFA_ms * region

    # Get the indices of maximum element in numpy array
    fa_seed = np.amax(region)
    seedx, seedy = np.where(region == fa_seed)
    
    # Defining seeds positions
    seed = [seedx, seedy]

    # Get principal eigenvector (direction of maximal diffusivity)
    max_comp_in = np.argmax(eigvects_ms[:,seed[0],seed[1]],axis=0)
    max_comp_in = np.argmax(np.bincount(max_comp_in.ravel()))
    
    # Max component value
    Cmax_seed = eigvects_ms[max_comp_in,seed[0],seed[1]]

    # First selection criterion 
    # Get pixels with the same maximum component (x,y or z) of the principal eigenvector
    princ = np.argmax(eigvects_ms,axis=0)
    fsc = princ == max_comp_in

    # Calculate magnification array (MA)
    alpha = 0.3
    beta = 0.3
    gamma = 0.5
    MA = (wFA_ms-np.amax(wFA_ms)*alpha)/(np.amax(wFA_ms)*beta)+gamma
    
    # Apply MA to eigenvector
    ssc = np.clip(np.amax(eigvects_ms*MA, axis=0), 0, 1)
    ssc = ssc*fsc
    
    # Keep only pixels with Cmax greater than Cmax_seed-0.1
    mask_cc = ssc > Cmax_seed-0.1
    labels = label(mask_cc)
    mask_cc = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    segm = binary_fill_holes(mask_cc)
    
    # Post processing
    contours = measure.find_contours(segm, 0.1)
    contour = sorted(contours, key=lambda x: len(x))[-1] 
        
    return segm, contour


def segm_staple(path, fissure, segm_import=None):
    
    import tempfile
    import numpy as np
    import nibabel as nib
    import SimpleITK as sitk

    segm_water = path+'CCLab/segm_watershed.nii.gz'
    segm_roqs = path+'CCLab/segm_roqs.nii.gz'

    staple = sitk.STAPLEImageFilter()
    writer = sitk.ImageFileWriter()
    reader = sitk.ImageFileReader()
    
    reader.SetFileName(segm_water)
    nii_water = reader.Execute()
    reader.SetFileName(segm_roqs)
    nii_roqs = reader.Execute()
    
    if segm_import == None:
        nii_staple = staple.Execute(nii_water, nii_roqs)
    else:
        reader.SetFileName(segm_import)
        nii_import = reader.Execute()   
        nii_staple = staple.Execute(nii_water, nii_roqs, nii_import)

    writer.SetFileName('{}/segm_staple.nii.gz'.format(path))
    writer.Execute(nii_staple)

    nii_staple = nib.load('{}/segm_staple.nii.gz'.format(path)).get_data()
    seg_staple = nii_staple[fissure,:,:]

    seg_staple[seg_staple == np.min(seg_staple)] = 0
    seg_staple[seg_staple > 0] = 1
    seg_staple = np.array(seg_staple, dtype='int32')

    return seg_staple

def segm_mask(path, threshold=0):

    import nibabel as nib
    import numpy as np
    import matplotlib.pyplot as plt

    volume = nib.load(path).get_data()

    # Normalize
    vol = np.array(volume, dtype='int32')
    vol[vol > threshold] == 1

    # Find direction 
    mask = None
    fissure = None
    for i in range(3):
        for j in range(2):
            if np.sum(np.max(np.max(vol,axis=i),axis=j)) == 1:
                fissure = np.argmax(np.max(np.max(vol,axis=i),axis=j))
      
    axis = 0      
    # Obtain mask from direction    
    if (i==1 and j==1) or (i==2 and j==1):
        mask = vol[fissure,:,:]
        axis = 0
    elif (i==0 and j==1) or (i==2 and j==0):
        mask = vol[:,fissure,:]
        axis = 1
    elif (i==0 and j==0) or (i==1 and j==0):
        mask = vol[:,:,fissure]
        axis = 2

    return mask, fissure, axis