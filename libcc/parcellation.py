# coding: utf-8

def parc_watershed(segm, FA_map, n_parcel, group_offset):

    '''
    Performs the watershed parcellation of the Corpus Callosum as described
    in the article https://ieeexplore.ieee.org/abstract/document/8065035/

    Inputs:
        - segm: 2D numpy array of the segmented midsaggital slice
        - FA_map: 2D numpy array of the weighted FA map midsagittal slice
        - n_parcel: int number of parcels to divide the structure
        - group_offset: number of points close to the borders of decision
                        to not be used as seeds in the Watershed transform 
    Outputs:
        - parcel: 2D numpy array (int) with same dimentions as segmentation 
                  input labeled with performe parcels
                  
    '''  

    from libcc import points, getGroupPoints, grad_morf
    from sklearn import cluster
    from skimage.morphology import disk, watershed, dilation
    from scipy import ndimage
    import numpy as np
    from skimage.measure import label

    ## KMEANS
    
    # Get CC's midline
    px, py = points(segm, 200)
        
    # Assemble vector [[x1,y1,FA1], [x2,y2,FA2] ... ]
    kpoints = []
    kfa =[]
    for aux in range(0, 199):
        x = int(round(px[aux]))
        y = int(round(py[aux]))
        try:
            fa = FA_map[y,x]
        except:
            x = int(np.floor(px[aux]))
            y = int(np.floor(py[aux]))
        kpoints.append([x,y,fa])
      
    # kMeans
    n_clusters = n_parcel
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=43).fit_predict(kpoints)
    
    # Get group of central points for each cluster
    offset = group_offset
    kcenters = getGroupPoints(kmeans, n_clusters, offset, kpoints)
    
    
    ## WATERSHED
    
    # Build initial canvas 
    parcel_markers = np.zeros(FA_map.shape, dtype = np.int32)
    
    # External markers (negative cc segmentation) 
    parcel_markers[segm == False] = n_clusters+1
    
    # Internal markers (from kmeans)
    for kc in kcenters:
              
        idx = kc[2]
        idy = kc[1]
        parcel_markers[idx, idy] = kc[0]+1    
    
    ## MORPHOLOGICAL GRADIENT

    # Gaussian filter
    wFA_gauss = ndimage.gaussian_filter(FA_map, sigma=0.3)
    
    # Structuring element
    se1 = np.zeros((3,3)).astype('bool')
    se1[1,:] = True
    se1[:,1] = True

    # Gradient
    grad_wFA = grad_morf(wFA_gauss, se1)

    # Watershed  
    parcel = watershed(grad_wFA, parcel_markers)
    parcel = np.rot90(label(np.rot90(parcel, 3)), 1);
    return parcel 

def parc_geometric_cc (segmentation, scheme = 'HOFER', eps=1**-5):

    '''
    Performs the geometric parcellation of the Corpus Callosum accordingly
    with the selected parcellation scheme.

    Inputs:
        - segmentation: 2D numpy array of the segmented midsaggital slice
        - scheme: string with the name of the parcellation scheme that can be:
                  'HOFER', 'WITELSON', 'CHAO' or 'FREESURFER'
        - eps: float used for mitigating problems with zero division. suggested
               use it to put the function in a try/except structure and zero its
               value in case of error
    Outputs:
        - Parcellation: 2D numpy array (int) with same dimentions as segmentation 
                        input labeled with performe parcels
                  
    '''
    
    import numpy as np 
    from skimage.measure import label

    segmentation = np.array(segmentation, dtype='int32')
    SCHEMES = ['HOFER', 'WITELSON', 'CHAO', 'FREESURFER']
    scheme = scheme.upper()
    if not scheme in SCHEMES:
        raise Exception('Unknown scheme!')

    def coef_linear (a, p):
        return p[0]-a*p[1] 

    def predicty(x, a, b):
        return a*x + b

    def predictx(y, a, b):
        return (y-b)/a

    # Vetor da base e vetor normal a base 
    M,N = np.nonzero(segmentation)
    minN = np.min(N)
    maxN = np.max(N) 
    minM = segmentation[:,minN].nonzero()[0].mean() + eps
    maxM = segmentation[:,maxN].nonzero()[0].mean() + 1**-7
    p1 = np.array([minM, minN])
    p2 = np.array([maxM, maxN])

    base_v = p2 - p1
    base_length = np.sqrt((base_v**2).sum())
    base_v = base_v / np.sqrt((base_v**2).sum())
    cut_v = np.array([-base_v[1], base_v[0]]) 

    # Coeficientes das retas
    hofer = np.array([1.0/6, 1.0/2, 2.0/3, 3.0/4]).reshape(4,1)
    witelson = np.array([1.0/3, 1.0/2, 2.0/3, 4.0/5]).reshape(4,1)
    chao = np.array([1.0/3, 2.0/3, 3.0/4, 5.0/6]).reshape(4,1)
    freesurfer = np.array([1.0/5, 2.0/5, 3.0/5, 4.0/5]).reshape(4,1)

    if scheme == 'HOFER':
        P = p1 + hofer*base_length*base_v

    if scheme == 'WITELSON':
        P = p1 + witelson*base_length*base_v
        
    if scheme == 'CHAO':
        P = p1 + chao*base_length*base_v

    if scheme == 'FREESURFER':
        P = p1 + freesurfer*base_length*base_v

    p3, p4, p5, p6 = P

    rbase_A = base_v[0]/base_v[1]
    rbase_B = p1[0]-rbase_A*p1[1]
    rA = cut_v[0]/cut_v[1]
    r3B = coef_linear(rA, p3)
    r4B = coef_linear(rA, p4)
    r5B = coef_linear(rA, p5)
    r6B = coef_linear(rA, p6)

    # Rotulação da máscara    
    H,W = np.shape(segmentation)
    Parcellation = np.ones((H,W), dtype='int')

    y,x = segmentation.nonzero()
    labels = np.zeros(y.size, dtype='int')
    above_base = (y <= predicty(x, rbase_A, rbase_B))
    left_r3 = (x <= predictx(y, rA, r3B))
    left_r4 = (x <= predictx(y, rA, r4B))
    left_r5 = (x <= predictx(y, rA, r5B))
    left_r6 = (x <= predictx(y, rA, r6B))


    labels[np.logical_and(left_r3==False, left_r4)] = 3
    labels[np.logical_and(left_r4==False, left_r5)] = 4
    labels[np.logical_and(left_r5==False, left_r6)] = 5
    labels[np.logical_or(np.logical_and(above_base==False, left_r4), left_r3)] = 2
    labels[np.logical_or(np.logical_and(above_base==False, left_r5==False), left_r6==False)] = 6

    Parcellation[y,x] = labels
    Parcellation = np.rot90(label(np.rot90(Parcellation, 3)), 1);
    return Parcellation

