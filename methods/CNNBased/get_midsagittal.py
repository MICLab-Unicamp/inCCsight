#rootdir = "/home/joany/Teste_sipaim/000161/"
import parcellation as pcl
import pandas as pd

def align_sagittal_plane(T):

    import numpy as np
    
    # CANONICAL BASE (i,j,k) (HOMOGENEOUS COORDINATES)
    c = np.array(
        [[0,1,0,0],
         [0,0,1,0],
         [0,0,0,1],
         [1,1,1,1]])

    # FIND BASE V
    V_ = np.dot(T,c)
    V_ = V_[:3,1:] - V_[:3,0].reshape(3,1)

    V = np.zeros((3,3))
    V[np.arange(3),np.argmax(np.abs(V_), axis=1)] = 1
    V_[V_==0] = 1
    V = V * (V_/np.abs(V_))

    # DESIRED BASE W
    W = np.array([[1,0,0],
                  [0,0,-1],
                  [0,-1,0]])

    R = np.dot(np.linalg.inv(W), V)
    r = np.diag(np.ones(4))
    r[:3,:3] = R
    return r
    
def getFractionalAnisotropy(eigvals):
    
    import numpy as np
    np.seterr(divide='ignore', invalid='ignore')

    MD = eigvals.mean(axis=0)
    FA = np.sqrt(3*((eigvals-MD)**2).sum(axis=0)) / np.sqrt(2*(eigvals**2).sum(axis=0))

    RD = (eigvals[1]+eigvals[2])/2
    AD = eigvals[0]
    
    return (FA,MD,RD,AD)

def getFissureSlice(eigvals, FA):
    
    import numpy as np

    MASK = (eigvals[0]>0)
    MASKcount = MASK.sum(axis=2).sum(axis=1)
    FAmean = FA.mean(axis=2).mean(axis=1)
    FAmeanJo = FA.mean(axis=2).mean(axis=1)
    FAmeanJo[MASKcount<=0.90*MASKcount.max()]# = 0
    FAmeanJo_min = FAmeanJo.min()
    FAmean[MASKcount<=0.90*MASKcount.max()] = 1
    return (np.argmin(FAmean), FAmean, FAmeanJo, FAmeanJo_min)

def loadNiftiDTI(basename, reorient=False):
    
    import nibabel as nib
    import numpy as np
    
    # ====== MAIN FUNCTION START ===========================
    # PRE-LOAD THE FIRST EIGENVALUE VOLUME TO GET HEADER PARAMS
    L = nib.load('{}/dti_L1.nii.gz'.format(basename))
    s,m,n = L.get_fdata().shape

    # LOAD AND BUILD EIGENVALUES VOLUME
    evl = [L.get_fdata()]
    evl.append(nib.load('{}/dti_L2.nii.gz'.format(basename)).get_fdata())
    evl.append(nib.load('{}/dti_L3.nii.gz'.format(basename)).get_fdata())
    evl = np.array(evl)
    evl[evl<0] = 0

    # LOAD AND BUILD EIGENVECTORS VOLUME
    evt = [nib.load('{}/dti_V1.nii.gz'.format(basename)).get_fdata()]
    evt.append(nib.load('{}/dti_V2.nii.gz'.format(basename)).get_fdata())
    evt.append(nib.load('{}/dti_V3.nii.gz'.format(basename)).get_fdata())
    evt = np.array(evt).transpose(0,4,1,2,3)

    load_output_mask = nib.load('{}/inCCsight/cnnBased.nii.gz'.format(basename)).get_fdata()
    
    T = np.diag(np.ones(4))
    if reorient:
        # GET QFORM AFFINE MATRIX (see Nifti and nibabel specifications)
        T = L.header.get_qform()

        # COMPUTE ROTATION MATRIX TO ALIGN SAGITTAL PLANE
        R = align_sagittal_plane(T)
        evl, evt, T, load_output_mask = rotateDTI(evl, evt, R, load_output_mask)

    return (evl, evt, T, load_output_mask)

def rotateDTI(evl, evt, R, load_output_mask):
    
    import numpy as np
        
    s,m,n = evl[0].shape

    # ====== DETERMINE TARGET DOMAIN SIZE AND A TRANSLATION TO FIT THE ROTATED IMAGE =======
    # VERTICES FROM THE CUBE DEFINING THE ORIGINAL VOLUME
    cube = np.array([[0,0,0,1],
                     [0,0,n,1],
                     [0,m,n,1],
                     [0,m,0,1],
                     [s,m,0,1],
                     [s,0,0,1],
                     [s,0,n,1],
                     [s,m,n,1]]).transpose()

    # COMPUTE THE FIT TRANSLATION AND COMBINE WITH THE ROTATION
    cube = np.dot(R,cube)
    t = -cube.min(axis=1)
    Tr = np.diag(np.ones(4, dtype='float'))
    Tr[:3,3] = t[:3]
    T = np.dot(Tr,R)

    # DEFINE THE TARGET DOMAIN
    cube = cube + t.reshape(4,1)
    domain = np.ceil(cube.max(axis=1))[:3].astype('int')

    # === TRANSFORMATION ===
    invT = np.linalg.inv(T)
    N = domain.prod()

    # GET INDICES IN TARGET SPACE
    points = np.array(np.indices(domain)).reshape(3,N)
    points = np.vstack((points, np.ones(N)))

    # COMPUTE POINT COORDINATES WITH NEAREST NEIGHBOR INTERPOLATION
    points = np.dot(invT, points)[:3]
    points = np.round(points).astype('int')
    out_of_space = np.logical_or(points<0, points>=np.array([s,m,n]).reshape(3,1)).max(axis=0)
    points[:,out_of_space] = 0
    z,y,x = points

    # APPLY TRANSFORMATION TO THE EIGENVALUES VOLUME
    eigenvals = evl[:,z,y,x].copy()
    eigenvals[:,out_of_space] = 0
    eigenvals.shape = (3,) + tuple(domain)

    # APPLY TRANSFORMATION TO THE MANUAL MASK VOLUME
    output_mask = load_output_mask[z,y,x].copy()
    output_mask[out_of_space] = 0
    output_mask.shape = () + tuple(domain)

    # APPLY ROTATION TO THE EIGENVECTORS
    evt = evt.copy()
    evt.shape = (3,3,s*m*n)
    for i in range(3):
        evt[i] = np.dot(R[:3,:3],evt[i])
    evt.shape = (3,3,s,m,n)

    # APPLY TRANSFORMATION TO THE EIGENVECTORS VOLUME
    eigenvects = evt[:,:,z,y,x]
    eigenvects[:,:,out_of_space] = 0
    eigenvects.shape = (3,3) + tuple(domain)

    return (eigenvals, eigenvects, T, output_mask)

def run_analysis(rootdir):  
    
    import numpy as np
           
    eigvals, eigvects, T3, output_mask = loadNiftiDTI(basename=rootdir, reorient=True)

    FA,MD,RD,AD = getFractionalAnisotropy(eigvals)
    FA[np.isnan(FA)] = 0
    FA[FA>1] = 1

    fissure, FA_mean, FAmeanJo, FAmeanJo_min = getFissureSlice(eigvals, FA)
    
    wFA = FA*abs(eigvects[0,0]) #weighted FA

    return (wFA, FA, MD, RD, AD, fissure, T3, output_mask, FA_mean, FAmeanJo, FAmeanJo_min)

#print(fissure)

# Parcellation

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
