# coding: utf-8

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
    FAmean[MASKcount<=0.90*MASKcount.max()] = 1
    return (np.argmin(FAmean), FAmean)

def loadNiftiDTI(basedir, basename='dti', reorient=False):
    
    import nibabel as nib
    import numpy as np
    import os
    
    # ====== MAIN FUNCTION START ===========================
    # PRE-LOAD THE FIRST EIGENVALUE VOLUME TO GET HEADER PARAMS
    L = nib.load(os.path.join(basedir, '{}_L1.nii.gz'.format(basename)))
    s,m,n = L.get_data().shape

    # LOAD AND BUILD EIGENVALUES VOLUME
    evl = [L.get_data()]
    evl.append(nib.load(os.path.join(basedir, '{}_L2.nii.gz'.format(basename))).get_data())
    evl.append(nib.load(os.path.join(basedir, '{}_L3.nii.gz'.format(basename))).get_data())
    evl = np.array(evl)
    evl[evl<0] = 0

    # LOAD AND BUILD EIGENVECTORS VOLUME
    evt = [nib.load(os.path.join(basedir, '{}_V1.nii.gz'.format(basename))).get_data()]
    evt.append(nib.load(os.path.join(basedir, '{}_V2.nii.gz'.format(basename))).get_data())
    evt.append(nib.load(os.path.join(basedir, '{}_V3.nii.gz'.format(basename))).get_data())
    evt = np.array(evt).transpose(0,4,1,2,3)

    T = np.diag(np.ones(4))
    if reorient:
        # GET QFORM AFFINE MATRIX (see Nifti and nibabel specifications)
        T = L.get_header().get_qform()

        # COMPUTE ROTATION MATRIX TO ALIGN SAGITTAL PLANE
        R = align_sagittal_plane(T)
        evl, evt, T = rotateDTI(evl, evt, R)

    return (evl, evt, T)

def rotateDTI(evl, evt, R):
    
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

    return (eigenvals, eigenvects, T)
  
def grad_morf(f,ee):
    
    from scipy.ndimage.morphology import grey_dilation
    
    f_dil = grey_dilation(f,size="Optional",structure=ee)
    #bh = scipy.ndimage.morphology.black_tophat(f, size=(3,3))
    
    return f_dil-f
    
def run_analysis(rootdir, basename='dti'):  
    
    import numpy as np
           
    eigvals, eigvects, T3 = loadNiftiDTI(basedir=rootdir, basename=basename, reorient=True)

    FA,MD,RD,AD = getFractionalAnisotropy(eigvals)
    FA[np.isnan(FA)] = 0
    FA[FA>1] = 1

    fissure, FA_mean = getFissureSlice(eigvals, FA)

    wFA = FA*abs(eigvects[0,0]) #weighted FA

    return (wFA, FA, MD, RD, AD, fissure, eigvals, eigvects, T3)
  
def getLargestCC(segmentation):
    
    import numpy as np
    
    labels = label(segmentation, neighbors=4)
    cont_label = np.bincount(labels.flat)
    cont_label[0] = 0 #Discarding background
    largestCC = labels == np.argmax(cont_label)
    return largestCC
  
    