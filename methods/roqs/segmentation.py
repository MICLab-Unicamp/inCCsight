import time
import nibabel as nib
import numpy as np
import os
import pandas as pd
import getParcellation as gm
import libcc
import save


def loadNiftiDTI(basedir, basename='dti', reorient=False):

    # ====== MAIN FUNCTION START ===========================
    # PRE-LOAD THE FIRST EIGENVALUE VOLUME TO GET HEADER PARAMS
    L = nib.load(os.path.join(basedir, '{}_L1.nii.gz'.format(basename)))
    s, m, n = L.get_data().shape

    # LOAD AND BUILD EIGENVALUES VOLUME
    evl = [L.get_data()]
    evl.append(nib.load(os.path.join(
        basedir, '{}_L2.nii.gz'.format(basename))).get_data())
    evl.append(nib.load(os.path.join(
        basedir, '{}_L3.nii.gz'.format(basename))).get_data())
    evl = np.array(evl)
    evl[evl < 0] = 0

    # LOAD AND BUILD EIGENVECTORS VOLUME
    evt = [nib.load(os.path.join(
        basedir, '{}_V1.nii.gz'.format(basename))).get_data()]
    evt.append(nib.load(os.path.join(
        basedir, '{}_V2.nii.gz'.format(basename))).get_data())
    evt.append(nib.load(os.path.join(
        basedir, '{}_V3.nii.gz'.format(basename))).get_data())
    evt = np.array(evt).transpose(0, 4, 1, 2, 3)

    T = np.diag(np.ones(4))
    if reorient:
        # GET QFORM AFFINE MATRIX (see Nifti and nibabel specifications)
        T = L.get_header().get_qform()

        # COMPUTE ROTATION MATRIX TO ALIGN SAGITTAL PLANE
        R = align_sagittal_plane(T)
        evl, evt, T = rotateDTI(evl, evt, R)

    return (evl, evt, T)


def align_sagittal_plane(T):

    import numpy as np

    # CANONICAL BASE (i,j,k) (HOMOGENEOUS COORDINATES)
    c = np.array(
        [[0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1],
         [1, 1, 1, 1]])

    # FIND BASE V
    V_ = np.dot(T, c)
    V_ = V_[:3, 1:] - V_[:3, 0].reshape(3, 1)

    V = np.zeros((3, 3))
    V[np.arange(3), np.argmax(np.abs(V_), axis=1)] = 1
    V_[V_ == 0] = 1
    V = V * (V_/np.abs(V_))

    # DESIRED BASE W
    W = np.array([[1, 0, 0],
                  [0, 0, -1],
                  [0, -1, 0]])

    R = np.dot(np.linalg.inv(W), V)
    r = np.diag(np.ones(4))
    r[:3, :3] = R
    return r


def rotateDTI(evl, evt, R):

    import numpy as np

    s, m, n = evl[0].shape

    # ====== DETERMINE TARGET DOMAIN SIZE AND A TRANSLATION TO FIT THE ROTATED IMAGE =======
    # VERTICES FROM THE CUBE DEFINING THE ORIGINAL VOLUME
    cube = np.array([[0, 0, 0, 1],
                     [0, 0, n, 1],
                     [0, m, n, 1],
                     [0, m, 0, 1],
                     [s, m, 0, 1],
                     [s, 0, 0, 1],
                     [s, 0, n, 1],
                     [s, m, n, 1]]).transpose()

    # COMPUTE THE FIT TRANSLATION AND COMBINE WITH THE ROTATION
    cube = np.dot(R, cube)
    t = -cube.min(axis=1)
    Tr = np.diag(np.ones(4, dtype='float'))
    Tr[:3, 3] = t[:3]
    T = np.dot(Tr, R)

    # DEFINE THE TARGET DOMAIN
    cube = cube + t.reshape(4, 1)
    domain = np.ceil(cube.max(axis=1))[:3].astype('int')

    # === TRANSFORMATION ===
    invT = np.linalg.inv(T)
    N = domain.prod()

    # GET INDICES IN TARGET SPACE
    points = np.array(np.indices(domain)).reshape(3, N)
    points = np.vstack((points, np.ones(N)))

    # COMPUTE POINT COORDINATES WITH NEAREST NEIGHBOR INTERPOLATION
    points = np.dot(invT, points)[:3]
    points = np.round(points).astype('int')
    out_of_space = np.logical_or(points < 0, points >= np.array(
        [s, m, n]).reshape(3, 1)).max(axis=0)
    points[:, out_of_space] = 0
    z, y, x = points

    # APPLY TRANSFORMATION TO THE EIGENVALUES VOLUME
    eigenvals = evl[:, z, y, x].copy()
    eigenvals[:, out_of_space] = 0
    eigenvals.shape = (3,) + tuple(domain)

    # APPLY ROTATION TO THE EIGENVECTORS
    evt = evt.copy()
    evt.shape = (3, 3, s*m*n)
    for i in range(3):
        evt[i] = np.dot(R[:3, :3], evt[i])
    evt.shape = (3, 3, s, m, n)

    # APPLY TRANSFORMATION TO THE EIGENVECTORS VOLUME
    eigenvects = evt[:, :, z, y, x]
    eigenvects[:, :, out_of_space] = 0
    eigenvects.shape = (3, 3) + tuple(domain)

    return (eigenvals, eigenvects, T)


def getFractionalAnisotropy(eigvals):

    import numpy as np
    np.seterr(divide='ignore', invalid='ignore')

    MD = eigvals.mean(axis=0)
    FA = np.sqrt(3*((eigvals-MD)**2).sum(axis=0)) / \
        np.sqrt(2*(eigvals**2).sum(axis=0))

    RD = (eigvals[1]+eigvals[2])/2
    AD = eigvals[0]

    return (FA, MD, RD, AD)


def getFissureSlice(eigvals, FA):

    import numpy as np

    MASK = (eigvals[0] > 0)
    MASKcount = MASK.sum(axis=2).sum(axis=1)
    FAmean = FA.mean(axis=2).mean(axis=1)
    FAmean[MASKcount <= 0.90*MASKcount.max()] = 1
    return (np.argmin(FAmean), FAmean)


def run_analysis(rootdir, basename='dti'):

    import numpy as np

    eigvals, eigvects, T3 = loadNiftiDTI(
        basedir=rootdir, basename=basename, reorient=True)

    FA, MD, RD, AD = getFractionalAnisotropy(eigvals)
    FA[np.isnan(FA)] = 0
    FA[FA > 1] = 1

    fissure, FA_mean = getFissureSlice(eigvals, FA)

    wFA = FA*abs(eigvects[0, 0])  # weighted FA

    return (wFA, FA, MD, RD, AD, fissure, eigvals, eigvects, T3)


def segm_roqs(wFA_ms, eigvects_ms):

    import numpy as np
    from scipy.ndimage.morphology import binary_fill_holes
    from skimage.measure import label
    from skimage import measure

    # Seed grid search - get highest FA seed within central area
    h, w = wFA_ms.shape

    # Define region to make search
    region = np.zeros((h, w))
    region[int(h/3):int(2*h/3), int(w/2):int(2*w/3)] = 1
    region = wFA_ms * region

    # Get the indices of maximum element in numpy array
    fa_seed = np.amax(region)
    seedx, seedy = np.where(region == fa_seed)

    # Defining seeds positions
    seed = [seedx, seedy]

    # Get principal eigenvector (direction of maximal diffusivity)
    max_comp_in = np.argmax(eigvects_ms[:, seed[0], seed[1]], axis=0)
    max_comp_in = np.argmax(np.bincount(max_comp_in.ravel()))

    # Max component value
    Cmax_seed = eigvects_ms[max_comp_in, seed[0], seed[1]]

    # First selection criterion
    # Get pixels with the same maximum component (x,y or z) of the principal eigenvector
    princ = np.argmax(eigvects_ms, axis=0)
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

    return segm


def getFAmidline(segm, wFA_ms, n_points=200):

    import numpy as np
    from libcc import points

    # Get CC's midline
    px, py = points(segm, n_points+1)

    fa_line = []
    for aux in range(0, n_points):
        try:
            x = int(round(px[aux]))
            y = int(round(py[aux]))
            fa = wFA_ms[y, x]
        except:
            x = int(np.floor(px[aux]))
            y = int(np.floor(py[aux]))
            fa = wFA_ms[y, x]
        fa_line.append(fa)

    return fa_line


def getScalars(segm, wFA, wMD, wRD, wAD):

    import numpy as np

    # Total value
    meanFA = np.mean(wFA[segm == True])
    stdFA = np.std(wFA[segm == True])

    meanMD = np.mean(wMD[segm == True])
    stdMD = np.std(wMD[segm == True])

    meanRD = np.mean(wRD[segm == True])
    stdRD = np.std(wRD[segm == True])

    meanAD = np.mean(wAD[segm == True])
    stdAD = np.std(wAD[segm == True])

    return meanFA, stdFA, meanMD, stdMD, meanRD, stdRD, meanAD, stdAD


def get_segm(data_paths):

    names = []
    meanFAList = []
    stdFAList = []
    meanMDList = []
    stdMDList = []
    meanRDList = []
    stdRDList = []
    meanADList = []
    stdADList = []
    parcellationsList = {"ROQS": {}}
    times = []

    for data_path in data_paths:
        try:
            folderpath = f"{data_path}/inCCsight"
            filename = f"segm_roqs"

            start = time.time()
            code = os.path.basename(data_path)
            sub = f'Subject_{code}'

            print(f"Executando ROQS para {data_path}", flush=True)

            wFA_v, FA_v, MD_v, RD_v, AD_v, fissure, eigvals, eigvects, affine = run_analysis(
                data_path)

            wFA = wFA_v[fissure, :, :]
            FA = FA_v[fissure, :, :]
            MD = MD_v[fissure, :, :]
            RD = RD_v[fissure, :, :]
            AD = AD_v[fissure, :, :]
            eigvects_ms = abs(eigvects[0, :, fissure])

            scalar_maps = (FA, MD, RD, AD)
            segmentation = segm_roqs(wFA, eigvects_ms)

            values = gm.getParcellation(segmentation, FA)
            parcellation_dict = gm.parcellations_dfs_dicts(scalar_maps, values)
            parcellationsList["ROQS"][sub] = parcellation_dict

            scalar_statistics = getScalars(segmentation, FA, MD, RD, AD)

            # Midline
            scalar_midlines = {}

            try:
                scalar_midlines['FA'] = getFAmidline(
                    segmentation, FA, n_points=200)
                scalar_midlines['MD'] = getFAmidline(
                    segmentation, MD, n_points=200)
                scalar_midlines['RD'] = getFAmidline(
                    segmentation, RD, n_points=200)
                scalar_midlines['AD'] = getFAmidline(
                    segmentation, AD, n_points=200)
            except:
                scalar_midlines = {'FA': [], 'MD': [], 'RD': [], 'AD': []}

            # Check segmentation errors (True/False)
            error_flag = False
            error_prob = []
            try:
                error_flag, error_prob = libcc.checkShapeSign(
                    segmentation, shape_imports, threshold=0.6)
            except:
                error_flag = True

            # data_tuple = (segmentation, scalar_maps, scalar_statistics, scalar_midlines, error_prob, parcellation_dict)

            names.append(sub)
            meanFAList.append(scalar_statistics[0])
            stdFAList.append(scalar_statistics[1])
            meanMDList.append(scalar_statistics[2])
            stdMDList.append(scalar_statistics[3])
            meanRDList.append(scalar_statistics[4])
            stdRDList.append(scalar_statistics[5])
            meanADList.append(scalar_statistics[6])
            stdADList.append(scalar_statistics[7])

            name = sub
            meanFA = scalar_statistics[0] 
            stdFA = scalar_statistics[1]
            meanMD = scalar_statistics[2]
            stdMD = scalar_statistics[3]
            meanRD = scalar_statistics[4]
            stdRD = scalar_statistics[5]
            meanAD = scalar_statistics[6]
            stdAD = scalar_statistics[7]
            
            sub_data = {}

            names_maps = list(["name", "meanFA", "stdFA", "meanMD", "stdMD", "meanRD", "stdRD", "meanAD", "stdAD"])
            scalars_values = list([name,scalar_statistics[0], scalar_statistics[1], scalar_statistics[2], scalar_statistics[3], scalar_statistics[4], scalar_statistics[5], scalar_statistics[6], scalar_statistics[7]])
            

            for i in range(0, len(names_maps)):
                sub_data[names_maps[i]] = scalars_values[i]
            
            # Salvando os Dados
            canvas = np.zeros(wFA_v.shape, dtype='int32')
            canvas[fissure, :, :] = segmentation

            save.save_nii(data_path, 'segm_roqs', canvas, affine)
            # save.save_os(data_path, filename, data_tuple)

            end = time.time()
            time_total = round(end - start, 2)
            times.append(time_total)

            gm.adjust_dict_parcellations_statistics(parcellationsList, sub_data, data_path)

        except:
            print(f"{data_path} Failed.")
            continue
        
    subjects = {"Names": names, "FA": meanFAList, "FA StdDev": stdFAList, "MD": meanMDList, "MD StdDev": stdMDList, "RD": meanRDList, "RD StdDev": stdRDList, "AD": meanADList, "AD StdDev": stdADList, "Time": times}

    df = pd.DataFrame(subjects)
    df.to_csv("./data/roqs_based.csv", sep=";")
    df.to_csv("../csvs/roqs_based.csv", sep=";")
    #gm.adjust_dict_parcellations_statistics(parcellationsList)
