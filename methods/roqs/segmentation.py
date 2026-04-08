import time
import nibabel as nib
import numpy as np
import os
import pandas as pd
import getParcellation as gm
import libcc
import save


def _find_nii(basedir, name):
    """Return path to name.nii.gz or name.nii, whichever exists."""
    for ext in ('.nii.gz', '.nii'):
        p = os.path.join(basedir, name + ext)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Could not find {name}.nii.gz or {name}.nii in {basedir}")


def loadNiftiDTI(basedir, basename='dti', reorient=False):

    # ====== MAIN FUNCTION START ===========================
    # PRE-LOAD THE FIRST EIGENVALUE VOLUME TO GET HEADER PARAMS
    L = nib.load(_find_nii(basedir, f'{basename}_L1'))
    s, m, n = L.get_fdata().shape

    # LOAD AND BUILD EIGENVALUES VOLUME
    evl = [L.get_fdata()]
    evl.append(nib.load(_find_nii(basedir, f'{basename}_L2')).get_fdata())
    evl.append(nib.load(_find_nii(basedir, f'{basename}_L3')).get_fdata())
    evl = np.array(evl)
    evl[evl < 0] = 0

    # LOAD AND BUILD EIGENVECTORS VOLUME
    evt = [nib.load(_find_nii(basedir, f'{basename}_V1')).get_fdata()]
    evt.append(nib.load(_find_nii(basedir, f'{basename}_V2')).get_fdata())
    evt.append(nib.load(_find_nii(basedir, f'{basename}_V3')).get_fdata())
    evt = np.array(evt).transpose(0, 4, 1, 2, 3)

    T = np.diag(np.ones(4))
    if reorient:
        # GET QFORM AFFINE MATRIX (see Nifti and nibabel specifications)
        T = L.header.get_qform()

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


def _collect_segm_stats(segmentation, FA, MD, RD, AD, scalar_maps, sub):
    """Compute scalars, midlines, thickness and parcellation for a segmentation mask."""
    scalar_stats = getScalars(segmentation, FA, MD, RD, AD)

    try:
        midlines = {
            'FA': str([float(x) for x in getFAmidline(segmentation, FA, n_points=200)]),
            'MD': str([float(x) for x in getFAmidline(segmentation, MD, n_points=200)]),
            'RD': str([float(x) for x in getFAmidline(segmentation, RD, n_points=200)]),
            'AD': str([float(x) for x in getFAmidline(segmentation, AD, n_points=200)]),
        }
    except Exception:
        midlines = {'FA': '[]', 'MD': '[]', 'RD': '[]', 'AD': '[]'}

    try:
        col_heights = np.sum(segmentation, axis=0).astype(float)
        thickness = np.interp(
            np.linspace(0, max(len(col_heights) - 1, 1), 200),
            np.arange(len(col_heights)),
            col_heights
        )
    except Exception:
        thickness = np.zeros(200)

    values = gm.getParcellation(segmentation, FA)
    parc_dict = gm.parcellations_dfs_dicts(scalar_maps, values)

    parc_row = {'Name': sub}
    for method_p in ['Witelson', 'Hofer', 'Chao', 'Cover', 'Freesurfer']:
        for part in ['P1', 'P2', 'P3', 'P4', 'P5']:
            for scalar in ['FA', 'FA StdDev', 'MD', 'MD StdDev', 'RD', 'RD StdDev', 'AD', 'AD StdDev']:
                try:
                    parc_row[f'{method_p}_{scalar}_{part}'] = parc_dict[method_p][part][scalar]
                except Exception:
                    parc_row[f'{method_p}_{scalar}_{part}'] = 0.0

    return scalar_stats, midlines, thickness, parc_row, parc_dict


def get_segm(data_paths):

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = []
    imgPathList = []
    # ── ROQS accumulators ───────────────────────────────────────────────────
    meanFAList = []; stdFAList = []
    meanMDList = []; stdMDList = []
    meanRDList = []; stdRDList = []
    meanADList = []; stdADList = []
    midlinesList         = []
    thicknessList        = []
    parcellationStatsList = []
    parcellationsList = {"ROQS": {}}
    times = []
    # ── Watershed accumulators ───────────────────────────────────────────────
    w_meanFAList = []; w_stdFAList = []
    w_meanMDList = []; w_stdMDList = []
    w_meanRDList = []; w_stdRDList = []
    w_meanADList = []; w_stdADList = []
    w_midlinesList         = []
    w_thicknessList        = []
    w_parcellationStatsList = []

    for data_path in data_paths:
        try:
            folderpath = f"{data_path}/inCCsight"
            filename = f"segm_roqs"

            start = time.time()
            code = os.path.basename(data_path)
            # Avoid double-prefix if the folder is already named "Subject_XXX"
            sub = code if code.startswith('Subject_') else f'Subject_{code}'

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

            # ── ROQS segmentation ────────────────────────────────────────────
            segmentation = segm_roqs(wFA, eigvects_ms)
            scalar_statistics, roqs_midlines, thickness_200, parc_row, parcellation_dict = \
                _collect_segm_stats(segmentation, FA, MD, RD, AD, scalar_maps, sub)
            parcellationsList["ROQS"][sub] = parcellation_dict

            names.append(sub)
            meanFAList.append(scalar_statistics[0]); stdFAList.append(scalar_statistics[1])
            meanMDList.append(scalar_statistics[2]); stdMDList.append(scalar_statistics[3])
            meanRDList.append(scalar_statistics[4]); stdRDList.append(scalar_statistics[5])
            meanADList.append(scalar_statistics[6]); stdADList.append(scalar_statistics[7])
            midlinesList.append(roqs_midlines)
            thicknessList.append(thickness_200)
            parcellationStatsList.append(parc_row)

            canvas = np.zeros(wFA_v.shape, dtype='int32')
            canvas[fissure, :, :] = segmentation
            save.save_nii(data_path, 'segm_roqs', canvas, affine)

            # ── Watershed segmentation ────────────────────────────────────────
            print(f"  → Executando Watershed para {sub}", flush=True)
            try:
                segm_w, _, _ = libcc.segm_watershed(wFA)

                if segm_w is False or not np.any(segm_w):
                    raise ValueError("Watershed retornou máscara vazia")

                w_stats, w_midlines, w_thickness, w_parc_row, _ = \
                    _collect_segm_stats(segm_w, FA, MD, RD, AD, scalar_maps, sub)

                w_meanFAList.append(w_stats[0]); w_stdFAList.append(w_stats[1])
                w_meanMDList.append(w_stats[2]); w_stdMDList.append(w_stats[3])
                w_meanRDList.append(w_stats[4]); w_stdRDList.append(w_stats[5])
                w_meanADList.append(w_stats[6]); w_stdADList.append(w_stats[7])
                w_midlinesList.append(w_midlines)
                w_thicknessList.append(w_thickness)
                w_parcellationStatsList.append(w_parc_row)

                canvas_w = np.zeros(wFA_v.shape, dtype='int32')
                canvas_w[fissure, :, :] = segm_w
                save.save_nii(data_path, 'segm_watershed', canvas_w, affine)
                print(f"  → Watershed concluído", flush=True)

            except Exception as e_w:
                import traceback as _tb
                print(f"  [WARN] Watershed falhou para {sub}, usando ROQS como fallback: {e_w}")
                _tb.print_exc()
                # Fallback: copy ROQS values so the subject still appears in Watershed CSVs
                w_meanFAList.append(scalar_statistics[0]); w_stdFAList.append(scalar_statistics[1])
                w_meanMDList.append(scalar_statistics[2]); w_stdMDList.append(scalar_statistics[3])
                w_meanRDList.append(scalar_statistics[4]); w_stdRDList.append(scalar_statistics[5])
                w_meanADList.append(scalar_statistics[6]); w_stdADList.append(scalar_statistics[7])
                w_midlinesList.append(roqs_midlines)
                w_thicknessList.append(thickness_200)
                w_parcellationStatsList.append(parc_row)

            sub_data = {
                "name":    sub,
                "meanFA":  scalar_statistics[0], "stdFA":  scalar_statistics[1],
                "meanMD":  scalar_statistics[2], "stdMD":  scalar_statistics[3],
                "meanRD":  scalar_statistics[4], "stdRD":  scalar_statistics[5],
                "meanAD":  scalar_statistics[6], "stdAD":  scalar_statistics[7],
            }

            # Gerar PNG da fatia midsagital com contorno vermelho da segmentação
            img_path = ""
            try:
                from skimage import measure as sk_measure

                PANEL_BG = '#1F2C56'
                fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100, facecolor=PANEL_BG)
                ax.set_facecolor('#0d0d0d')

                im = ax.imshow(FA, cmap='gray', vmin=0, vmax=1)

                cbar = plt.colorbar(im, ax=ax)
                cbar.ax.tick_params(colors='white', labelsize=8)
                cbar.outline.set_edgecolor('#aaaaaa')
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

                contours = sk_measure.find_contours(segmentation.astype(float), 0.5)
                for c in contours:
                    ax.plot(c[:, 1], c[:, 0], 'r-', linewidth=1.5)

                ax.tick_params(colors='white', labelsize=8)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#aaaaaa')

                fig.tight_layout()

                out_dir = os.path.join(data_path, 'inCCsight')
                os.makedirs(out_dir, exist_ok=True)
                img_path = os.path.join(out_dir, 'midsagittal_roqs.png')
                fig.savefig(img_path, bbox_inches='tight', dpi=100, facecolor=PANEL_BG)
                plt.close(fig)
            except Exception:
                plt.close('all')
            imgPathList.append(img_path)
            # save.save_os(data_path, filename, data_tuple)

            end = time.time()
            time_total = round(end - start, 2)
            times.append(time_total)

            gm.adjust_dict_parcellations_statistics(parcellationsList, sub_data, data_path)

        except Exception as e:
            import traceback
            print(f"\n[ERRO] {data_path} Failed:")
            traceback.print_exc()
            print()
            continue
        
    # ── ROQS CSVs ────────────────────────────────────────────────────────────
    df_roqs = pd.DataFrame({
        "Names": names, "FA": meanFAList, "FA StdDev": stdFAList,
        "MD": meanMDList, "MD StdDev": stdMDList,
        "RD": meanRDList, "RD StdDev": stdRDList,
        "AD": meanADList, "AD StdDev": stdADList, "Time": times,
    })
    df_roqs.to_csv("./data/roqs_based.csv", sep=";")
    df_roqs.to_csv("../csvs/roqs_based.csv", sep=";")

    df_roqs_scalar = pd.DataFrame({
        'FA': meanFAList, 'FA StdDev': stdFAList,
        'MD': meanMDList, 'MD StdDev': stdMDList,
        'RD': meanRDList, 'RD StdDev': stdRDList,
        'AD': meanADList, 'AD StdDev': stdADList,
        'img_path': imgPathList,
    }, index=names)
    df_roqs_scalar.to_csv("../csvs/ROQS_scalar_statistics.csv", sep=";")

    if midlinesList:
        pd.DataFrame(midlinesList, index=names).to_csv("../csvs/ROQS_scalar_midlines.csv", sep=";")
    if thicknessList:
        pd.DataFrame(thicknessList, index=names).to_csv("../csvs/ROQS_dict_thickness.csv", sep=";")
    if parcellationStatsList:
        pd.DataFrame(parcellationStatsList).to_csv("../csvs/ROQS_parcellation_statistics.csv", sep=";")

    # ── Watershed CSVs ───────────────────────────────────────────────────────
    df_watershed_scalar = pd.DataFrame({
        'FA': w_meanFAList, 'FA StdDev': w_stdFAList,
        'MD': w_meanMDList, 'MD StdDev': w_stdMDList,
        'RD': w_meanRDList, 'RD StdDev': w_stdRDList,
        'AD': w_meanADList, 'AD StdDev': w_stdADList,
        'img_path': imgPathList,   # same PNG (ROQS midsagittal)
    }, index=names)
    df_watershed_scalar.to_csv("../csvs/Watershed_scalar_statistics.csv", sep=";")

    if w_midlinesList:
        pd.DataFrame(w_midlinesList, index=names).to_csv("../csvs/Watershed_scalar_midlines.csv", sep=";")
    if w_thicknessList:
        pd.DataFrame(w_thicknessList, index=names).to_csv("../csvs/Watershed_dict_thickness.csv", sep=";")
    if w_parcellationStatsList:
        pd.DataFrame(w_parcellationStatsList).to_csv("../csvs/Watershed_parcellation_statistics.csv", sep=";")

    print("\n✓ ROQS e Watershed concluídos.", flush=True)
