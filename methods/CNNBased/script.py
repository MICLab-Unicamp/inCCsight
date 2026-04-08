import os
import glob
import numpy as np
import nibabel as nib


def _find_nii(folder, name):
    for ext in ('.nii.gz', '.nii'):
        p = os.path.join(folder, name + ext)
        if os.path.isfile(p):
            return p
    return None


def _compute_fa_from_eigenvalues(folder, basename='dti'):
    """Compute FA volume from DTI eigenvalues and save as iso_dti_FA_norm.nii.gz."""
    l1_path = _find_nii(folder, f'{basename}_L1')
    l2_path = _find_nii(folder, f'{basename}_L2')
    l3_path = _find_nii(folder, f'{basename}_L3')

    if not (l1_path and l2_path and l3_path):
        return False

    print(f"      Computando FA a partir dos eigenvalues em {folder}...")
    img_l1 = nib.load(l1_path)
    l1 = img_l1.get_fdata().astype(np.float32)
    l2 = nib.load(l2_path).get_fdata().astype(np.float32)
    l3 = nib.load(l3_path).get_fdata().astype(np.float32)

    l1 = np.clip(l1, 0, None)
    l2 = np.clip(l2, 0, None)
    l3 = np.clip(l3, 0, None)

    eigvals = np.stack([l1, l2, l3], axis=0)
    MD = eigvals.mean(axis=0)
    num = np.sqrt(3 * ((eigvals - MD) ** 2).sum(axis=0))
    den = np.sqrt(2 * (eigvals ** 2).sum(axis=0))

    with np.errstate(divide='ignore', invalid='ignore'):
        FA = np.where(den > 0, num / den, 0.0).astype(np.float32)
    FA = np.clip(FA, 0, 1)

    out_path = os.path.join(folder, 'iso_dti_FA_norm.nii.gz')
    nib.save(nib.Nifti1Image(FA, img_l1.affine, img_l1.header), out_path)
    print(f"      Salvo: {out_path}")
    return True


def rename_files(path):
    """
    Ensure every subject folder under `path` has iso_dti_FA_norm.nii.gz.
    Priority:
      1. Already exists → nothing to do
      2. dti_FA.nii.gz or FA.nii.gz → rename
      3. dti_L1/L2/L3 eigenvalues → compute FA and save
    """
    for subdir, dirs, files in os.walk(path):
        target = os.path.join(subdir, 'iso_dti_FA_norm.nii.gz')

        if os.path.isfile(target):
            continue

        # Try renaming an existing FA file
        renamed = False
        for candidate in ('dti_FA.nii.gz', 'FA.nii.gz'):
            src = os.path.join(subdir, candidate)
            if os.path.isfile(src):
                os.rename(src, target)
                print(f"      Renomeado {candidate} → iso_dti_FA_norm.nii.gz em {subdir}")
                renamed = True
                break

        if not renamed:
            _compute_fa_from_eigenvalues(subdir)
