import sklearn
print(sklearn.__version__)

def shapeSignImports():
    
    from libcc import default_config as df_conf
    from libcc import func as sign
    from libcc import aux_fnc
    import numpy as np
    import joblib

    DIR_SAVE = './libcc/saves/'

    # Importing parameters
    parms_refs = joblib.load('{}sign_refs.joblib'.format(DIR_SAVE)) 
    prof_ref = parms_refs['prof_ref']
    res_chs = parms_refs['res_chs']
 
    # Importing 
    d_train = joblib.load('{}arr_models_ind.joblib'.format(DIR_SAVE))
    clf = joblib.load('{}ensemble_model.joblib'.format(DIR_SAVE))
    val_norm = parms_refs['val_norm']
    
    # Resolutions
    resols = np.arange(df_conf.RESOLS_INF,df_conf.RESOLS_SUP,df_conf.RESOLS_STEP)
    resols = np.insert(resols,0,df_conf.FIT_RES)
    
    return (prof_ref, res_chs, d_train, clf, val_norm, resols)
    
def checkShapeSign(segmentation, imports, threshold=0.5):
    
    prof_ref, res_chs, d_train, clf, val_norm, resols = imports
    from libcc import default_config as df_conf
    from libcc import func as sign
    from libcc import aux_fnc
    import numpy as np

    from scipy.ndimage.morphology import binary_fill_holes
    from skimage import measure

    # Make mask out of the array
    contours = measure.find_contours(segmentation, 0.1)
    contour = sorted(contours, key=lambda x: len(x))[-1]
    r_mask = np.zeros_like(segmentation, dtype='bool')
    r_mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
    segmentation = binary_fill_holes(r_mask)
    
    # Extract shape signature
    refer_temp = sign.sign_extract(segmentation, resols, df_conf.SMOOTHNESS, df_conf.POINTS)
    prof_vec = sign.sign_fit(prof_ref, refer_temp, df_conf.POINTS)
    
    # Filtering the fitting resolution
    X_test = prof_vec[1:,:]

    # Normalization
    X_test = X_test / val_norm
    
    # Concatenating infos
    svm_ind = np.array([]).reshape(0,X_test.shape[0])
    for res_ch in res_chs:
        svm_ind = np.vstack((svm_ind, d_train["string{0}".format(res_ch)].predict_proba(X_test[:,res_ch,:])[:,1]))
    svm_ind = svm_ind.T
    
    # Predict and threshold
    y_pred_probs = clf.predict_proba(svm_ind)[:,1]
    y_pred = y_pred_probs > threshold
    
    return y_pred[0], y_pred_probs