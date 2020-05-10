"""
Auxiliar functions Module
"""

import math, itertools, glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
#from sklearn.utils.fixes import signature
from inspect import signature
from scipy.ndimage.morphology import binary_erosion
import nibabel as nib
import matplotlib.pyplot as plt

def print_div(string,len_print=50):
    len_prn_str = len_print - len(string) - 2
    assert (len_prn_str > 0), 'Length print greater than expected'
    print('='*len_print)
    print('='*int(math.floor(len_prn_str/2)),string,'='*int(math.ceil(len_prn_str/2)))
    print('='*len_print)

def agreement_matrix(x1,x2):
    x = np.vstack((x1,x2)).T
    y = x.dot(1 << np.arange(x.shape[-1] - 1, -1, -1)).astype('uint8')
    m_ag = np.zeros((4))
    for pos in np.unique(y):
        m_ag[pos] = np.sum(y==pos)
    return m_ag[::-1]

def plot_matrix(cm, classes, normalize=False, title='Confusion matrix', fig_size=8, cmap=plt.cm.Blues, opt_bar=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #print(cm)

    plt.figure(figsize=(fig_size, fig_size))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    if opt_bar:
        plt.colorbar()
    tick_marks = np.arange(len(list(classes)))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if not opt_bar:
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt), size=fig_size*3, ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

    plt.show()

def plot_roc(y_true, y_pred_prob):
    """
    This function plots the ROC curve.
    """
    fpr, tpr, __ = metrics.roc_curve(y_true, y_pred_prob)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic Val')
    plt.plot(fpr, tpr, 'black', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], 'gray', linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    print('------------------------------------')
    print('ROC Curve Teste:')
    plt.show()

    return roc_auc

def plot_prc(y_true, y_pred_prob):
    """
    This function plots the Precision Recall F1 curve.
    """
    precision, recall, threshold = metrics.precision_recall_curve(y_true, y_pred_prob)
    f1 = 2*(precision*recall)/(precision+recall)
    f1_max_ix = threshold[np.argmax(f1)]
    average_precision = metrics.average_precision_score(y_true, y_pred_prob)
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})

    fig, ax = plt.subplots(1,2,figsize=(12,5))

    ax[0].step(recall, precision, color='b', alpha=0.2, where='post')
    ax[0].fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
    ax[0].set_xlabel('Recall')
    ax[0].set_ylabel('Precision')
    ax[0].set_ylim([0.0, 1.05])
    ax[0].set_xlim([0.0, 1.0])
    ax[0].set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

    ax[1].set_title('Precision and Recall Scores as a function of the decision threshold')
    ax[1].plot(threshold, precision[:-1], 'b-', label='Precision')
    ax[1].plot(threshold, recall[:-1], 'g-', label='Recall')
    ax[1].plot(threshold, f1[:-1], 'r-', label='f1')
    ax[1].axvline(x=f1_max_ix, label='Th at = {0:.2f}'.format(f1_max_ix), c='r', linestyle='--')
    ax[1].set_ylabel('Score')
    ax[1].set_xlabel('Decision Threshold')
    ax[1].legend(loc='best')
    plt.show()

    return average_precision, f1_max_ix

def report_metrics(cm):
    tp, tn = cm[1,1], cm[0,0]
    fp, fn = cm[0,1], cm[1,0]

    acc = (tp+tn)/np.sum(cm)
    rec = tp/(tp+fn)
    prec = tp/(tp+fp)
    f1 = 2*tp/(2*tp+fp+fn)

    return acc, rec, prec, f1

def print_mask_img(subj_, map_bd):

    pre_msp = 'msp_points_reg'
    msp_points_reg = glob.glob('{}{}.nii.gz'.format(subj_[0],pre_msp))
    gen_img_path = glob.glob('{}*_msp.nii'.format(subj_[0]))
    if msp_points_reg != []:
        in_img_msp = nib.load(msp_points_reg[0]).get_data()
        msp = np.argmax(np.sum(np.sum(in_img_msp,axis=-1),axis=-1))
        gen_img = nib.load('{}t1_reg.nii.gz'.format(subj_[0])).get_data()[msp]
        gen_mask = nib.load('{}mask_reg.nii.gz'.format(subj_[0])).get_data()[msp]
    elif gen_img_path != []:
        gen_img = nib.load(gen_img_path[0]).get_data()[::-1,::-1,0]
        gen_mask_path = glob.glob('{}*corrected.cc.nii'.format(subj_[0]))[0]
        gen_mask = nib.load(gen_mask_path).get_data()[::-1,::-1,0]
    else:
        gen_mask = np.load(subj_[0]).swapaxes(0,1)[::-1,::-1]
        gen_img = np.zeros((gen_mask.shape))

    seg_bin = gen_mask > 0.7
    seg_ero = binary_erosion(seg_bin)
    seg_brd = np.logical_xor(seg_bin,seg_ero)
    y, x = np.mgrid[0:seg_brd.shape[0], 0:seg_brd.shape[1]]

    fig, ax = plt.subplots(1,2,figsize = (8,4))
    ax[0].imshow(gen_img, cmap='gray')
    ax[0].grid(False)
    cb = ax[0].contourf(x, y, (seg_brd), 15, cmap=map_bd)
    ax[0].set_title('Prob: {}/ Label: {}'.format(subj_[2],subj_[3]))
    ax[1].plot(subj_[1].T)
    ax[1].set_title('Shape signature')
    plt.show()

def transparent_cmap(cmap, N=255):
    "Copy colormap and set alpha values"
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0, 0.2, N+4)
    return mycmap
