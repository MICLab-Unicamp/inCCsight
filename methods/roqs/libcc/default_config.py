"""
This module contains all the configuration of simulation environment
"""

DIR_BAS = '/winuno/Unicamp-FEEC/CC_clas_project/images_will/dados_pipe/'
DIR_SAVE = './saves/'

REG_EX = 0.15
SMOOTHNESS = 700
DEGREE = 5
FIT_RES = 0.35
RESOLS_INF = 0.01
RESOLS_SUP = 0.5
RESOLS_STEP = 0.01
POINTS = 500

#Method to choose cluster representant:
#'min_dist': Element with smallest intra-cluster distance
#'random': Element selected randomly
#'best_acc': Element with best AUC
#'min_dist', 'random', 'best_acc'
CHOSEN_METHOD = 'best_acc'

FL_GRAPH = False
FL_SAVE = False
