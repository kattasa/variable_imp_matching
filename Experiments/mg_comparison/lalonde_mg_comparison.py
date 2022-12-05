# import json
# import numpy as np
# import os
# import pandas as pd
# import time
# import warnings
#
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# import seaborn as sns
#
# from datagen.dgp_df import dgp_lalonde
# from other_methods import pymalts, bart, causalforest, prognostic
# from src.linear_coef_matching import LCM
# from src.linear_coef_matching_mf import LCM_MF
# import pickle
#
# warnings.filterwarnings("ignore")
# np.random.seed(0)
#
#
# df_data, x_cols, discrete = dgp_lalonde()
# n_splits = 2
# n_repeats = 1
# k_est = 10
#
# ate = 886
#
# lcm = LCM_MF(outcome='Y', treatment='T', data=df_data, n_splits=n_splits, n_repeats=n_repeats)
# lcm.fit(double_model=False)
# lcm.MG(k=k_est)
# lcm.CATE(cate_methods=[['linear_pruned', False]])
# print('.')
#
# split_strategy = lcm.gen_skf
#
# # m = pymalts.malts_mf('Y', 'T', data=df_data, discrete=discrete, k_tr=15, k_est=k_est, n_splits=n_splits,
# #                      estimator='linear', smooth_cate=False, gen_skf=split_strategy)
#
# cate_est_prog, prog_c_mg, prog_t_mg = prognostic.prognostic_cv('Y', 'T', df_data, k_est=k_est, gen_skf=split_strategy)
#
# print('hi')