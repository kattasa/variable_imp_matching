import numpy as np
import pandas as pd

from datagen.dgp_df import dgp_poly_basic_df
from src.linear_coef_matching_mf import LCM_MF


random_state = 1
n_samples = 1000
n_imp = 1
n_t_imp = 1
n_unimp = 3
n_splits = 2
k_est = 15

_, df_data, df_true, x_cols = dgp_poly_basic_df(n_samples=1000,  n_imp=n_imp, n_t_imp=n_t_imp, n_unimp=n_unimp,
                                                n_train=0)

df_err = pd.DataFrame(columns=['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)'])

method_name = 'LASSO Coefficient Matching'
lcm = LCM_MF(outcome='Y', treatment='T', data=df_data, n_splits=n_splits, n_repeats=1, random_state=random_state)
split_strategy = lcm.gen_skf
lcm.fit(method='linear')
lcm.MG(k=k_est)
lcm.CATE(cate_methods=[['mean', False]])
cate_df = lcm.cate_df
cate_df = cate_df.rename(columns={'avg.CATE': 'Est_CATE'})
cate_df['True_CATE'] = df_true['TE'].to_numpy()
cate_df['Relative Error (%)'] = np.abs((cate_df['Est_CATE']-cate_df['True_CATE'])/np.abs(cate_df['True_CATE']).mean())
cate_df['Method'] = [method_name for i in range(cate_df.shape[0])]
df_err = pd.concat([df_err, cate_df[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']].copy(deep=True)])
print(f'{method_name} method complete')

method_name = 'RF Feature Importance Matching'
rfm = LCM_MF(outcome='Y', treatment='T', data=df_data, n_splits=n_splits, n_repeats=1, random_state=random_state)
rfm.gen_skf = split_strategy
rfm.fit(method='tree', params={'max_depth': 4})
rfm.MG(k=k_est)
rfm.CATE(cate_methods=[['mean', False]])
cate_df2 = rfm.cate_df
cate_df2 = cate_df2.rename(columns={'avg.CATE': 'Est_CATE'})
cate_df2['True_CATE'] = df_true['TE'].to_numpy()
cate_df2['Relative Error (%)'] = np.abs((cate_df2['Est_CATE']-cate_df2['True_CATE'])/np.abs(cate_df2['True_CATE']).mean())
cate_df2['Method'] = [method_name for i in range(cate_df2.shape[0])]
df_err = pd.concat([df_err, cate_df2[['Method', 'True_CATE', 'Est_CATE', 'Relative Error (%)']].copy(deep=True)])
print(f'{method_name} method complete')

print('hi')
