from datagen.dgp_df import dgp_learning_df

from src.linear_coef_matching_mf import LCM_MF
from other_methods import prognostic

df = dgp_learning_df(n_train=0)

lcm = LCM_MF(outcome='Y', treatment='T', data=df,
             n_splits=10, n_repeats=1, random_state=0)

# lcm.fit(model='linear')
# lcm.MG(k=15)
# lcm.CATE()
# mean = lcm.cate_df['CATE_mean'].mean().mean()
# std = lcm.cate_df['CATE_mean'].mean().std()
# print(mean - 1.96*std)
# print(mean + 1.96*std)

prog = prognostic.prognostic_cv(outcome='Y', treatment='T', data=df,
                                double=True, k_est=15, est_method='mean',
                                gen_skf=lcm.gen_skf, random_state=0)
print('hi')
