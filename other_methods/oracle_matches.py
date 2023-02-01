import numpy as np
import pandas as pd


def oracle_matches(samples, df_true, k=10):
    c_mg = []
    t_mg = []
    for s in samples:
        c_mg.append(list(
            np.abs(df_true.loc[samples].reset_index().loc[df_true.loc[samples].reset_index()['T'] == 0, 'Y0_true'] - df_true.loc[s, 'Y0_true']).sort_values().head(k).index))
        t_mg.append(list(
            np.abs(df_true.loc[samples].reset_index().loc[df_true.loc[samples].reset_index()['T'] == 1, 'Y1_true'] - df_true.loc[s, 'Y1_true']).sort_values().head(k).index))

    c_mg = pd.DataFrame(c_mg)
    t_mg = pd.DataFrame(t_mg)
    c_mg.index = samples
    t_mg.index = samples
    return c_mg, t_mg
