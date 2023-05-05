"""Script to run LCM Augment PGM experiment."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.variable_imp_matching import VIM
from other_methods import prognostic
from Experiments.helpers import get_data
import warnings
import seaborn as sns
import matplotlib

matplotlib.rcParams["mathtext.fontset"] = "custom"
matplotlib.rcParams["mathtext.rm"] = "Bitstream Vera Sans"
matplotlib.rcParams["mathtext.it"] = "Bitstream Vera Sans:italic"
matplotlib.rcParams["mathtext.bf"] = "Bitstream Vera Sans:bold"

warnings.filterwarnings("ignore")

np.random.seed(0)
config = {
    "num_samples": 5000,
    "imp_c": 5,
    "unimp_c": 15,
    "imp_d": 0,
    "unimp_d": 0,
    "n_train": 3000,
}
df_train, df_est, df_true, binary = get_data(data="dense_continuous",
                                             config=config)

prog = prognostic.Prognostic(
    "Y", "T", df_train, method="linear", double=True, random_state=0
)
_, c_mg, t_mg = prog.get_matched_group(df_est, k=25, diameter_prune=None)
mg = c_mg.join(t_mg, lsuffix="_c")

lcm = VIM("Y", "T", df_train)
lcm.fit()


def lcm_cates(row, df, lcm):
    idx = row.name
    matches = row.values
    df_ = df.loc[matches]
    cates = lcm.est_cate(df_estimation=df_, k=1, diameter_prune=None)
    cate_df = pd.DataFrame(cates)
    cate_df["idx"] = matches
    return cate_df.loc[cate_df["idx"] == idx, "CATE_mean"].values[0]


combo_cates_lin = mg.apply(lcm_cates, df=df_est, lcm=lcm, axis=1)

prog = prognostic.Prognostic(
    "Y", "T", df_train, method="ensemble", double=True, random_state=0
)
_, c_mg, t_mg = prog.get_matched_group(df_est, k=25, diameter_prune=None)
mg = c_mg.join(t_mg, lsuffix="_c")

lcm = VIM("Y", "T", df_train)
lcm.fit()


def lcm_cates(row, df, lcm):
    idx = row.name
    matches = row.values
    df_ = df.loc[matches]
    cates = lcm.est_cate(df_estimation=df_, k=1, diameter_prune=None)
    cate_df = pd.DataFrame(cates)
    cate_df["idx"] = matches
    return cate_df.loc[cate_df["idx"] == idx, "CATE_mean"].values[0]


combo_cates_gbt = mg.apply(lcm_cates, df=df_est, lcm=lcm, axis=1)

cates = lcm.est_cate(df_estimation=df_est, k=1, diameter_prune=None)
cate_df = pd.DataFrame(cates)

sns.set(font_scale=4, font="times")
fig, ax = plt.subplots(figsize=(20, 10))
df_res = pd.DataFrame()
df_res["Linear LAP"] = np.abs(combo_cates_lin - df_true["TE"])
df_res["Non-parametric\n LAP"] = np.abs(combo_cates_gbt - df_true["TE"])
df_res["LCM"] = np.abs(cate_df["CATE_mean"] - df_true["TE"])
sns.boxplot(data=df_res, fliersize=0, ax=ax, linewidth=5)
plt.xticks(rotation=0)
plt.ylabel("Absolute CATE \nEstimation Error")
plt.ylim(-2, 65)
plt.savefig("combining_pgm_w_lcm_results.png")
df_res.to_csv("combining_pgm_w_lcm_results.csv")

std_aug = []
std_lcm = []
for idx in df_est.index:
    mg_0 = lcm.create_mgs(df_estimation=df_est.loc[mg.loc[idx]], k=1)
    a = set(mg_0[0][0].loc[0]).union(mg_0[0][1].loc[0])
    df_est.loc[mg.loc[idx]].iloc[list(a)]

    mg_1 = lcm.create_mgs(df_estimation=df_est, k=1)
    b = set(mg_1[0][0].loc[idx]).union(mg_1[0][1].loc[idx])

    std_aug.append(df_est.loc[mg.loc[idx]].iloc[list(a)].std().drop(["Y", "T"]).values)
    std_lcm.append(df_est.loc[b].std().drop(["Y", "T"]).values)

std_aug = np.array(std_aug)
std_lcm = np.array(std_lcm)

fig, ax = plt.subplots(figsize=(20, 10))
pd.DataFrame(std_aug, columns=df_est.drop(columns=["Y", "T"]).columns).mean().plot()
pd.DataFrame(std_lcm, columns=df_est.drop(columns=["Y", "T"]).columns).mean().plot()
plt.axvline(4.5, c="black")
plt.xticks(
    np.arange(0, len(df_est.loc[a].std().drop(["Y", "T"]).index)),
    df_est.loc[a].std().drop(["Y", "T"]).index,
)
plt.xlim(0, 10)
plt.legend(["LCM aug PGM", "LCM"])
plt.ylabel("Avg. stdev per covariate within a MG")
plt.savefig("aug_pgm_matched_tight.png")

df_std_aug = (
    pd.DataFrame(std_aug, columns=df_est.drop(columns=["Y", "T"]).columns)
    .stack()
    .reset_index()
    .assign(Method=lambda x: "Non-parame-\ntric LAP")
    .rename(
        columns={0: "avg. stdev per covariate within a MG", "level_1": "covariates"}
    )
)
df_std_lcm = (
    pd.DataFrame(std_lcm, columns=df_est.drop(columns=["Y", "T"]).columns)
    .stack()
    .reset_index()
    .assign(Method=lambda x: "LCM")
    .rename(
        columns={0: "avg. stdev per covariate within a MG", "level_1": "covariates"}
    )
)


sns.set(font_scale=4, font="times")
fig, ax = plt.subplots(figsize=(18, 10))
sns.pointplot(
    x="covariates",
    y="avg. stdev per covariate within a MG",
    data=df_std_aug.append(df_std_lcm),
    join=False,
    hue="Method",
    dodge=True,
    scale=2.5,
    capsize=0.1,
    markers=["s", "o"],
)
plt.axvline(4.5, c="black")
ax.set_xlim(-1, 10.5)
plt.ylabel("Average stdev per \ncovariate within a MG")
plt.savefig("aug_pgm_matched_tight.png")