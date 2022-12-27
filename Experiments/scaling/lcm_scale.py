import os
import numpy as np
import pandas as pd
import time

from src.linear_coef_matching import LCM

random_state = int(os.getenv('RANDOM_STATE'))
save_folder = os.getenv('SAVE_FOLDER')
np.random.seed(random_state)

df_train = pd.read_csv(f'{save_folder}/df_train.csv')

lcm = LCM(outcome='Y', treatment='T', data=df_train, random_state=random_state)

start = time.time()
lcm.fit()
fit_time = time.time() - start
print(fit_time)
