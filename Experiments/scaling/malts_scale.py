import os
import numpy as np
import pandas as pd
import time


from other_methods.pymalts import malts

random_state = int(os.getenv('RANDOM_STATE'))
save_folder = os.getenv('SAVE_FOLDER')
np.random.seed(random_state)

df_train = pd.read_csv(f'{save_folder}/df_train.csv')

malts = malts(outcome='Y', treatment='T', data=df_train, discrete=[], C=1, k=15, reweight=False,
              random_state=random_state)

start = time.time()
malts.fit()
fit_time = time.time() - start
print(fit_time)
