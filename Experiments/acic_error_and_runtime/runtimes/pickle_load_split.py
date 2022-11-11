import numpy as np
import pickle

def pickle_load_split(acic_results_folder, split_num):
    with open(f'{acic_results_folder}split.pkl', 'rb') as f:
        idx = pickle.load(f)[split_num]
    idx = [(np.array(i) + 1).tolist() for i in idx]
    return idx