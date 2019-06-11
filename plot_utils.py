from logger import *
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 11})
import scipy
from scipy.signal import savgol_filter
import seaborn as sns
import pandas as pd
# sns.set_style("darkgrid")
# sns.set_style({'axes.facecolor':'0.0'})
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

def learning_curves(run_id,if_plt=0,feature='acc',phase='test'):
    lg = Logger()
    lg_file = lg.load(run_id)
    lg_list = lg_file.get(phase)
    feat = [lg_list[i][feature] for i in range(len(lg_list))]
    return feat

def get_yticks_range(acc):
    mini = np.min(acc)
    maxi = np.max(acc)
    mini_floor = np.floor(mini)
    maxi_ceil = np.ceil(maxi)
    if mini-mini_floor > 0.5:
        mini_floor+=0.5
    if maxi_ceil - maxi > 0.5:
        maxi_ceil-=0.5
    return mini_floor,maxi_ceil

def get_log_file(run_id):
    lg = Logger()
    lg_file = lg.load(run_id)
    return lg_file