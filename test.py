from operator import itemgetter
import json
import itertools
from logger import Logger
import argparse
import numpy as np
import matplotlib as mpl

import matplotlib.ticker as tck
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
from scipy.signal import savgol_filter
from scipy.interpolate import splprep, splev
import pandas as pd
from plot_utils import *

def main():
    plt.plot(np.arange(10))
    plt.show()
    
main()