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

def get_experiment_list(algorithm='pd_const',dataset='digits',char='best',name=None):
    
    if name is None:
        log_list_name = '{0}_{1}_{2}'.format(algorithm,dataset,char)
    else:log_list_name = name
    return d.item().get(log_list_name)

def plot_w_distri(ax,W):
    w_rows = np.linalg.norm(W,1,axis=1)
    w_rows /= np.max(w_rows)
    ax.hist(w_rows,bins=25)
    ax.set_xlim([-0.05,1.05])
    ax.set_title('Feature Selection',fontsize=12)
    ax.set_xlabel('Norm of the rows of W',size=12)
    ax.set_ylabel('Frequency',size=12)
    return ax
    
def plot_basic(ax, experiments, algo, key1=[], key2=[], title='Loss', ylabel=None, labels=None, hide_y = False):
    exlog = experiments 
    xlabel = 'Iterations'
    if title == 'losst':xlabel = 'Time (msec)'
    x, y = [], []
    for id_no,run_id in enumerate(exlog):

        try:
            logger = Logger.load(run_id)
        except ValueError:
            continue

        for j in range(len(key1)):
            y = [v[key1[j]] for v in logger.get(key2[j])]
            x = np.arange(len(y))+1
            label = labels[j]
            
            if 'loss' in title:
                ax.set_yscale('log')
                ax.set_xscale('log')
                    
            if title == 'losst':
                x = np.cumsum([v['time'] for v in logger.get(key2[j])])
                
            #ax.get_xaxis().set_major_formatter(matplotlib.ticker.LogFormatter())
            #ax.xaxis.set_major_locator(tck.MultipleLocator(base=50))
            #ax.grid()
            
            ax.grid(True,which="both",ls="-",lw=0.5)
            ax.plot(x, y, label=label)
            ax.set_xlabel(xlabel,size=12)
            
    if 'acc' in title:
        ax.set_ylim(top=1.05,bottom=0)
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_major_locator(tck.MultipleLocator(base=.25))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_major_locator(tck.MultipleLocator(base=25))

    ax.set_ylabel(ylabel,size=12)

    if 'loss' in title:
        title='Objective v.s. k'
    elif title=='acc':
        title='Training/Test Accuracy'
    elif title == 'p_residue':
        title='Primal Residue v.s. k'
    
    if title is not None:
        ax.set_title(title,fontsize=12)
    if hide_y:
        ax.yaxis.label.set_visible(False)
        
    legend_size = 9
    legend_loc = 'best'
    legend_fo = True
    if 'admm' in algo and 'ject' in title:
        legend_size = 6
        legend_loc = 'center'
        legend_fo = False
    ax.legend(prop={'size': legend_size},loc=legend_loc,frameon=True)
    
def plot_sen(ax, experiments, parameter, title='Loss', labels=None, hide_y = False):

    key1=[]
    key2=[]
    labels=[]

    if title=='Loss':
        ylabel = 'Final Cost'
#         labels.append('Frobenius norm')
#         key1.append('t7')
#         key2.append('train')
        labels.append('L_1 norm')
        key1.append('t1')
        key2.append('train')
    elif title == 'Acc':
        ylabel = 'Final Accuracy'
#         labels.append('Train Accuracy')
#         key1.append('acc')
#         key2.append('train')
        labels.append('Test Accuracy')
        key1.append('acc')
        key2.append('test')
    elif title==None:print('Please select one of loss or accuracy')    
            
    for j in range(len(key1)):
        x, y = [], []
        x_ax = 0
        for k in range(10):
            exlog = [experiments[0] + "_{0}".format(k)]

            for id_no,run_id in enumerate(exlog):

                try:
                    logger = Logger.load(run_id)
                except ValueError:
                    continue

                y_curr = [v[key1[j]] for v in logger.get(key2[j])]
                y.append(np.mean(y_curr[-10:]))
                x_ax = logger['args'][parameter]
            x.append(x_ax)    
        label = labels[j]
#         if 'Loss' in title:
#             y /= y[0]
        ax.loglog()
        ax.grid(True,which="both",ls="-",lw=0.5)
        ax.plot(x, y, label=label, marker='o', markersize=4)
        #ax.set_xscale('log')

    ax.set_xlabel(parameter,size=10)
    ax.set_ylabel(ylabel,size=10)
    
    if title=='Loss':
        title='Final Cost: ||Y\u03BC - XW||'
    elif title=='Acc':
        title='Final Training/Test Accuracy'
    
    if title is not None:
        ax.set_title(title,fontsize=10)
    if hide_y:
        ax.yaxis.label.set_visible(False)
        
    ax.legend(prop={'size': 10},frameon=False)
    
    plot_dict = {}
    plot_dict['x']=x
    plot_dict['y']=y
    np.save('Dicts/{0}'.format(experiments[0]),plot_dict)
    return x,y