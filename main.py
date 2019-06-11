#!/usr/bin/env python3

import argparse
import copy
import os
import shutil
import time
import sys


import cvxpy as cp
import numpy as np
import pywt

from logger import Logger

import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from projection import *
from datasets import *
from logger import *
from utils import *
from plot_basic_figures import *

parser = argparse.ArgumentParser(description='Large-Scale-Optimization')
parser.add_argument('--niter',default=200, type=int, help='Number of iterations')
parser.add_argument('--lamda', default=1, type=float, help='Lambda')
parser.add_argument('--eta', default=100, type=float, help='Eta')
parser.add_argument('--rho', default=0.0001, type=float, help='Rho')
parser.add_argument('--rho_', default=0.0001, type=float, help='Rho_')
parser.add_argument('--gamma', default=1, type=float, help='Gamma')
parser.add_argument('--alpha', default=1, type=float, help='Alpha')
parser.add_argument('--alpha_', default=1, type=float, help='Alpha_')
parser.add_argument('--tau', default=0.01, type=float, help='Tau')
parser.add_argument('--tau_', default=0.01, type=float, help='Tau_')
parser.add_argument('--tau_mu', default=0.1, type=float, help='Tau mu')
parser.add_argument('--beta', default=10, type=float, help='beta')
parser.add_argument('--tau_incr', default=2, type=float, help='Tau incr')
parser.add_argument('--tau_decr', default=2, type=float, help='Tau decr')
parser.add_argument('--theta', default=0.01, type=float, help='Theta')
parser.add_argument('--sigma', default=0, type=float, help='Sigma')
parser.add_argument('--dataset',default='digits',type=str, help='Dataset')
parser.add_argument('--algorithm',default='pd_const',type=str, help='Algorithm')
parser.add_argument('--log-name',default=None,type=str, help='Log-name')
parser.add_argument('--log-scale',action='store_true', help='Log-scale')
parser.add_argument('--train-size',default=100,type=int,help='Size of training set')
parser.add_argument('--test-size',default=100,type=int,help='Size of testing set')
parser.add_argument('--classes',default=10,type=int,help='Number of classes')
parser.add_argument('--sen-range',default=10,type=int,help='Sensitivity range')
parser.add_argument('--sen', action='store_true',help='Sensitivity')
parser.add_argument('--save', action='store_true',help='Save model')
parser.add_argument('--save-fig', action='store_true',help='Save figure')
parser.add_argument('--residual', action='store_true',help='Compute and save residual')
parser.add_argument('--param',default='tau',type=str, help='Chosse which parameter')
args = parser.parse_args()

def get_dictionary():
    dicti = {}
    dicti['niter']=args.niter
    dicti['lamda']=args.lamda
    dicti['eta']=args.eta
    dicti['rho']=args.rho
    dicti['rho_']=args.rho_
    dicti['gamma']=args.gamma
    dicti['alpha']=args.alpha
    dicti['alpha_']=args.alpha_
    dicti['tau']=args.tau
    dicti['tau_']=args.tau_
    dicti['tau_mu']=args.tau_mu
    dicti['tau_incr']=args.tau_incr
    dicti['tau_decr']=args.tau_decr
    dicti['beta']=args.beta
    dicti['theta']=args.theta
    dicti['sigma']=args.sigma
    dicti['dataset']=args.dataset
    dicti['algorithm']=args.algorithm
    dicti['log_name']=args.log_name
    dicti['log_scale']=args.log_scale
    dicti['train_size']=args.train_size
    dicti['test_size']=args.test_size
    dicti['classes']=args.classes
    dicti['sen_range']=args.sen_range
    dicti['sen']=args.sen
    dicti['save']=args.save
    dicti['save_fig']=args.save_fig
    dicti['param']=args.param
    return dicti

def plot_exp(dicti,W):
    if 'admm' in dicti['algorithm']:
        fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(1 + 3*3., 2.5), sharey=False)
        ax1 = plot_basic(ax1,experiments=[dicti['log_name']],key1=['acc','acc'],key2=['train','test'],algo=dicti['algorithm'], labels=['Train Acc.','Test Acc.'],ylabel='Accuracy',title='acc')
        ax2 = plot_basic(ax2,experiments=[dicti['log_name']],key1=['t3','t1','t5','t6'],key2=['train','train','train','train'],algo=dicti['algorithm'],labels=['||\u03BC - I||_F','||Y\u03BC - XW||_1','||P-Y\u03BC+XD||_F','||W-D||_F'],ylabel='Objective',title='loss')
        #ax3 = plot_basic(ax3,experiments=[dicti['log_name']],key1=['t5','t6'],key2=['train','train'],labels=['||P-Y\u03BC+XD||_F','||W-D||_F'],ylabel='Primal Residue',title='p_residue')
        ax3 = plot_w_distri(ax3,W)
        fig.suptitle('{0}-{1}'.format('ADMM',dicti['dataset']),x=0.52,y=1.02,size=12)
        fig.tight_layout()
        if dicti['save_fig']:
            fig.savefig('{0}{1}_{2}{3}'.format('Plots/',dicti['log_name'],'LxAc','.pdf'),format='pdf', bbox_inches='tight')
        plt.show()
            
    elif 'pd' in dicti['algorithm']:
        fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(1 + 3*3., 2.5), sharey=False)
        ax1 = plot_basic(ax1,experiments=[dicti['log_name']],key1=['acc','acc'],key2=['train','test'],algo=dicti['algorithm'],labels=['Train Acc.','Test Acc.'],ylabel='Accuracy',title='acc')
        ax2 = plot_basic(ax2,experiments=[dicti['log_name']],key1=['t3','t1'],key2=['train','train'],algo=dicti['algorithm'],labels=['||\u03BC - I||_F','||Y\u03BC - XW||_1'],ylabel='Objective',title='loss')
        #ax3 = plot_basic(ax3,experiments=[dicti['log_name']],key1=['t7','t1'],key2=['train','train'],labels=['Frobenius','L1'],ylabel='Cost',title='losst')
        fig.suptitle('{0} on {1} {2}'.format('Primal-Dual Algorithm',dicti['dataset'],'dataset'),x=0.52,y=1.02,size=12)
        ax3 = plot_w_distri(ax3,W)
        fig.tight_layout()
        if dicti['save_fig']:
            fig.savefig('{0}{1}_{2}{3}'.format('Plots/',dicti['log_name'],'LxAc','.pdf'),format='pdf', bbox_inches='tight')
        plt.show()
        
def param_distance(M1,M2):
    assert M1.shape == M2.shape
    return np.linalg.norm(M1-M2,'fro')/np.linalg.norm(M2,'fro')
    
def operator_norm(M,norm_type='eig'):
    nrm = 0
    if norm_type=='eig':
        nrm = np.linalg.norm(M,2) 
    elif norm_type=='fro':
        nrm = np.linalg.norm(M,'fro')
    return nrm

def initialize_params(W_shape,mu_shape,Z_shape,P_shape,D_shape,V_shape):
    W = np.random.randn(W_shape[0],W_shape[1])
    mu = np.eye(mu_shape[0])
    Z = np.random.randn(Z_shape[0],Z_shape[1])
    P = np.random.randn(P_shape[0],P_shape[1])
    D = np.random.randn(D_shape[0],D_shape[1])
    V = np.random.randn(V_shape[0],V_shape[1])
    return W,mu,Z,P,D,V

def get_sigma(Y,X,tau,tau_mu,rho,gamma,over):
    nr = 'eig'
    if not over:
        term = ((tau_mu/(1+tau_mu*rho/4))*(operator_norm(Y,nr)**2) + tau*(operator_norm(X,nr)**2))
    else:
        if gamma >= 0.5: rho=0
        term_0 = (1-2*gamma)/(1-gamma)
        term = (tau_mu/(1+term_0*tau_mu*rho/4)*(operator_norm(Y,nr)**2) + tau*(operator_norm(X,nr)**2))    
    return 1/term

def get_loss(W,mu,Z,P,D,V,X,Y,train):
    
    term1 = np.linalg.norm(np.matmul(Y,mu)-np.matmul(X,W),1)
    term2 = (np.sum(np.abs(W)))
    term3 = np.linalg.norm(np.eye(mu.shape[0])-mu)
    term4 = np.linalg.norm(P,1)
    term5 = 0
    if train:
        term5 = np.linalg.norm(P-np.matmul(Y,mu)+np.matmul(X,D),'fro')
    term6 = np.linalg.norm(W-D,'fro')
    term7 = np.linalg.norm(np.matmul(Y,mu)-np.matmul(X,W),'fro')
    
    if 'pd' in args.algorithm:
        if 'soft' in args.algorithm:
            loss = term1+term2+term3
        elif 'const' in args.algorithm or 'frob' in args.algorithm:
            loss = term1+term3
        elif 'elastic' in args.algorithm: 
            loss = term1+term3+(args.alpha/2)*(np.linalg.norm(W,'fro'))
    elif 'admm' in args.algorithm:
        if 'soft' in args.algorithm:
            loss = term4+term2+term3
        elif 'const' in args.algorithm or 'frob' in args.algorithm:
            loss = term4+term3
        elif 'elastic' in args.algorithm:
            loss = term4+term3+(args.alpha/2)*(np.linalg.norm(W,'fro'))
            
    return term1,term2,term3,term4,term5,term6,term7,loss

def get_accuracy(W,mu,Z,X,Y):
    feature_pred = np.matmul(X,W)
    accuracy = 0

    for i in range(Y.shape[0]):
        class_pred = np.argmin(np.linalg.norm(mu-feature_pred[i,:],1,axis=1))
        if Y[i,class_pred] == 1:accuracy+=1
    return accuracy/Y.shape[0]

def train(epoch,W,mu,Z,P,D,V,X,Y,dicti,algorithm,logger,inv_x,inv_y,train=True):
    label='testt'
    time1 = 0
    time2 = 0
    time_taken = 0
    if train:
        label = 'train'
        
        #store the old value
        if args.residual:
            W_old = copy.deepcopy(W)
            mu_old = copy.deepcopy(mu)
            Z_old = copy.deepcopy(Z)
            P_old = copy.deepcopy(P)
            D_old = copy.deepcopy(D)
            V_old = copy.deepcopy(V)

        time1 = time.time()
        W,mu,Z,P,D,V = algorithm(W,mu,Z,P,D,V,X,Y,dicti,inv_x,inv_y)
        time2 = time.time()
        time_taken = time2-time1
        print('Time for iteration {0} : {1} {2}'.format(epoch,round((time_taken)*1000,2),'msec'))

        #compute the residual
        if args.residual:
            W_dist = param_distance(W,W_old)
            mu_dist = param_distance(mu,mu_old)
            Z_dist = param_distance(Z,Z_old)
            P_dist = param_distance(P,P_old)
            D_dist = param_distance(D,D_old)
            V_dist = param_distance(V,V_old)
    
    t1,t2,t3,t4,t5,t6,t7,lx = get_loss(W,mu,Z,P,D,V,X,Y,train)
    acc = get_accuracy(W,mu,Z,X,Y)

    #Store train and test performance
    if 'pd' in args.algorithm:
        print(f"{label}: [{epoch}] ||Y\u03BC-XW||_1: {t1:.2f} ||W||_1: {t2:.2f} ||\u03BC-I||_F: {t3:.2f} ||Y\u03BC-XW||_F: {t7:.2f} loss: {lx:.2f} acc: {acc:.2f}")    
        logger.append('train' if train else 'test', epoch=epoch,t1=t1,t2=t2,t3=t3,t7=t7,loss=lx,acc=acc,time=time_taken)
    elif 'admm' in args.algorithm:
        print(f"{label}: [{epoch}] ||Y\u03BC-XW||_1: {t1:.2f} ||W||_1: {t2:.2f} ||\u03BC-I||_F: {t3:.2f} ||P||_1: {t4:.2f} ||P-Y\u03BC+XD||_F: {t5:.2f} ||W-D||_F: {t6:.2f} ||Y\u03BC-XW||_F: {t7:.2f} loss: {lx:.2f} acc: {acc:.2f}")    
        logger.append('train' if train else 'test', epoch=epoch,t1=t1,t2=t2,t3=t3,t4=t4,t5=t5,t6=t6,t7=t7,loss=lx,\
                      acc=acc,time=time_taken)
        
    if not train:print('')
        
    #Store the residuals
    if args.residual:
        if train:
            label = 'residual'
            print(f"{label}: [{epoch}] W_dist: {W_dist:.2f} mu_dist: {mu_dist:.2f} Z_dist: {Z_dist:.2f}\
            P_dist: {P_dist:.2f} D_dist: {D_dist:.2f} V_dist: {V_dist:.2f}")
            logger.append('residual',epoch=epoch,W_dist=W_dist,mu_dist=mu_dist,Z_dist=Z_dist,\
                          P_dist=P_dist,D_dist=D_dist,V_dist=V_dist)
    
    return W,mu,Z,P,D,V
    
def test(epoch,W,mu,Z,P,D,V,X,Y,dicti,algorithm,logger,inv_x,inv_y):
    _,_,_,_,_,_ = train(epoch,W,mu,Z,P,D,V,X,Y,dicti,algorithm,logger,inv_x,inv_y,train=False)
    
def soft_thresholding(W,lamda):
    W = pywt.threshold(W, lamda, mode='soft', substitute=0)
    return W

########################
## Primal-Dual method ##
########################

def pd_soft(W,mu,Z,P,D,V,X,Y,dicti,inv_x,inv_y):
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    
    W += dicti['tau']*(np.matmul(X.transpose(),Z))
    W = soft_thresholding(W,dicti['lamda'])
    mu = (1/(1+dicti['tau']*dicti['rho']))*(mu + dicti['rho']*dicti['tau_mu']*np.eye(mu.shape[0]) - dicti['tau_mu']*(np.matmul(Y.transpose(),Z)))
    Z = np.maximum(-1,(np.minimum(1,Z + dicti['sigma']*(np.matmul(Y,2*mu - mu_old) - np.matmul(X,2*W - W_old)))))
    return W,mu,Z,P,D,V

def pd_const(W,mu,Z,P,D,V,X,Y,dicti,inv_x,inv_y):
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    
    W_shape = W.shape
    W += dicti['tau']*(np.matmul(X.transpose(),Z))
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=dicti['eta'])
    W = W_vec.reshape(W_shape)
    mu = (1/(1+dicti['tau_mu']*dicti['rho']))*(mu_old + dicti['rho']*dicti['tau_mu']*np.eye(mu.shape[0]) - dicti['tau_mu']*(np.matmul(Y.transpose(),Z)))
    Z = np.maximum(-1,(np.minimum(1,Z + dicti['sigma']*(np.matmul(Y,2*mu - mu_old) - np.matmul(X,2*W - W_old)))))
    return W,mu,Z,P,D,V

def pd_const_over(W,mu,Z,P,D,V,X,Y,dicti,inv_x,inv_y):
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    
    W_shape = W.shape
    W += dicti['tau']*(np.matmul(X.transpose(),Z))
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=dicti['eta'])
    W = W_vec.reshape(W_shape)
    mu = (1/(1+dicti['tau']*dicti['rho']))*(mu + dicti['rho']*dicti['tau_mu']*np.eye(mu.shape[0]) - dicti['tau_mu']*(np.matmul(Y.transpose(),Z)))
    Z = np.maximum(-1,(np.minimum(1,Z + dicti['sigma']*(np.matmul(Y,2*mu - mu_old) - np.matmul(X,2*W - W_old)))))
    
    W += dicti['gamma']*(W - W_old)
    mu += dicti['gamma']*(mu - mu_old)
    Z += dicti['gamma']*(Z - Z_old)
    return W,mu,Z,P,D,V

def pd_frob(W,mu,Z,P,D,V,X,Y,dicti,inv_x,inv_y):
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    
    W_shape = W.shape
    W += dicti['tau']*(np.matmul(X.transpose(),Z))
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=dicti['eta'])
    W = W_vec.reshape(W_shape)
    mu = (1/(1+dicti['tau']*dicti['rho']))*(mu_old + dicti['rho']*dicti['tau_mu']*np.eye(mu.shape[0]) - dicti['tau_mu']*(np.matmul(Y.transpose(),Z)))
    Z += dicti['sigma']*(np.matmul(Y,2*mu - mu_old) - np.matmul(X,2*W - W_old))
    Z = Z/(np.maximum(1,np.linalg.norm(Z,'fro')))
    return W,mu,Z,P,D,V

def pd_frob_over(W,mu,Z,P,D,V,X,Y,dicti,inv_x,inv_y):
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    
    W_shape = W.shape
    W += dicti['tau']*(np.matmul(X.transpose(),Z))
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=dicti['eta'])
    W = W_vec.reshape(W_shape)
    mu = (1/(1+dicti['tau']*dicti['rho']))*(mu + dicti['rho']*dicti['tau_mu']*np.eye(mu.shape[0]) - dicti['tau_mu']*(np.matmul(Y.transpose(),Z)))
    Z += dicti['sigma']*(np.matmul(Y,2*mu - mu_old) - np.matmul(X,2*W - W_old))
    Z = Z/(np.maximum(1,np.linalg.norm(Z,'fro')))
    
    W += dicti['gamma']*(W - W_old)
    mu += dicti['gamma']*(mu - mu_old)
    Z += dicti['gamma']*(Z - Z_old)
    return W,mu,Z,P,D,V

def pd_elastic(W,mu,Z,P,D,V,X,Y,dicti,inv_x,inv_y):
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
        
    W_shape = W.shape
    W = (1/(1+dicti['tau']*dicti['alpha']))*(W + dicti['tau']*(np.matmul(X.transpose(),Z)))
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=dicti['eta'])
    W = W_vec.reshape(W_shape)
    mu = (1/(1+dicti['tau']*dicti['rho']))*(mu_old + dicti['rho']*dicti['tau_mu']*np.eye(mu.shape[0]) - dicti['tau_mu']*(np.matmul(Y.transpose(),Z)))
    Z = np.maximum(-1,(np.minimum(1,Z + dicti['sigma']*(np.matmul(Y,2*mu - mu_old) - np.matmul(X,2*W - W_old)))))
    return W,mu,Z,P,D,V

def pd_elastic_var(W,mu,Z,P,D,V,X,Y,dicti,inv_x,inv_y):
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    
    dicti['tau'] = dicti['tau_']/dicti['alpha_']
    dicti['tau_mu'] = dicti['tau_']/dicti['rho_']
    W_shape = W.shape
    W = (1/(1+dicti['tau']*dicti['alpha']))*(W + dicti['tau']*(np.matmul(X.transpose(),Z)))
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=dicti['eta'])
    W = W_vec.reshape(W_shape)
    mu = (1/(1+dicti['tau']*dicti['rho']))*(mu_old + dicti['rho']*dicti['tau_mu']*np.eye(mu.shape[0]) - dicti['tau_mu']*(np.matmul(Y.transpose(),Z)))
    
    dicti['theta'] = 1/np.sqrt(1+dicti['tau_'])
    dicti['tau_'] = dicti['theta']*dicti['tau_']
    dicti['sigma'] = dicti['sigma']/dicti['theta']
    W_ = W + dicti['theta']*(W - W_old)
    mu_ = mu + dicti['theta']*(mu - mu_old)
    Z = np.maximum(-1,(np.minimum(1,Z + dicti['sigma']*(np.matmul(Y,mu_) - np.matmul(X,W_)))))
    return W,mu,Z,P,D,V

##########
## ADMM ##
##########

def admm_soft(W,mu,Z,P,D,V,X,Y,dicti,inv_x,inv_y):        
    D_old = copy.deepcopy(D)
    #Step1
    P = soft_thresholding(np.matmul(Y,mu) - np.matmul(X,D) - (1/dicti['tau'])*Z,(1/dicti['tau']))
    W = soft_thresholding(D - (1/dicti['tau'])*V,(dicti['lamda']/dicti['tau']))
    #Step2
    D = np.matmul(inv_x,W + (1/dicti['tau'])*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/dicti['tau'])*Z))
    mu = np.matmul(inv_y,dicti['tau']*np.matmul(Y.transpose(),P + np.matmul(X,D_old) + (1/dicti['tau'])*Z) + dicti['rho']*np.eye(Z.shape[1]))
    #Step3
    Z += dicti['tau']*(P - np.matmul(Y,mu) + np.matmul(X,D))
    V += dicti['tau']*(W-D)
    return W,mu,Z,P,D,V

def admm_soft_vp(W,mu,Z,P,D,V,X,Y,dicti,inv_x,inv_y):        
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    P_old = copy.deepcopy(P)
    D_old = copy.deepcopy(D)
    V_old = copy.deepcopy(V)
    #Step1
    P = soft_thresholding(np.matmul(Y,mu) - np.matmul(X,D) - (1/dicti['tau'])*Z,(1/dicti['tau']))
    W = soft_thresholding(D - (1/dicti['tau'])*V,(dicti['lamda']/dicti['tau']))
    #Step2
    D = np.matmul(inv_x,W + (1/dicti['tau'])*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/dicti['tau'])*Z))
    mu = np.matmul(inv_y,tau*np.matmul(Y.transpose(),P + np.matmul(X,D_old) + (1/dicti['tau'])*Z) + dicti['rho']*np.eye(Z.shape[1]))
    #Step3
    
    diff1 = P - np.matmul(Y,mu) + np.matmul(X,D)
    diff2 = W-D
    
    Z += dicti['tau']*(diff1)
    V += dicti['tau']*(diff2)
    
    r = np.sqrt(np.linalg.norm(diff1)**2 + np.linalg.norm(diff2)**2)
    s = dicti['tau']*np.sqrt(np.linalg.norm(np.matmul(X,D-D_old) - np.matmul(Y,mu-mu_old))**2 + np.linalg.norm(D-D_old)**2)
    
    if r > dicti['beta']*s:
        dicti['tau'] = dicti['tau']*dicti['tau_incr']
    elif s > dicti['beta']*r:
        dicti['tau'] = dicti['tau']/dicti['tau_decr']
    return W,mu,Z,P,D,V
                 
def admm_const(W,mu,Z,P,D,V,X,Y,dicti,inv_x,inv_y):        
    D_old = copy.deepcopy(D)
    mu_old = copy.deepcopy(mu)
    #Step1
    P = soft_thresholding(np.matmul(Y,mu) - np.matmul(X,D) - (1/dicti['tau'])*Z,(1/dicti['tau']))
    W_shape = W.shape
    W = D - (1/dicti['tau'])*V
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=dicti['eta'])
    W = W_vec.reshape(W_shape)
    #Step2
    D = np.matmul(inv_x,W + (1/dicti['tau'])*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/dicti['tau'])*Z))
    mu = np.matmul(inv_y,dicti['tau']*np.matmul(Y.transpose(),P + np.matmul(X,D_old) + (1/dicti['tau'])*Z) + dicti['rho']*np.eye(Z.shape[1]))
    #Step3
    Z += dicti['tau']*(P - np.matmul(Y,mu) + np.matmul(X,D))
    V += dicti['tau']*(W-D)
    return W,mu,Z,P,D,V

def admm_const_vp(W,mu,Z,P,D,V,X,Y,dicti,inv_x,inv_y):        
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    P_old = copy.deepcopy(P)
    D_old = copy.deepcopy(D)
    V_old = copy.deepcopy(V)
    #Step1
    P = soft_thresholding(np.matmul(Y,mu) - np.matmul(X,D) - (1/dicti['tau'])*Z,(1/dicti['tau']))
    W_shape = W.shape
    W = D - (1/dicti['tau'])*V
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=dicti['eta'])
    W = W_vec.reshape(W_shape)
    #Step2
    D = np.matmul(inv_x,W + (1/dicti['tau'])*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/dicti['tau'])*Z))
    mu = np.matmul(inv_y,dicti['tau']*np.matmul(Y.transpose(),P + np.matmul(X,D_old) + (1/dicti['tau'])*Z) + dicti['rho']*np.eye(Z.shape[1]))
    #Step3
    diff1 = P - np.matmul(Y,mu) + np.matmul(X,D)
    diff2 = W-D
    
    Z += dicti['tau']*(diff1)
    V += dicti['tau']*(diff2)
    
    r = np.sqrt(np.linalg.norm(diff1)**2 + np.linalg.norm(diff2)**2)
    s = dicti['tau']*np.sqrt(np.linalg.norm(np.matmul(X,D-D_old) - np.matmul(Y,mu-mu_old))**2 + np.linalg.norm(D-D_old)**2)
    
    if r > dicti['beta']*s:
        dicti['tau'] = dicti['tau']*dicti['tau_incr']
        print(1)
    elif s > dicti['beta']*r:
        dicti['tau'] = dicti['tau']/dicti['tau_decr']
        print(2)
    return W,mu,Z,P,D,V
                   
def admm_const_over(W,mu,Z,P,D,V,X,Y,dicti,inv_x,inv_y):        
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    P_old = copy.deepcopy(P)
    D_old = copy.deepcopy(D)
    V_old = copy.deepcopy(V)
    
    #Step1               
    P = soft_thresholding(np.matmul(Y,mu) - np.matmul(X,D) - (1/dicti['tau'])*Z,(1/dicti['tau']))
    W_shape = W.shape
    W = D - (1/dicti['tau'])*V
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=dicti['eta'])
    W = W_vec.reshape(W_shape)
    #Step2
    D = np.matmul(inv_x,W + (1/dicti['tau'])*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/dicti['tau'])*Z))
    mu = np.matmul(inv_y,dicti['tau']*np.matmul(Y.transpose(),P + np.matmul(X,D_old) + (1/dicti['tau'])*Z) + dicti['rho']*np.eye(Z.shape[1]))
    #Step3
    Z += dicti['tau']*(P - np.matmul(Y,mu) + np.matmul(X,D))
    V += dicti['tau']*(W-D)
    
    W += dicti['gamma']*(W - W_old)
    mu += dicti['gamma']*(mu - mu_old)
    P += dicti['gamma']*(P - P_old)
    D += dicti['gamma']*(D - D_old)
    Z += dicti['gamma']*(Z - Z_old)
    V += dicti['gamma']*(V - V_old)
    
    return W,mu,Z,P,D,V
                   
def admm_frob(W,mu,Z,P,D,V,X,Y,dicti,inv_x,inv_y):        
    D_old = copy.deepcopy(D)
    mu_old = copy.deepcopy(mu)
    #Step1
    P = (dicti['tau']/(dicti['tau'] + 2))*(np.matmul(Y,mu) - np.matmul(X,D) - (1/dicti['tau'])*Z)
    W_shape = W.shape
    W = D - (1/dicti['tau'])*V
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=dicti['eta'])
    W = W_vec.reshape(W_shape)
    #Step2
    D = np.matmul(inv_x,W + (1/dicti['tau'])*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/dicti['tau'])*Z))
    mu = np.matmul(inv_y,dicti['tau']*np.matmul(Y.transpose(),P + np.matmul(X,D_old) + (1/dicti['tau'])*Z) + dicti['rho']*np.eye(Z.shape[1]))
    #Step3
    Z += dicti['tau']*(P - np.matmul(Y,mu) + np.matmul(X,D))
    V += dicti['tau']*(W-D)
    return W,mu,Z,P,D,V

def admm_frob_vp(W,mu,Z,P,D,V,X,Y,dicti,inv_x,inv_y):        
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    P_old = copy.deepcopy(P)
    D_old = copy.deepcopy(D)
    V_old = copy.deepcopy(V)
    #Step1
    P = (dicti['tau']/(dicti['tau'] + 2))*(np.matmul(Y,mu) - np.matmul(X,D) - (1/dicti['tau'])*Z)
    W_shape = W.shape
    W = D - (1/dicti['tau'])*V
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=dicti['eta'])
    W = W_vec.reshape(W_shape)
    #Step2
    D = np.matmul(inv_x,W + (1/dicti['tau'])*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/dicti['tau'])*Z))
    mu = np.matmul(inv_y,dicti['tau']*np.matmul(Y.transpose(),P + np.matmul(X,D_old) + (1/dicti['tau'])*Z) + dicti['rho']*np.eye(Z.shape[1]))
    #Step3
    diff1 = P - np.matmul(Y,mu) + np.matmul(X,D)
    diff2 = W-D
    
    Z += dicti['tau']*(diff1)
    V += dicti['tau']*(diff2)
    
    r = np.sqrt(np.linalg.norm(diff1)**2 + np.linalg.norm(diff2)**2)
    s = dicti['tau']*np.sqrt(np.linalg.norm(np.matmul(X,D-D_old) - np.matmul(Y,mu-mu_old))**2 + np.linalg.norm(D-D_old)**2)
    
    if r > dicti['beta']*s:
        dicti['tau'] = dicti['tau']*dicti['tau_incr']
    elif s > dicti['beta']*r:
        dicti['tau'] = dicti['tau']/dicti['tau_decr']
    return W,mu,Z,P,D,V
                   
def admm_frob_over(W,mu,Z,P,D,V,X,Y,dicti,inv_x,inv_y):        
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    P_old = copy.deepcopy(P)
    D_old = copy.deepcopy(D)
    V_old = copy.deepcopy(V)

    #Step1
    P = (dicti['tau']/(dicti['tau'] + 2))*(np.matmul(Y,mu) - np.matmul(X,D) - (1/dicti['tau'])*Z)
    W_shape = W.shape
    W = D - (1/dicti['tau'])*V
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=dicti['eta'])
    W = W_vec.reshape(W_shape)
    #Step2
    D = np.matmul(inv_x,W + (1/dicti['tau'])*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/dicti['tau'])*Z))
    mu = np.matmul(inv_y,dicti['tau']*np.matmul(Y.transpose(),P + np.matmul(X,D_old) + (1/dicti['tau'])*Z) + dicti['rho']*np.eye(Z.shape[1]))
    #Step3
    Z += dicti['tau']*(P - np.matmul(Y,mu) + np.matmul(X,D))
    V += dicti['tau']*(W-D)
                   
    W += dicti['gamma']*(W - W_old)
    mu += dicti['gamma']*(mu - mu_old)
    Z += dicti['gamma']*(Z - Z_old)
    P += dicti['gamma']*(P - P_old)
    D += dicti['gamma']*(D - D_old)
    V += dicti['gamma']*(V - V_old)
                   
    return W,mu,Z,P,D,V
                   
def admm_elastic(W,mu,Z,P,D,V,X,Y,dicti,inv_x,inv_y):
    D_old = copy.deepcopy(D)
    mu_old = copy.deepcopy(mu)
    #Step1
    P = soft_thresholding(np.matmul(Y,mu) - np.matmul(X,D) - (1/dicti['tau'])*Z,(1/dicti['tau']))
    W_shape = W.shape
    W = (dicti['tau']/(dicti['tau'] + dicti['alpha']))*(D - (1/dicti['tau'])*V)
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=dicti['eta'])
    W = W_vec.reshape(W_shape)
    #Step2
    D = np.matmul(inv_x,W + (1/dicti['tau'])*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/dicti['tau'])*Z))
    mu = np.matmul(inv_y,dicti['tau']*np.matmul(Y.transpose(),P + np.matmul(X,D_old) + (1/dicti['tau'])*Z) + dicti['rho']*np.eye(Z.shape[1]))
    #Step3
    Z += dicti['tau']*(P - np.matmul(Y,mu) + np.matmul(X,D))
    V += dicti['tau']*(W-D)
    return W,mu,Z,P,D,V

def admm_elastic_vp(W,mu,Z,P,D,V,X,Y,dicti,inv_x,inv_y):
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    P_old = copy.deepcopy(P)
    D_old = copy.deepcopy(D)
    V_old = copy.deepcopy(V)
    #Step1
    P = soft_thresholding(np.matmul(Y,mu) - np.matmul(X,D) - (1/dicti['tau'])*Z,(1/dicti['tau']))
    W_shape = W.shape
    W = (dicti['tau']/(dicti['tau'] + alpha))*(D - (1/dicti['tau'])*V)
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=dicti['eta'])
    W = W_vec.reshape(W_shape)
    #Step2
    D = np.matmul(inv_x,W + (1/dicti['tau'])*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/dicti['tau'])*Z))
    mu = np.matmul(inv_y,dicti['tau']*np.matmul(Y.transpose(),P + np.matmul(X,D_old) + (1/dicti['tau'])*Z) + dicti['rho']*np.eye(Z.shape[1]))
    #Step3
    diff1 = P - np.matmul(Y,mu) + np.matmul(X,D)
    diff2 = W-D
    
    Z += dicti['tau']*(diff1)
    V += dicti['tau']*(diff2)
    
    r = np.sqrt(np.linalg.norm(diff1)**2 + np.linalg.norm(diff2)**2)
    s = dicti['tau']*np.sqrt(np.linalg.norm(np.matmul(X,D-D_old) - np.matmul(Y,mu-mu_old))**2 + np.linalg.norm(D-D_old)**2)
    
    if r > dicti['beta']*s:
        dicti['tau'] = dicti['tau']*dicti['tau_incr']
    elif s > dicti['beta']*r:
        dicti['tau'] = dicti['tau']/dicti['tau_decr']
    return W,mu,Z,P,D,V

def admm_elastic_var(W,mu,Z,P,D,V,X,Y,dicti,inv_x,inv_y):
    D_old = copy.deepcopy(D)
    mu_old = copy.deepcopy(mu)
    dicti['tau']*=dicti['theta']
    #Step1
    P = soft_thresholding(np.matmul(Y,mu) - np.matmul(X,D) - (1/dicti['tau'])*Z,(1/dicti['tau']))
    W_shape = W.shape
    W = (dicti['tau']/(dicti['tau'] + alpha))*(D - (1/dicti['tau'])*V)
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=dicti['eta'])
    W = W_vec.reshape(W_shape)
    #Step2
    D = np.matmul(inv_x,W + (1/dicti['tau'])*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/dicti['tau'])*Z))
    mu = np.matmul(inv_y,dicti['tau']*np.matmul(Y.transpose(),P + np.matmul(X,D_old) + (1/dicti['tau'])*Z) + dicti['rho']*np.eye(Z.shape[1]))
    #Step3
    dicti['theta'] = 1/np.sqrt(1 + dicti['tau'])
    mu_ = mu + dicti['theta']*(mu - mu_old)
    D_ = D + dicti['theta']*(D - D_old)
    Z += dicti['tau']*(P - np.matmul(Y,mu) + np.matmul(X,D))
    V += dicti['tau']*(W-D)
    return W,mu,Z,P,D,V

def cvx_unconst(W,mu,Z,P,D,V,X,Y,dicti,inv_x,inv_y):
    W = cp.Variable(W.shape)
    mu = cp.Variable(mu.shape)
    time1 = time.time()
    objective = cp.Minimize(cp.norm(Y*mu - X*W,1) + 0.5*dicti['rho']*(mu - np.eye(mu.shape)) + dicti['lamda']*cp.norm(W,1))
    constraints = []
    prob = cp.Problem(objective, constraints)
    result = prob.solve()
    time2 = time.time()
    time_taken = time2-time1
    
    t1,t2,t3,t4,t5,t6,t7,lx = get_loss(W,mu,Z,P,D,V,X,Y,train)
    acc = get_accuracy(W.value,mu.value,Z,X,Y)

    #Store train and test performance
    print(f"{label}: [{epoch}] ||Y\u03BC-XW||_1: {t1:.2f} ||W||_1: {t2:.2f} ||\u03BC-I||_F: {t3:.2f} ||Y\u03BC-XW||_F: {t7:.2f} loss: {lx:.2f} acc: {acc:.2f}")    
    logger.append('train' if train else 'test', epoch=epoch,t1=t1,t2=t2,t3=t3,t7=t7,loss=lx,acc=acc,time=t2-t1)
        
def sensitivity(dicti=None):
    mid_value = dicti[dicti['param']]
    initial_log_name = dicti['log_name']
    
    if dicti['log_scale']:iter_range = np.logspace(np.log10(mid_value*0.1),np.log10(mid_value*10),10)
    else: iter_range = np.linspace(mid_value-dicti['sen_range'],mid_value+dicti['sen_range'],10)
    
    for i,val in enumerate(iter_range):
        dicti[dicti['param']]=val
        dicti['log_name']=initial_log_name+"_sen_{0}_{1}".format(dicti['param'],i)
        run(dicti=dicti)
        
    fig, (ax1) = plt.subplots(1, 1, figsize=(1 + 1*3., 2.5), sharey=False)
    fig_name = initial_log_name+'_{0}_{1}'.format('sen',dicti['param'])
    x,y = plot_sen(ax1,experiments=[fig_name],parameter=dicti['param'],title='Acc')
    #ax2 = plot_sen(ax2,experiments=[fig_name],parameter=dicti['param'],title='Loss')
    fig.tight_layout()
    if dicti['save_fig']:
        fig.savefig('{0}{1}{2}'.format('Plots/',fig_name,'.pdf'),format='pdf', bbox_inches='tight')
    plt.show()
        
def run(dicti=None):
    
    np.random.seed(0)
    
    mkdir('logs')
    logger = Logger(index=dicti['log_name'])
    logger['args'] = dicti
    
    print("[Logging in {}]".format(logger.index))
    
    #Select the algorithm
    if dicti['algorithm'] == 'pd_soft':method = pd_soft
    elif dicti['algorithm'] == 'pd_const':method = pd_const
    elif dicti['algorithm'] == 'pd_const_over':method = pd_const_over
    elif dicti['algorithm'] == 'pd_frob':method = pd_frob
    elif dicti['algorithm'] == 'pd_frob_over':method = pd_frob_over
    elif dicti['algorithm'] == 'pd_elastic':method = pd_elastic
    elif dicti['algorithm'] == 'admm_soft':method = admm_soft
    elif dicti['algorithm'] == 'admm_soft_vp':method = admm_soft_vp
    elif dicti['algorithm'] == 'admm_const':method = admm_const
    elif dicti['algorithm'] == 'admm_const_vp':method = admm_const_vp
    elif dicti['algorithm'] == 'admm_const_over':method = admm_const_over
    elif dicti['algorithm'] == 'admm_frob':method = admm_frob
    elif dicti['algorithm'] == 'admm_frob_vp':method = admm_frob_vp
    elif dicti['algorithm'] == 'admm_frob_over':method = admm_frob_over
    elif dicti['algorithm'] == 'admm_elastic':method = admm_elastic
    elif dicti['algorithm'] == 'admm_elastic_vp':method = admm_elastic_vp
    else: raise ValueError("Invalid method '{}'".format(dicti['algorithm']))
        
    #Select the dataset
    if dicti['dataset'] == 'MNIST':ds = get_mnist
    elif dicti['dataset'] == 'Digits':ds = get_digits
    elif dicti['dataset'] == 'Synthetic':ds = get_syn
    elif dicti['dataset'] == 'rna':ds = get_rna
    else: raise ValueError("Invalid method '{}'".format(dicti['dataset']))
    X_train,Y_train,X_test,Y_test = ds(train_size=dicti['train_size'],\
                                       test_size=dicti['test_size'],classes=dicti['classes'])
    print('Data loaded')
    
    nrm_type = 'eig'
    #Normalize the data
    X_train/=operator_norm(X_train,nrm_type)
    X_test/=operator_norm(X_test,nrm_type)
    print('Data normalized')
    
    #Initialize the variables
    W_shape = (X_train.shape[1],Y_train.shape[1])
    mu_shape = (Y_train.shape[1],Y_train.shape[1])
    Z_shape = Y_train.shape
    P_shape = Z_shape
    D_shape = W_shape
    V_shape = W_shape
    W,mu,Z,P,D,V = initialize_params(W_shape,mu_shape,Z_shape,P_shape,D_shape,V_shape)
    print('Input initialized')
    
    #Tune the hyper-parameters
    #if dicti['sigma'] == 0:
    dicti['sigma'] = get_sigma(Y_train,X_train,dicti['tau'],dicti['tau_mu'],\
                                dicti['rho'],dicti['gamma'],'over' in dicti['algorithm'])
    print('Parameters Set')

    #For admm
    inv_x = None
    inv_y = None
    if 'admm' in dicti['algorithm']:
        inv_x = np.linalg.inv(np.matmul(X_train.transpose(),X_train) + np.eye(X_train.shape[1]))
        inv_y = np.linalg.inv(dicti['tau']*np.matmul(Y_train.transpose(),Y_train) + dicti['rho']*np.eye(Y_train.shape[1])) 
                   
    total_training_time = []
    try:
        t1 = time.time()
        for epoch in range(dicti['niter']):
            
            W,mu,Z,P,D,V = train(epoch,W,mu,Z,P,D,V,X_train,Y_train,dicti,method,logger,inv_x,inv_y)

            test(epoch,W,mu,Z,P,D,V,X_test,Y_test,dicti,method,logger,inv_x,inv_y)
        t2 = time.time()
        logger['finished'] = True
    
        print('Tau : ',dicti['tau'])
        print('Tau mu: ',dicti['tau_mu'])
        print('Sigma: ',dicti['sigma'])
        print('Rho: ',dicti['rho'])
        print('Eta: ',dicti['eta'])
        print('Lamda: ',dicti['lamda'])
        print('Total training time',t2-t1)
        logger = Logger.load(dicti['log_name'])
        y_curr = [v['acc'] for v in logger.get('train')]
        print('Train Accuracy :', np.mean(y_curr[-5:]))
        logger = Logger.load(dicti['log_name'])
        y_curr = [v['acc'] for v in logger.get('test')]
        print('Test Accuracy :', np.mean(y_curr[-5:]))
        w_rows = np.linalg.norm(W,1,axis=1)
        w_rows /= np.max(w_rows)
        print('Feature selection:',1-np.sum(w_rows<0.25)/len(w_rows))
        print('W shape',W.shape)
        
        if not dicti['sen']:
            param_dict = {}
            param_dict['mu'] = mu
            param_dict['W'] = W
            if dicti['save']:
                np.save('{0}{1}{2}'.format('models/',dicti['log_name'],'.npy'),param_dict)
            plot_exp(dicti,W)
    
    except KeyboardInterrupt:
        print("Run interrupted")
        logger.append('interrupt', epoch=epoch)
    print("[Logs in {}]".format(logger.index))
    

if __name__ == '__main__':
    
    dicti = get_dictionary()
    if args.sen:
        sensitivity(dicti=dicti)
    else:   
        run(dicti=dicti)
    