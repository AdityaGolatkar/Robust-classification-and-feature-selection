#!/usr/bin/env python3

import argparse
import copy
import os
import shutil
import time
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

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
from dataloader import *

parser = argparse.ArgumentParser(description='Large-Scale-Optimization')
parser.add_argument('--niter',default=200, type=int, help='Number of iterations')
parser.add_argument('--lamda', default=1, type=float, help='Lambda')
parser.add_argument('--eta', default=100, type=float, help='Eta')
parser.add_argument('--rho', default=0.0001, type=float, help='Rho')
parser.add_argument('--gamma', default=1, type=float, help='Gamma')
parser.add_argument('--alpha', default=1, type=float, help='Alpha')
parser.add_argument('--tau', default=0.01, type=float, help='Tau')
parser.add_argument('--tau_mu', default=0.1, type=float, help='Tau mu')
parser.add_argument('--sigma', default=0, type=float, help='Sigma')
parser.add_argument('--dataset',default='digits',type=str, help='Dataset')
parser.add_argument('--algorithm',default='pd_const',type=str, help='Algorithm')
parser.add_argument('--log-name',default=None,type=str, help='Log-name')
parser.add_argument('--train-size',default=100,type=int,help='Size of training set')
parser.add_argument('--test-size',default=100,type=int,help='Size of testing set')
parser.add_argument('--classes',default=10,type=int,help='Number of classes')
parser.add_argument('--batch-size',default=200,type=int,help='Batch-size')
parser.add_argument('--sen', action='store_true',help='Sensitivity')
parser.add_argument('--save', action='store_true',help='Save model')
parser.add_argument('--save_fig', action='store_true',help='Save figure')
parser.add_argument('--residual', action='store_true',help='Compute and save residual')
parser.add_argument('--param',default='tau',type=str, help='Chosse which parameter')
args = parser.parse_args()

def get_dictionary():
    dicti = {}
    dicti['niter']=args.niter
    dicti['lamda']=args.lamda
    dicti['eta']=args.eta
    dicti['rho']=args.rho
    dicti['gamma']=args.gamma
    dicti['alpha']=args.alpha
    dicti['tau']=args.tau
    dicti['tau_mu']=args.tau_mu
    dicti['sigma']=args.sigma
    dicti['dataset']=args.dataset
    dicti['algorithm']=args.algorithm
    dicti['log_name']=args.log_name
    dicti['train_size']=args.train_size
    dicti['test_size']=args.test_size
    dicti['classes']=args.classes
    dicti['sen']=args.sen
    dicti['save']=args.save
    dicti['save_fig']=args.save_fig
    dicti['param']=args.param
    dicti['batch_size']=args.batch_size
    return dicti

def plot_exp(dicti,W):
    if 'admm' in dicti['algorithm'] or 'alt' in dicti['algorithm']:
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(1 + 4*3., 2.5), sharey=False)
        ax1 = plot_basic(ax1,experiments=[dicti['log_name']],key1=['acc','acc'],key2=['train','test'],labels=['Train Acc.','Test Acc.'],ylabel='Accuracy',title='acc')
        ax2 = plot_basic(ax2,experiments=[dicti['log_name']],key1=['t7','t1'],key2=['train','train'],labels=['Frobenius','L1'],ylabel='Cost',title='loss')
        ax3 = plot_basic(ax3,experiments=[dicti['log_name']],key1=['t5','t6'],key2=['train','train'],labels=['||P-Y\u03BC+XD||_F','||W-D||_F'],ylabel='Primal Residue',title='p_residue')
        ax4 = plot_w_distri(ax4,W)
        fig.tight_layout()
        if dicti['save_fig']:
            fig.savefig('{0}{1}_{2}{3}'.format('Plots/',dicti['log_name'],'LxAc','.pdf'),format='pdf', bbox_inches='tight')
        plt.show()
            
    elif 'pd' in dicti['algorithm']:
        fig, (ax1,ax2,ax3,ax4) = plt.subplots(1, 4, figsize=(1 + 4*3., 2.5), sharey=False)
        ax1 = plot_basic(ax1,experiments=[dicti['log_name']],key1=['acc','acc'],key2=['train','test'],labels=['Train Acc.','Test Acc.'],ylabel='Accuracy',title='acc')
        ax2 = plot_basic(ax2,experiments=[dicti['log_name']],key1=['t7','t1'],key2=['train','train'],labels=['Frobenius','L1'],title='loss')
        ax3 = plot_basic(ax3,experiments=[dicti['log_name']],key1=['t7','t1'],key2=['train','train'],labels=['Frobenius','L1'],ylabel='Cost',title='losst')
        ax4 = plot_w_distri(ax4,W)
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

def get_loss(W,mu,Z,P,D,V,X,Y,lamda,rho,train):
    
    term1 = np.linalg.norm(np.matmul(Y,mu)-np.matmul(X,W),1)
    term2 = (np.sum(np.abs(W)))
    term3 = np.sum((np.eye(mu.shape[0])-mu)**2)
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
    elif 'admm' in args.algorithm or 'alt' in args.algorithm:
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

def train(epoch,W,mu,Z,P,D,V,train_loader,sigma,tau,tau_mu,lamda,rho,gamma,eta,alpha,algorithm,logger,train=True):
    
    t1 = AverageMeter()
    t2 = AverageMeter()
    t3 = AverageMeter()
    t4 = AverageMeter()
    t5 = AverageMeter()
    t6 = AverageMeter()
    t7 = AverageMeter()
    lx = AverageMeter()
    acc = AverageMeter()
    ti = AverageMeter()

    label='testt'
    if train:label='train'    
    
    for i, (input, target) in enumerate(train_loader):
        X = input.reshape(input.shape[0],-1).numpy()
        X/=operator_norm(X,'eig')
        Y = get_Y(target.numpy(),args.classes)
        if train:
            dicti['sigma'] = get_sigma(Y,X,dicti['tau'],dicti['tau_mu'],\
                                dicti['rho'],dicti['gamma'],'over' in dicti['algorithm'])
            inv_x = None
            inv_y = None
            if 'admm' in dicti['algorithm']:
                inv_x = np.linalg.inv(np.matmul(X.transpose(),X) + np.eye(X.shape[1]))
                inv_y = np.linalg.inv(dicti['tau']*np.matmul(Y.transpose(),Y) + dicti['rho']*np.eye(Y.shape[1])) 
        
            ti1 = time.time()
            W,mu,Z,P,D,V = algorithm(W,mu,Z,P,D,V,X,Y,sigma,tau,tau_mu,lamda,rho,gamma,eta,alpha,inv_x,inv_y)
            ti2 = time.time()
        
        te1,te2,te3,te4,te5,te6,te7,lsx = get_loss(W,mu,Z,P,D,V,X,Y,lamda,rho,train)
        accy = get_accuracy(W,mu,Z,X,Y)
        
        X_shape_0 = X.shape[0]
        t1.update(te1, X_shape_0)
        t2.update(te2, X_shape_0)
        t3.update(te3, X_shape_0)
        t4.update(te4, X_shape_0)
        t5.update(te5, X_shape_0)
        t6.update(te6, X_shape_0)
        t7.update(te7, X_shape_0)
        lx.update(lsx, X_shape_0)
        acc.update(accy, X_shape_0)
        if train:
            ti.update(ti2-ti1,1)
        
    #Store train and test performance
    if 'pd' in args.algorithm:
        print(f"{label}: [{epoch}] ||Y\u03BC-XW||_1: {t1.avg:.2f} ||W||_1: {t2.avg:.2f} ||\u03BC-I||_F: {t3.avg:.2f} ||Y\u03BC-XW||_F: {t7.avg:.2f} loss: {lx.avg:.2f} acc: {acc.avg:.2f}")    
        logger.append('train' if train else 'test', epoch=epoch,t1=t1.avg,t2=t2.avg,t3=t3.avg,t7=t7.avg,loss=lx.avg,acc=acc.avg,time=ti.avg)
    elif 'admm' in args.algorithm or 'alt' in args.algorithm:
        print(f"{label}: [{epoch}] ||Y\u03BC-XW||_1: {t1.avg:.2f} ||W||_1: {t2.avg:.2f} ||\u03BC-I||_F: {t3.avg:.2f} ||P||_1: {t4.avg:.2f} ||P-Y\u03BC+XD||_F: {t5.avg:.2f} ||W-D||_F: {t6.avg:.2f} ||Y\u03BC-XW||_F: {t7.avg:.2f} loss: {lx.avg:.2f} acc: {acc.avg:.2f}")    
        logger.append('train' if train else 'test', epoch=epoch,t1=t1.avg,t2=t2.avg,t3=t3.avg,t4=t4.avg,t5=t5.avg,t6=t6.avg,t7=t7.avg,loss=lx.avg,\
                      acc=acc.avg,time=ti.avg)

    if not train:print('')
    return W,mu,Z,P,D,V

def test(epoch,W,mu,Z,P,D,V,train_loader,sigma,tau,tau_mu,lamda,rho,gamma,eta,alpha,algorithm,logger):
    _,_,_,_,_,_ = train(epoch,W,mu,Z,P,D,V,train_loader,sigma,tau,tau_mu,lamda,rho,gamma,\
                  eta,alpha,algorithm,logger,train=False)
    
def soft_thresholding(W,lamda):
    W = pywt.threshold(W, lamda, mode='soft', substitute=0)
    return W
    
def pd_soft(W,mu,Z,P,D,V,X,Y,sigma,tau,tau_mu,lamda,rho,gamma,eta,alpha,inv_x,inv_y):
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    
    W += tau*(np.matmul(X.transpose(),Z))
    W = soft_thresholding(W,lamda)
    mu = (1/(1+tau*rho))*(mu + rho*tau_mu*np.eye(mu.shape[0]) - tau_mu*(np.matmul(Y.transpose(),Z)))
    Z = np.maximum(-1,(np.minimum(1,Z + sigma*(np.matmul(Y,2*mu - mu_old) - np.matmul(X,2*W - W_old)))))
    return W,mu,Z,P,D,V

def pd_const(W,mu,Z,P,D,V,X,Y,sigma,tau,tau_mu,lamda,rho,gamma,eta,alpha,inv_x,inv_y):
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    
    W_shape = W.shape
    W += tau*(np.matmul(X.transpose(),Z))
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=eta)
    W = W_vec.reshape(W_shape)
    mu = (1/(1+tau_mu*rho))*(mu_old + rho*tau_mu*np.eye(mu.shape[0]) - tau_mu*(np.matmul(Y.transpose(),Z)))
    Z = np.maximum(-1,(np.minimum(1,Z + sigma*(np.matmul(Y,2*mu - mu_old) - np.matmul(X,2*W - W_old)))))
    return W,mu,Z,P,D,V

def pd_const_over(W,mu,Z,P,D,V,X,Y,sigma,tau,tau_mu,lamda,rho,gamma,eta,alpha,inv_x,inv_y):
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    
    W_shape = W.shape
    W += tau*(np.matmul(X.transpose(),Z))
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=eta)
    W = W_vec.reshape(W_shape)
    mu = (1/(1+tau*rho))*(mu + rho*tau_mu*np.eye(mu.shape[0]) - tau_mu*(np.matmul(Y.transpose(),Z)))
    Z = np.maximum(-1,(np.minimum(1,Z + sigma*(np.matmul(Y,2*mu - mu_old) - np.matmul(X,2*W - W_old)))))
    
    W += gamma*(W - W_old)
    mu += gamma*(mu - mu_old)
    Z += gamma*(Z - Z_old)
    return W,mu,Z,P,D,V

def pd_frob(W,mu,Z,P,D,V,X,Y,sigma,tau,tau_mu,lamda,rho,gamma,eta,alpha,inv_x,inv_y):
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    
    W_shape = W.shape
    W += tau*(np.matmul(X.transpose(),Z))
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=eta)
    W = W_vec.reshape(W_shape)
    mu = (1/(1+tau*rho))*(mu_old + rho*tau_mu*np.eye(mu.shape[0]) - tau_mu*(np.matmul(Y.transpose(),Z)))
    Z += sigma*(np.matmul(Y,2*mu - mu_old) - np.matmul(X,2*W - W_old))
    Z = Z/(np.maximum(1,np.linalg.norm(Z,'fro')))
    return W,mu,Z,P,D,V

def pd_frob_over(W,mu,Z,P,D,V,X,Y,sigma,tau,tau_mu,lamda,rho,gamma,eta,alpha,inv_x,inv_y):
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    
    W_shape = W.shape
    W += tau*(np.matmul(X.transpose(),Z))
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=eta)
    W = W_vec.reshape(W_shape)
    mu = (1/(1+tau*rho))*(mu + rho*tau_mu*np.eye(mu.shape[0]) - tau_mu*(np.matmul(Y.transpose(),Z)))
    Z += sigma*(np.matmul(Y,2*mu - mu_old) - np.matmul(X,2*W - W_old))
    Z = Z/(np.maximum(1,np.linalg.norm(Z,'fro')))
    
    W += gamma*(W - W_old)
    mu += gamma*(mu - mu_old)
    Z += gamma*(Z - Z_old)
    return W,mu,Z,P,D,V

def pd_elastic(W,mu,Z,P,D,V,X,Y,sigma,tau,tau_mu,lamda,rho,gamma,eta,alpha,inv_x,inv_y):
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    
    W_shape = W.shape
    W = (1/(1+tau*alpha))*(W + tau*(np.matmul(X.transpose(),Z)))
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=eta)
    W = W_vec.reshape(W_shape)
    mu = (1/(1+tau*rho))*(mu_old + rho*tau_mu*np.eye(mu.shape[0]) - tau_mu*(np.matmul(Y.transpose(),Z)))
    Z = np.maximum(-1,(np.minimum(1,Z + sigma*(np.matmul(Y,2*mu - mu_old) - np.matmul(X,2*W - W_old)))))
    return W,mu,Z,P,D,V

def admm_soft(W,mu,Z,P,D,V,X,Y,sigma,tau,tau_mu,lamda,rho,gamma,eta,alpha,inv_x,inv_y):        
    D_old = copy.deepcopy(D)
    #Step1
    P = soft_thresholding(np.matmul(Y,mu) - np.matmul(X,D) - (1/tau)*Z,(1/tau))
    W = soft_thresholding(D - (1/tau)*V,(lamda/tau))
    #Step2
    D = np.matmul(inv_x,W + (1/tau)*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/tau)*Z))
    mu = np.matmul(inv_y,tau*np.matmul(Y.transpose(),P + np.matmul(X,D_old) + (1/tau)*Z) + rho*np.eye(Z.shape[1]))
    #Step3
    Z += tau*(P - np.matmul(Y,mu) + np.matmul(X,D))
    V += tau*(W-D)
    return W,mu,Z,P,D,V
                 
def admm_const(W,mu,Z,P,D,V,X,Y,sigma,tau,tau_mu,lamda,rho,gamma,eta,alpha,inv_x,inv_y):        
    D_old = copy.deepcopy(D)
    mu_old = copy.deepcopy(mu)
    #Step1
    P = soft_thresholding(np.matmul(Y,mu) - np.matmul(X,D) - (1/tau)*Z,(1/tau))
    W_shape = W.shape
    W = D - (1/tau)*V
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=eta)
    W = W_vec.reshape(W_shape)
    #Step2
    D = np.matmul(inv_x,W + (1/tau)*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/tau)*Z))
    mu = np.matmul(inv_y,tau*np.matmul(Y.transpose(),P + np.matmul(X,D_old) + (1/tau)*Z) + rho*np.eye(Z.shape[1]))
    #Step3
    Z += tau*(P - np.matmul(Y,mu) + np.matmul(X,D))
    V += tau*(W-D)
    return W,mu,Z,P,D,V
                   
def admm_const_over(W,mu,Z,P,D,V,X,Y,sigma,tau,tau_mu,lamda,rho,gamma,eta,alpha,inv_x,inv_y):        
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    P_old = copy.deepcopy(P)
    D_old = copy.deepcopy(D)
    V_old = copy.deepcopy(V)
    
    #Step1               
    P = soft_thresholding(np.matmul(Y,mu) - np.matmul(X,D) - (1/tau)*Z,(1/tau))
    W_shape = W.shape
    W = D - (1/tau)*V
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=eta)
    W = W_vec.reshape(W_shape)
    #Step2
    D = np.matmul(inv_x,W + (1/tau)*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/tau)*Z))
    mu = np.matmul(inv_y,tau*np.matmul(Y.transpose(),P + np.matmul(X,D_old) + (1/tau)*Z) + rho*np.eye(Z.shape[1]))
    #Step3
    Z += tau*(P - np.matmul(Y,mu) + np.matmul(X,D))
    V += tau*(W-D)
    
    W += gamma*(W - W_old)
    mu += gamma*(mu - mu_old)
    Z += gamma*(Z - Z_old)
    P += gamma*(P - P_old)
    D += gamma*(D - D_old)
    V += gamma*(V - V_old)
    
    return W,mu,Z,P,D,V
                   
def admm_frob(W,mu,Z,P,D,V,X,Y,sigma,tau,tau_mu,lamda,rho,gamma,eta,alpha,inv_x,inv_y):        
    D_old = copy.deepcopy(D)
    mu_old = copy.deepcopy(mu)
    #Step1
    P = (tau/(tau + 2))*(np.matmul(Y,mu) - np.matmul(X,D) - (1/tau)*Z)
    W_shape = W.shape
    W = D - (1/tau)*V
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=eta)
    W = W_vec.reshape(W_shape)
    #Step2
    D = np.matmul(inv_x,W + (1/tau)*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/tau)*Z))
    mu = np.matmul(inv_y,tau*np.matmul(Y.transpose(),P + np.matmul(X,D_old) + (1/tau)*Z) + rho*np.eye(Z.shape[1]))
    #Step3
    Z += tau*(P - np.matmul(Y,mu) + np.matmul(X,D))
    V += tau*(W-D)
    return W,mu,Z,P,D,V
                   
def admm_frob_over(W,mu,Z,P,D,V,X,Y,sigma,tau,tau_mu,lamda,rho,gamma,eta,alpha,inv_x,inv_y):        
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    P_old = copy.deepcopy(P)
    D_old = copy.deepcopy(D)
    V_old = copy.deepcopy(V)

    #Step1
    P = (tau/(tau + 2))*(np.matmul(Y,mu) - np.matmul(X,D) - (1/tau)*Z)
    W_shape = W.shape
    W = D - (1/tau)*V
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=eta)
    W = W_vec.reshape(W_shape)
    #Step2
    D = np.matmul(inv_x,W + (1/tau)*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/tau)*Z))
    mu = np.matmul(inv_y,tau*np.matmul(Y.transpose(),P + np.matmul(X,D_old) + (1/tau)*Z) + rho*np.eye(Z.shape[1]))
    #Step3
    Z += tau*(P - np.matmul(Y,mu) + np.matmul(X,D))
    V += tau*(W-D)
                   
    W += gamma*(W - W_old)
    mu += gamma*(mu - mu_old)
    Z += gamma*(Z - Z_old)
    P += gamma*(P - P_old)
    D += gamma*(D - D_old)
    V += gamma*(V - V_old)
                   
    return W,mu,Z,P,D,V
                   
def admm_elastic(W,mu,Z,P,D,V,X,Y,sigma,tau,tau_mu,lamda,rho,gamma,eta,alpha,inv_x,inv_y):
    D_old = copy.deepcopy(D)
    mu_old = copy.deepcopy(mu)
    #Step1
    P = soft_thresholding(np.matmul(Y,mu) - np.matmul(X,D) - (1/tau)*Z,(1/tau))
    W_shape = W.shape
    W = (tau/(tau + alpha))*(D - (1/tau)*V)
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=eta)
    W = W_vec.reshape(W_shape)
    #Step2
    D = np.matmul(inv_x,W + (1/tau)*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/tau)*Z))
    mu = np.matmul(inv_y,tau*np.matmul(Y.transpose(),P + np.matmul(X,D_old) + (1/tau)*Z) + rho*np.eye(Z.shape[1]))
    #Step3
    Z += tau*(P - np.matmul(Y,mu) + np.matmul(X,D))
    V += tau*(W-D)
    return W,mu,Z,P,D,V

def alt_soft(W,mu,Z,P,D,V,X,Y,sigma,tau,tau_mu,lamda,rho,gamma,eta,alpha,inv_x,inv_y):        
    #Step1
    mu = (1/rho)*np.matmul(Y.transpose(),Z) + np.eye(Y.shape[1])
    #Step2
    P = soft_thresholding(np.matmul(Y,mu) - np.matmul(X,D) - (1/tau)*Z,(1/tau))
    W = soft_thresholding(D - (1/tau)*V,(lamda/tau))
    D = np.matmul(inv_x,W + (1/tau)*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/tau)*Z))
    #Step3
    Z += tau*(P - np.matmul(Y,mu) + np.matmul(X,D))
    V += tau*(W-D)
    return W,mu,Z,P,D,V

def alt_const(W,mu,Z,P,D,V,X,Y,sigma,tau,tau_mu,lamda,rho,gamma,eta,alpha,inv_x,inv_y):        
    #Step1
    mu = (1/rho)*np.matmul(Y.transpose(),Z) + np.eye(Y.shape[1])
    print(mu)
    print(Z)
    import pdb;pdb.set_trace()
    #Step2
    P = soft_thresholding(np.matmul(Y,mu) - np.matmul(X,D) - (1/tau)*Z,(1/tau))
    W_shape = W.shape
    W = D - (1/tau)*V
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=eta)
    W = W_vec.reshape(W_shape)
    D = np.matmul(inv_x,W + (1/tau)*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/tau)*Z))
    #Step3
    Z += tau*(P - np.matmul(Y,mu) + np.matmul(X,D))
    V += tau*(W-D)
    return W,mu,Z,P,D,V
                   
def alt_const_over(W,mu,Z,P,D,V,X,Y,sigma,tau,tau_mu,lamda,rho,gamma,eta,alpha,inv_x,inv_y):        
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    P_old = copy.deepcopy(P)
    D_old = copy.deepcopy(D)
    V_old = copy.deepcopy(V)
    
    #Step1
    mu = (1/rho)*np.matmul(Y.transpose(),Z) + np.eye(Y.shape[1])
    #Step2               
    P = soft_thresholding(np.matmul(Y,mu) - np.matmul(X,D) - (1/tau)*Z,(1/tau))
    W_shape = W.shape
    W = D - (1/tau)*V
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=eta)
    W = W_vec.reshape(W_shape)
    D = np.matmul(inv_x,W + (1/tau)*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/tau)*Z))
    #Step3
    Z += tau*(P - np.matmul(Y,mu) + np.matmul(X,D))
    V += tau*(W-D)
    
    W += gamma*(W - W_old)
    mu += gamma*(mu - mu_old)
    Z += gamma*(Z - Z_old)
    P += gamma*(P - P_old)
    D += gamma*(D - D_old)
    V += gamma*(V - V_old)
    
    return W,mu,Z,P,D,V
                   
def alt_frob(W,mu,Z,P,D,V,X,Y,sigma,tau,tau_mu,lamda,rho,gamma,eta,alpha,inv_x,inv_y):
    #Step1
    mu = (1/rho)*np.matmul(Y.transpose(),Z) + np.eye(Y.shape[1])
    #Step2
    P = (tau/(tau + 2))*(np.matmul(Y,mu) - np.matmul(X,D) - (1/tau)*Z)
    W_shape = W.shape
    W = D - (1/tau)*V
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=eta)
    W = W_vec.reshape(W_shape)
    D = np.matmul(inv_x,W + (1/tau)*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/tau)*Z))
    #Step3
    Z += tau*(P - np.matmul(Y,mu) + np.matmul(X,D))
    V += tau*(W-D)
    return W,mu,Z,P,D,V
                   
def alt_frob_over(W,mu,Z,P,D,V,X,Y,sigma,tau,tau_mu,lamda,rho,gamma,eta,alpha,inv_x,inv_y):        
    W_old = copy.deepcopy(W)
    mu_old = copy.deepcopy(mu)
    Z_old = copy.deepcopy(Z)
    P_old = copy.deepcopy(P)
    D_old = copy.deepcopy(D)
    V_old = copy.deepcopy(V)

    #Step1
    mu = (1/rho)*np.matmul(Y.transpose(),Z) + np.eye(Y.shape[1])
    #Step2
    P = (tau/(tau + 2))*(np.matmul(Y,mu) - np.matmul(X,D) - (1/tau)*Z)
    W_shape = W.shape
    W = D - (1/tau)*V
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=eta)
    W = W_vec.reshape(W_shape)
    D = np.matmul(inv_x,W + (1/tau)*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/tau)*Z))
    #Step3
    Z += tau*(P - np.matmul(Y,mu) + np.matmul(X,D))
    V += tau*(W-D)
                   
    W += gamma*(W - W_old)
    mu += gamma*(mu - mu_old)
    Z += gamma*(Z - Z_old)
    P += gamma*(P - P_old)
    D += gamma*(D - D_old)
    V += gamma*(V - V_old)
                   
    return W,mu,Z,P,D,V
                   
def alt_elastic(W,mu,Z,P,D,V,X,Y,sigma,tau,tau_mu,lamda,rho,gamma,eta,alpha,inv_x,inv_y):
    
    #Step1
    mu = (1/rho)*np.matmul(Y.transpose(),Z) + np.eye(Y.shape[1])
    #Step2
    P = soft_thresholding(np.matmul(Y,mu) - np.matmul(X,D) - (1/tau)*Z,(1/tau))
    W_shape = W.shape
    W = (tau/(tau + alpha))*(D - (1/tau)*V)
    W_vec = euclidean_proj_l1ball(W.reshape(-1), s=eta)
    W = W_vec.reshape(W_shape)
    D = np.matmul(inv_x,W + (1/tau)*V - np.matmul(X.transpose(),P - np.matmul(Y,mu) + (1/tau)*Z))
    #Step3
    Z += tau*(P - np.matmul(Y,mu) + np.matmul(X,D))
    V += tau*(W-D)
    return W,mu,Z,P,D,V

def sensitivity(dicti=None):
    
    mid_value = dicti[dicti['param']]
    initial_log_name = dicti['log_name']
    
    for i,val in enumerate(np.logspace(np.log10(mid_value*0.1),np.log10(mid_value*10),10)):
        dicti[dicti['param']]=val
        dicti['log_name']=initial_log_name+"_sen_{0}_{1}".format(dicti['param'],i)
        run(dicti=dicti)
        
    fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(1 + 2*3., 2.5), sharey=False)
    fig_name = initial_log_name+'_{0}_{1}'.format('sen',dicti['param'])
    ax1 = plot_sen(ax1,experiments=[fig_name],title='Acc')
    ax2 = plot_sen(ax2,experiments=[fig_name],title='Loss')
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
    elif dicti['algorithm'] == 'admm_const':method = admm_const
    elif dicti['algorithm'] == 'admm_const_over':method = admm_const_over
    elif dicti['algorithm'] == 'admm_frob':method = admm_frob
    elif dicti['algorithm'] == 'admm_frob_over':method = admm_frob_over
    elif dicti['algorithm'] == 'admm_elastic':method = admm_elastic
    else: raise ValueError("Invalid method '{}'".format(dicti['algorithm']))

    #Data loader
    train_loader, val_loader = get_mnist_loaders(batch_size=dicti['batch_size'])
    print('Data loaded')
        
    #Initialize the variables
    W_shape = (1024,10)
    mu_shape = (10,10)
    Z_shape = (dicti['batch_size'],10)
    P_shape = Z_shape
    D_shape = W_shape
    V_shape = W_shape
    W,mu,Z,P,D,V = initialize_params(W_shape,mu_shape,Z_shape,P_shape,D_shape,V_shape)
    print('Input initialized')
                   
    total_training_time = []
    try:
        t1 = time.time()
        for epoch in range(dicti['niter']):
            
            W,mu,Z,P,D,V = train(epoch,W,mu,Z,P,D,V,train_loader,dicti['sigma'],dicti['tau']\
                           ,dicti['tau_mu'],dicti['lamda'],dicti['rho'],dicti['gamma'],dicti['eta']\
                           ,dicti['alpha'],method,logger)

            test(epoch,W,mu,Z,P,D,V,val_loader,dicti['sigma'],dicti['tau'],dicti['tau_mu']\
                 ,dicti['lamda'],dicti['rho'],dicti['gamma'],dicti['eta'],dicti['alpha'],method,logger)
        t2 = time.time()
        logger['finished'] = True
    
        print('Tau : ',dicti['tau'])
        print('Tau mu: ',dicti['tau_mu'])
        print('Sigma: ',dicti['sigma'])
        print('Rho: ',dicti['rho'])
        print('Eta: ',dicti['eta'])
        print('Lamda: ',dicti['lamda'])
        print('Total training time',t2-t1)

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
    