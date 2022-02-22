#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 18:13:58 2022

@author: lingxiaoli
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]='5,6,7'

from typing import Tuple, Generator, Callable, Optional
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

import sys
sys.path.append('../robustbench')

from robustbench.data import load_cifar10c,load_cifar10
from robustbench.utils import clean_accuracy
import pickle
def unpickle(file):
    '''Load byte data from file'''
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
    return data

def normalize_dataset(data):
    data_ = data
    mean = data_.mean(axis=(0,1,2))
    std = data_.std(axis=(0,1,2))
    return (data_ - mean)/std

n_examples = 1000*15

corruption = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                       'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                       'snow', 'frost', 'fog', 'brightness', 'contrast',
                       'elastic_transform', 'pixelate', 'jpeg_compression']
# severities = [5, 4, 3, 2, 1]
#corruption = ['snow']
n_examples = int(n_examples/len(corruption))*len(corruption)
n_test = int(n_examples/len(corruption))
X_corr, y_corr = load_cifar10c(data_dir = 'robustbench/robustbench/data', n_examples=n_examples, corruptions=corruption, severity = 2)
#all images come from cifar10 test set
label_encoding = unpickle("robustbench/robustbench/data/cifar-10-batches-py/batches.meta")
label_encoding=label_encoding['label_names']

n_corr = len(corruption)
X_c = [normalize_dataset(X_corr[i * n_test:(i + 1) * n_test]) for i in range(n_corr)]
y_c = [y_corr[i * n_test:(i + 1) * n_test] for i in range(n_corr)]
x_test, y_test = load_cifar10(data_dir = 'robustbench/robustbench/data/cifar-data', n_examples= n_test)

np.random.seed(0)
idx = np.random.choice(n_test, size=n_test // 2, replace=False)
idx_h0 = np.delete(np.arange(n_test), idx, axis=0)
X_ref,y_ref = normalize_dataset(x_test[idx]), y_test[idx]
X_h0, y_h0 = normalize_dataset(x_test[idx_h0]), y_test[idx_h0]
print(X_ref.shape, X_h0.shape)

# check that the classes are more or less balanced
classes, counts_ref = np.unique(y_ref, return_counts=True)
counts_h0 = np.unique(y_h0, return_counts=True)[1]

from timeit import default_timer as timer
from k_means_KS import kmeans

def preprocess_data(x_corr, y_corr, corruption):
    all_data = []
    all_label = []
    if isinstance(x_corr, list):
        for x, y, c in zip(x_corr, y_corr, corruption):
            x = x.numpy()
            y = y.numpy()
            x_test = x[:10]
            y_test = y[:10]
            print(y_test)
            temp = []
            for i in range(len(x_test)):
                te = x_test[i].flatten()
                temp.append(te)
            x = temp
            all_data += x
            all_label.append(y_test)
    return all_data, all_label
    
t = timer()
print(y_c[0])
all_data, all_label = preprocess_data(X_c,y_c, corruption)
all_data = np.array(all_data)
all_label = np.array(all_label)
dt = timer() -t
print(f'Time (s) {dt:.3f}')

print(type(all_data))
t = timer()
label = kmeans(all_data, 15, 10)
print(label)
print(all_label)
dt = timer() -t
print(f'Time (s) {dt:.3f}')
