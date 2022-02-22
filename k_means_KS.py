#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 10:19:45 2022

@author: lingxiaoli
"""

import numpy as np
from scipy.spatial.distance import cdist 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
 
#Defining our function 
def KS_kmeans(x, k, no_of_iterations):
    idx = np.random.choice(len(x), k, replace=False)
    #Randomly choosing Centroids 
    centroids = x[idx, :] #Step 1
    #finding the distance between centroids and all the data points
    
    distances = []
    for n in range(len(x)):
        distance = []
        for m in range(len(centroids)):
            dist, p_val = ks_2samp(x[n], centroids[m], alternative='two-sided', mode='asymp')
            distance.append(dist)
        distances.append(distance)
        
    distances = np.array(distances)
    
    # distances = cdist(x, centroids ,'euclidean') #Step 2
    #Centroid with the minimum Distance
    points = np.array([np.argmin(i) for i in distances]) #Step 3
     
    #Repeating the above steps for a defined number of iterations
    #Step 4
    for _ in range(no_of_iterations): 
        centroids = []
        for idx in range(k):
            #Updating Centroids by taking mean of Cluster it belongs to
            temp_cent = x[points==idx].mean(axis=0) 
            centroids.append(temp_cent)
 
        centroids = np.vstack(centroids) #Updated Centroids 
        distances = []
        for n in range(len(x)):
            distance = []
            for m in range(len(centroids)):
                dist, p_val = ks_2samp(x[n], centroids[m], alternative='two-sided', mode='asymp')
                distance.append(dist)
            distances.append(distance)
        distances = np.array(distances)
        
        # distances = cdist(x, centroids ,'euclidean') #Step 2
        points = np.array([np.argmin(i) for i in distances])
         
    return points 
