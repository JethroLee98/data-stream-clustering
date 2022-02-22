#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  16 17:05:49 2022

@author: lingxiaoli
"""

import numpy as np
from scipy.special import rel_entr, softmax
from scipy.stats import entropy

class KLDivergence:

    def __init__(self, x_ref, threshold: float = .05):
        self.x_ref = x_ref
        self.threshold = threshold

    def updata_ref(self, x_ref):
        self.x_ref = x_ref
    
    def update_threshold(self, threshold):
        self.threshold = threshold

    def get_divergence(self, x_ref, x):
        kl_pq = rel_entr(x_ref, x)
        return sum(kl_pq)

    def process_data(self, x):
        return entropy(softmax(x.detach().numpy(), axis=-1), axis=-1)

    def get_result(self, x):
        x_ref = self.process_data(self.x_ref)
        x = self.process_data(x)
        kl_dis = self.get_divergence(x_ref, x)
        threshold = self.threshold
        drift_pred = int((kl_dis < threshold).any())  # type: ignore[assignment]
        cd = {}
        cd['is_drift'] = drift_pred
        cd['threshold'] = threshold
        cd['distance'] = kl_dis
        return cd
