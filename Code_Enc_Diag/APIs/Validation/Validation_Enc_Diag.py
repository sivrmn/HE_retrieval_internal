#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WARNING: This code is intended for research and development purposes only. 
It is not intended for clinical or medical use. It has not been reviewed or 
approved by any medical or regulatory authorities. 

@author: Sivaraman Rajaganapathy
"""

# =============================================================================
# Description
# =============================================================================

# Code for validating the encryption RAG pipeline performance with application 
# to rare disease diagnosis

# =============================================================================
 
 
# =============================================================================
# Imports
# ============================================================================= 
# Scientific computing
import numpy as np
import pandas as pd
import scipy as sc

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns

# Runtime tracking
from tqdm import tqdm

# Encryption Scheme
# import tenseal as ts


# Data
# import datasets as ds

#AI/ML
import sentence_transformers as st
import torch

# General
import sys
import os
sys.path.append('../../')
import re

# Internal Files

# =============================================================================
#%%
# =============================================================================
# Class containing functions for validating Enc RAG
# =============================================================================
class Enc_RAG_Validation():
    """
    Class of methods for validating retrieval augmented generation based phenotyping
    with and without encryption of embeddings.  
    """
    #--------------------------------------------------------------------------        
    # Init
    #--------------------------------------------------------------------------  
    def __init__(self): 
        
        a = 1
        # Temporary - reroute from RD Class
        self.disease_dict = {'ALS': 'Amyotrophic lateral sclerosis',
                        'TN': 'Trigeminal neuralgia',
                        'SC': 'sclerosing cholangitis',
                        'FOP': 'Fibrodysplasia ossificans progressiva',
                        'IPF': 'Idiopathic pulmonary fibrosis'}
        
    
    #--------------------------------------------------------------------------        

    
    #--------------------------------------------------------------------------        
    # Normalized Discounted Cumulative Gain (nDCG) for one correct answer
    #--------------------------------------------------------------------------  
    def get_ndcg_single_output(self, rank, cutoff_rank):
        """
        Find the normalized Discounted Cumulative Gain (nDCG) for one correct answer.  

        <span style="color: darkred;">Parameters</span>  
        ----------
        **`rank`** : `int`  
            Rank of correct answer in the list.  
        **`cutoff_rank`** : `int`  
            Cutoff rank above which a nDCG of 0 is assigned.  

        <span style="color: darkblue;">Returns</span>
        -------
        **`ndcg`** : `float`  
            nDCG score. 
        """                
        if(rank > cutoff_rank):
            ndcg = 0
        else:
            ndcg = 1/(np.log2(rank + 1))
        
        
        return(ndcg)

    #--------------------------------------------------------------------------     



    #--------------------------------------------------------------------------        
    # Pick out the rank of the disease from retrieval list of rare diseases
    #--------------------------------------------------------------------------  
    def get_dis_rank(self, srch_df, disease_abbr):
        """
        Pick out the rank of the disease from retrieval list of rare diseases.    

        <span style="color: darkred;">Parameters</span>  
        ----------
        **`srch_df`** : `pd.DataFrame`  
            Pandas DataFrame with ranked disease list.  
        **`disease_abbr`** : `str`  
            Abbreviation of selected disease.  

        <span style="color: darkblue;">Returns</span>  
        -------
        **`disease_rank`** : `int`  
            Rank of the selected in the list.  
        """
        disease_name = self.disease_dict[disease_abbr]
        
        idx = srch_df.index[srch_df["Disease"].str.contains(disease_name)].tolist()
        
        disease_rank = np.min(idx) + 1
        
        return(disease_rank)
    #--------------------------------------------------------------------------    
    
    
    #--------------------------------------------------------------------------
    # Get normalized root mean squared error, normalized by range
    #--------------------------------------------------------------------------
    def get_nrmse(self, vec_x, vec_y):
        """
        Get normalized root mean squared error, normalized by range.  

        <span style="color: darkred;">Parameters</span>  
        ----------
        **`vec_x`** : `np.ndarray`  
            Array 1 in a pair of arrays compared.  
        **`vec_y`** : `np.ndarray`  
            Array 2 in a pair of arrays compared.  

        <span style="color: darkblue;">Returns</span>  
        -------
        **`nrmse`** : `float`  
        Normalized Root Mean Squared Error between teh compared arrays.  

        """
        vec_diff = vec_x - vec_y
        
        vec_dim = np.size(vec_diff)
        
        rmse = np.sqrt(((np.linalg.norm(vec_diff, ord = 2))**2)/vec_dim)
        
        nrmse = rmse/(np.max(vec_x) - np.min(vec_x))
        
        return(nrmse)
    #--------------------------------------------------------------------------



    #--------------------------------------------------------------------------
    # Get relative l2 error
    #--------------------------------------------------------------------------
    def get_rel_l2_error(self, vec_x, vec_y):
        """
        Get relative error between two arrays (l2 norm).

        <span style="color: darkred;">Parameters</span>
        ----------
        **`vec_x`** : `np.ndarray`  
            Array 1 in a pair of arrays compared.  
        **`vec_y`** : `np.ndarray`  
            Array 2 in a pair of arrays compared.  

        <span style="color: darkblue;">Returns</span>
        -------
        **`rel_l2`** : 2 norm of the relative error vector  

        """
        
        rel_l2 = np.linalg.norm(vec_x - vec_y, ord = 2)/np.linalg.norm(vec_x, ord = 2)
        
        
        return(rel_l2)
    #--------------------------------------------------------------------------
    
    
    #--------------------------------------------------------------------------        
    # Confidence interval from KS metric
    #--------------------------------------------------------------------------  
    def ks_distance_ci(self, x, y, n_boot=100, alpha=0.05, random_state=None):
        """
        Compute Kolmogorov-Smirnov distance between two samples, 
        and a bootstrap CI for that distance.  

        <span style="color: darkred;">Parameters</span>  
        ----------
        **`x`** : `array-like`  
            1D samples from the two distributions.    
        **`y`** : `array-like`  
            1D samples from the two distributions.  
        **`n_boot`** : `int`, optional   
            Number of bootstrap resamples. The default is 100.  
        **`alpha`** : `float`, optional  
            Significance level (confidence = 100*(1-alpha)). The default is 0.05.  
        **`random_state`** : `int` or `None`, optional  
            Randomization seed for reproducibility. The default is None.  

        <span style="color: darkblue;">Returns</span>
        -------
        **`KS_dist`** : `float`  
            Kolmogorov-Smirnov distance metric.  
        **`lower`** : `float` 
            Lower bound of bootstrap confidence interval at level alpha.  
        **`upper`** : `float` 
            Upper bound of bootstrap confidence interval at level alpha.  
        **`boot_stats`** : `sc.stats.object` 
            KS test results                

        """
        x = np.asarray(x)
        y = np.asarray(y)
    
        rng = np.random.default_rng(random_state)
    
        # Observed KS distance
        KS_dist, _ = sc.stats.ks_2samp(x, y, alternative="two-sided", mode="auto")
    
        # Bootstrap distribution
        boot_stats = np.empty(n_boot)
        n_x = len(x)
        n_y = len(y)
    
        for b in range(n_boot):
            x_b = rng.choice(x, size=n_x, replace=True)
            y_b = rng.choice(y, size=n_y, replace=True)
            boot_stats[b], _ = sc.stats.ks_2samp(x_b, y_b, alternative="two-sided", mode="auto")
    
        # Percentile CI
        lower = np.quantile(boot_stats, alpha / 2)
        upper = np.quantile(boot_stats, 1 - alpha / 2)
    
        return(KS_dist, lower, upper, boot_stats)
    #--------------------------------------------------------------------------    
    
    
    
    #--------------------------------------------------------------------------        
    # Confidence interval from Wasserstein distance
    #--------------------------------------------------------------------------  
    def wasserstein_ci(self, x, y, n_boot = 10000, alpha=0.05, delta=None, random_state=None):
        """
        1D Wasserstein (Earth Mover's) distance between two samples,
        with a bootstrap confidence interval and optional "probability
        distance < delta".          

        <span style="color: darkred;">Parameters</span>  
        ----------
        **`x`** : `array-like`  
            1D samples from the two distributions.      
        **`y`** : `array-like`  
            Array 2 in a pair of arrays compared.  
        **`n_boot`** : `int`, optional  
            Number of bootstrap resamples. The default is 10000.  
        **`alpha`** : `float`, optional  
            Significance level (confidence = 100*(1-alpha)). The default is 0.05.   
        **`delta`** : `float` or `None`, optional  
            Tolerance threshold for "practically identical" distance.
            If provided, we estimate P(W < delta) from the bootstrap. The default is None.    
        **`random_state`** : `int` or `None`, optional  
            Randomization seed for reproducibility. The default is None.  

        <span style="color: darkblue;">Returns</span>
        -------
        **`W_obs`** : `float`  
            Observed Wasserstein distance.  
        **`ci_low`** : `float` 
            Lower bound of bootstrap confidence interval at level alpha.  
        **`ci_high`** : `float` 
            Upper bound of bootstrap confidence interval at level alpha.  
        **`p_boot_less_delta`** : `rng.object` 
            estimated P(W < delta) if delta given else None.  

        """
        x = np.asarray(x)
        y = np.asarray(y)
    
        rng = np.random.default_rng(random_state)
    
        # Observed Wasserstein distance
        W_obs = sc.stats.wasserstein_distance(x, y)
    
        n_x = len(x)
        n_y = len(y)
    
        # Bootstrap distribution of Wasserstein distances
        boot_stats = np.empty(n_boot)
        for b in range(n_boot):
            x_b = rng.choice(x, size=n_x, replace=True)
            y_b = rng.choice(y, size=n_y, replace=True)
            boot_stats[b] = sc.stats.wasserstein_distance(x_b, y_b)
    
        # Percentile CI
        ci_low = np.quantile(boot_stats, alpha / 2)
        ci_high = np.quantile(boot_stats, 1 - alpha / 2)
    
        # Probability distance < delta (if requested)
        p_boot_less_delta = None
        if delta is not None:
            p_boot_less_delta = float(np.mean(boot_stats < delta))
    
        return( W_obs, ci_low, ci_high, p_boot_less_delta)
    #-------------------------------------------------------------------------- 

# =============================================================================


