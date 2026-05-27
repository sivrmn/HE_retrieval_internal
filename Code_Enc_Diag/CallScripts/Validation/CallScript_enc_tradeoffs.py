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

# Code for testing encryption trade offs with the TenSEAL CKKS algorithm

# =============================================================================
 
# =============================================================================
# Imports
# ============================================================================= 
# Scientific computing
import numpy as np
import pandas as pd

# Graphics
import matplotlib.pyplot as plt
import seaborn as sns

# Runtime tracking
from tqdm import tqdm
import time as time

# Encryption Scheme
import tenseal as ts



# General
import sys
import os
sys.path.append('../../')
import re

# Internal Files
from APIs.Validation.Validation_Enc_Diag import Enc_RAG_Validation
from APIs.Utils.Utils import Utils
# =============================================================================


#=============================================================================#
#========================= Instantiate Objects ===============================#
UObj = Utils()


VALObj = Enc_RAG_Validation()


#=============================================================================#


#--------------------------------------------------------------------------        
# Generate tenseal CKKS context
#--------------------------------------------------------------------------  
def get_ts_ckks_context(poly_modulus_degree = 8192, 
                        coeff_mod_bit_sizes = [60, 40, 40, 60], 
                        global_scale = pow(2,40)):
    
    context = ts.context(ts.SCHEME_TYPE.CKKS, 
                         poly_modulus_degree=poly_modulus_degree, 
                         coeff_mod_bit_sizes=coeff_mod_bit_sizes)
    
    context.global_scale = global_scale
    
    context.generate_galois_keys()
    
    return(context)

#--------------------------------------------------------------------------   


#--------------------------------------------------------------------------        
# Get the relative errors in performing encrypted dot products
#--------------------------------------------------------------------------  
def get_err_encdot(vec_a, vec_b, ts_context, secret_key):

    vec_c = np.dot(vec_a, vec_b)
    
    vec_enc_a = ts.ckks_vector(ts_context, vec_a)
    vec_enc_b = ts.ckks_vector(ts_context, vec_b)
    vec_enc_c = vec_enc_b.dot(vec_enc_a)
    
    
    vec_recov_a = vec_enc_a.decrypt(secret_key)
    vec_recov_b = vec_enc_b.decrypt(secret_key)
    vec_recov_c = vec_enc_c.decrypt(secret_key)
    
    
    dist_a = VALObj.get_rel_l2_error(vec_a, vec_recov_a)
    dist_b = VALObj.get_rel_l2_error(vec_b, vec_recov_b)
    dist_c = np.abs(vec_c - vec_recov_c)/np.abs(vec_c)
    
    return(dist_a, dist_b, dist_c)
#--------------------------------------------------------------------------  




#%%
#--------------------------------------------------------------------------
# Generate a random vector, encrypt, decrypt then compare with original
#--------------------------------------------------------------------------
reps = 100

vec_dim_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

pmd_list = [2**13, 2**14, 2**15] # Security level


cmbz_list = [[60, 30, 60],
             # [60, 30, 30, 60],
             # [60, 30, 30, 30, 60],
             # [60, 30, 30, 30, 30, 60],
             # [60, 30, 30, 30, 30, 30, 60],
             [60, 40, 60],
             # [60, 40, 40, 60],
             # [60, 40, 40, 40, 60],
             # [60, 40, 40, 40, 40, 60],
             # [60, 40, 40, 40, 40, 40, 60]
             ]

gs_list = [pow(2,20),
           pow(2,30),
           pow(2,40)]


results_df = pd.DataFrame([])
results_df['VEC_DIM'] = []
results_df['PMD'] = []
results_df['CMBZ'] = []
results_df['GS'] = []


results_df['DIST_A'] = []
results_df['DIST_B'] = []
results_df['DIST_C'] = []

results_df['CONFIG_TIME'] = []
results_df['COMPUTE_TIME'] = []


lcl_cnt = 0


for vec_dim_cnt in tqdm(range(0,len(vec_dim_list),1)):

    vec_dim = vec_dim_list[vec_dim_cnt]   

    for rep_cnt in range(0,reps,1):
        
        vec_a = np.random.uniform(low = -1.0, high = 1.0,  size = vec_dim)
        vec_b = np.random.uniform(low = -1.0, high = 1.0,  size = vec_dim)

        for pmd_cnt in range(0,len(pmd_list),1):
        
            pmd = pmd_list[pmd_cnt]
            
            for cmbz_cnt in range(0,len(cmbz_list),1):
                
                cmbz = cmbz_list[cmbz_cnt]
                    
                for gs_cnt in range(0,len(gs_list),1):
                    
                    gs = gs_list[gs_cnt]
                     
                        
                    try:
                        t_start_conf = time.time()
                        ts_context = get_ts_ckks_context(poly_modulus_degree = pmd,
                                                         coeff_mod_bit_sizes = cmbz,
                                                         global_scale = gs)
                        
                        secret_key = ts_context.secret_key()
                        ts_context.make_context_public()
                        t_end_conf = time.time()
                        
                        t_start_comp = time.time()
                        [dist_a, dist_b, dist_c] = get_err_encdot(vec_a, vec_b, ts_context, secret_key)
                        t_end_comp = time.time()
                    
                        results_df.loc[lcl_cnt,'VEC_DIM'] = vec_dim
                        results_df.loc[lcl_cnt,'PMD'] = str(np.log2(pmd))
                        results_df.loc[lcl_cnt,'CMBZ'] = str(cmbz)
                        results_df.loc[lcl_cnt,'GS'] = str(np.log2(gs))
                        
                        
                        results_df.loc[lcl_cnt,'DIST_A'] = dist_a
                        results_df.loc[lcl_cnt,'DIST_B'] = dist_b
                        results_df.loc[lcl_cnt,'DIST_C'] = dist_c
                        
                        results_df.loc[lcl_cnt,'CONFIG_TIME'] = t_end_conf - t_start_conf
                        results_df.loc[lcl_cnt,'COMPUTE_TIME'] = t_end_comp - t_start_comp
                    
                    except:
                        
                        results_df.loc[lcl_cnt,'VEC_DIM'] = vec_dim
                        results_df.loc[lcl_cnt,'PMD'] = str(np.log2(pmd))
                        results_df.loc[lcl_cnt,'CMBZ'] = str(cmbz)
                        results_df.loc[lcl_cnt,'GS'] = str(np.log2(gs))
                        
                        
                        results_df.loc[lcl_cnt,'DIST_A'] = np.nan
                        results_df.loc[lcl_cnt,'DIST_B'] = np.nan
                        results_df.loc[lcl_cnt,'DIST_C'] = np.nan
                        
                        results_df.loc[lcl_cnt,'CONFIG_TIME'] = np.nan
                        results_df.loc[lcl_cnt,'COMPUTE_TIME'] = np.nan
                        
                        print('skipping condition')
                
                
                    lcl_cnt = lcl_cnt + 1



#--------------------------------------------------------------------------



#%%

UObj.folder_check('../Results/EncTradeoffs/')


results_df.to_csv('../Results/EncTradeoffs/Enc_tradeoff_v2_reps_' + str(reps)+ '.csv')



