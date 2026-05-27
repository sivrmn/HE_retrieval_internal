#!/usr/bin/env python3
"""
WARNING: This code is intended for research and development purposes only. 
It is not intended for clinical or medical use. It has not been reviewed or 
approved by any medical or regulatory authorities. 

@author: Sivaraman Rajaganapathy
"""

# =============================================================================
# Description
# =============================================================================

# Script for plotting the encryption recovery errors with parameter changes

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
from matplotlib.patches import Patch
import seaborn as sns

# Runtime tracking
from tqdm import tqdm
import time as time


# General
import sys
import os
sys.path.append('../../')
import re

# Internal Files
from APIs.Plots.Enc_Diag_Plots import Enc_RAG_Plots
from APIs.Utils.Utils import Utils
# =============================================================================


#=============================================================================#
#========================= Instantiate Objects ===============================#
UObj = Utils()

PLTObj = Enc_RAG_Plots()

#=============================================================================#




#=============================================================================#
#========================= Plot Encryption Performance =======================#
reps = 100

results_df = pd.read_csv('../Results/EncTradeoffs/Enc_tradeoff_v2_reps_' + str(reps)+ '.csv')

#-----------------------------------------------------------------------------#
#--- Create summary data ---#
#%%

def get_p25(x):
    
    q = np.percentile(x, 25)
    
    return(q)

def get_p75(x):
    
    q = np.percentile(x, 75)
    
    return(q)



pmd_dict = {13: 'Low Security',
            14: 'Med. Security',
            15: 'High Security'}

gs_dict = {30: 'Low Precision',
           40: 'High Precision'}


#-----------------------------------------------------------------------------#

err_df = results_df.groupby(['VEC_DIM', 'PMD', 'CMBZ', 'GS'], as_index = False)['DIST_C'].agg(
                                MEAN = 'mean',                                
                                STD = 'std',
                                MED = 'median', 
                                P25 = get_p25,
                                P75 = get_p75)

err_df[['MEAN', 'STD', 'MED', 'P25', 'P75']] = err_df[['MEAN', 'STD', 'MED', 'P25', 'P75']] *100
             


conf_df = results_df.groupby(['VEC_DIM', 'PMD', 'CMBZ', 'GS'], as_index = False)['CONFIG_TIME'].agg(
                                MEAN = 'mean',                                
                                STD = 'std',
                                MED = 'median', 
                                P25 = get_p25,
                                P75 = get_p75)


comp_df = results_df.groupby(['VEC_DIM', 'PMD', 'CMBZ', 'GS'], as_index = False)['COMPUTE_TIME'].agg(
                                MEAN = 'mean',                                
                                STD = 'std',
                                MED = 'median', 
                                P25 = get_p25,
                                P75 = get_p75)
                                                                           
#-----------------------------------------------------------------------------#

    
#%%

#-----------------------------------------------------------------------------#
#-- Error plot for CKKS encryption trade offs ---------#
#-----------------------------------------------------------------------------#
def get_plot(sum_df, y_label, plt_name_tag, xscale = 'log', yscale = 'log', yticks = None, ylim = None):
    
    # Plot fonts
    plot_fonts = {'font.family':'sans-serif',
                  'font.size' : 24,
                  'figure.figsize':(17,12),
        }
    plt.rcParams.update(plot_fonts)
    plt.figure()
    # color_list = ['#77AADD', '#EE8866', '#EEDD88', '#FFAABB', '#99DDFF',
                # '#44BB99', '#BBCC33', '#AAAA00', '#DDDDDD'] # Ptol light except black
    
    # color_list = ['#EE7733', '#0077BB', '#33BBEE', '#EE3377', '#CC3311',
    #                     '#009988', '#BBBBBB', '#000000'] # Ptol vibrant
    
    color_list = ['#CCBB44', '#228833', '#66CCEE']
    marker_list = ['o', '^', 's', '*']
    
    
    #------------------ Create Isocline of interest-------------------------#
    vd_list = np.unique(sum_df['VEC_DIM'])
    pmd_list = np.unique(sum_df['PMD'])
    cmbz_list = ['[60, 30, 60]', '[60, 40, 60]']#np.unique(sum_df['CMBZ'])[[4,9]]
    gs_list = [30, 40]#np.unique(sum_df['GS'])[[1,2]]
    
    
    # cmbz = cmbz_list[2]
    # gs = gs_list[2]
    
    
    
    
    iso_cnt = 0
    
    for cmbz_cnt in range(0,len(cmbz_list),1):
        cmbz = cmbz_list[cmbz_cnt]
        gs = gs_list[cmbz_cnt]
        
        # for gs_cnt in range(0,len(gs_list),1):
            # gs = gs_list[gs_cnt]
            
        for pmd_cnt in range(0,len(pmd_list),1):
            pmd = pmd_list[pmd_cnt]
            
            cond_chk = (sum_df['PMD'] == pmd) & (sum_df['CMBZ'] == cmbz) & (sum_df['GS'] == gs)
            plt_df = sum_df[cond_chk].sort_values('VEC_DIM')
            
            
            y_err_lower = plt_df['MED'] - plt_df['P25']
            y_err_upper = plt_df['P75'] - plt_df['MED']
            yerr = np.vstack([y_err_lower, y_err_upper])
            
            plt.errorbar(plt_df['VEC_DIM'], plt_df['MED'],
                         yerr = yerr,
                         linewidth = 3,
                         linestyle = '--',
                         elinewidth = 3,
                         capsize = 5,
                         capthick = 2,
                         marker = marker_list[cmbz_cnt],
                         markersize = 25,
                         markeredgewidth = 1.5,
                         markeredgecolor = 'black',
                         color = color_list[pmd_cnt],
                         # label = "PMD = " + str(pmd) + " | GS = " + str(gs) + " | CMBZ = " + cmbz
                         label = pmd_dict[pmd] + ' | ' + gs_dict[gs]
                         )
        
            
            
            iso_cnt = iso_cnt + 1
    
    
    plt.xticks()
    plt.xscale(xscale)     
    plt.yscale(yscale)     
    
    if yticks is not None:
        plt.yticks(yticks)

    if ylim is not None:
        plt.ylim(ylim)    
    
    plt.legend(loc='best', fontsize = 18)    
              
    # plt.tight_layout()
    plt.ylabel(y_label)
    plt.xlabel('Vector Size')
    # plt.ylim([0,1.05])
    plt.grid('minor')
    # plt.title('Encryption on different LLMs')
    
    UObj.folder_check('Plots/')
    plt.savefig('Plots/Fig3_enc_tradeoff'+plt_name_tag+'.jpg')
    plt.savefig('Plots/Fig3_enc_tradeoff'+plt_name_tag+'.svg')  
    #-----------------------------------------------------------------------------#        


#%%

get_plot(err_df, 'Percent Deviation', '_DotProdErrors', xscale = 'log', yscale = 'log')
            

get_plot(conf_df, 'Time (s)', '_ConfTimes', xscale = 'log', yscale = 'log')
            

get_plot(comp_df, 'Time (s)', '_ComputeTimes', xscale = 'log', yscale = 'log', yticks=[1e-2, 1e-1, 1e0],
    ylim=[1e-2, 1e0])
#-----------------------------------------------------------------------------#

#=============================================================================#

