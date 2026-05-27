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

# Script for plotting encrypted retrieval across LLMs

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


# Data
import datasets as ds
from itertools import chain


# General
import sys
import os
sys.path.append('../../')
import re
from io import StringIO

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
#========================= Plot Retrieval Performance ========================#

INCLUDE_ENCRYPTION = True

#-----------------------------------------------------------------------------#


model_name_list = ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'pritamdeka/S-PubMedBert-MS-MARCO', 'sentence-t5-xxl', 'Qwen/Qwen3-Embedding-8B']
model_name_abbr_list = ['all_minim_l6_v2', 'all_mpnet_base_v2', 'pritamdeka_pubmedbert', 'sentence_t5_xxl', 'Qwen3_8B']

rename_list = lambda ori, new_dict: [new_dict[x] for x in ori]

model_name_dict = {'all_minim_l6_v2' : 'MiniLM', 
                        'all_mpnet_base_v2': 'MPNet', 
                        'pritamdeka_pubmedbert': 'S-PubMedBERT', 
                        'sentence_t5_xxl': 'Sentence-T5', 
                        'Qwen3_8B': 'Qwen3-8B'}

x_model_names = rename_list(model_name_abbr_list, model_name_dict)

disease_list = ['ALS','FOP', 'IPF' , 'SC','TN']

#-----------------------------------------------------------------------------#

results_master_df = pd.DataFrame([])
results_master_df['MODEL_NAME'] = []
results_master_df['MODEL_ABBR'] = []
results_master_df['RESULTS_SUMMARY'] = []
results_master_df['RESULTS_SUMMARY'] = results_master_df['RESULTS_SUMMARY'].astype(object)

results_master_df['NDCG_WOE'] = []
results_master_df['NDCG_WOE'] = results_master_df['NDCG_WOE'].astype(object)

results_master_df['RANK_WOE'] = []
results_master_df['RANK_WOE'] = results_master_df['RANK_WOE'].astype(object)

results_master_df['IQR_WOE'] = []
results_master_df['MED_WOE'] = []

results_master_df['AVG_WOE'] = []
results_master_df['STD_WOE'] = []


results_master_df['NDCG_WE'] = []
results_master_df['NDCG_WE'] = results_master_df['NDCG_WE'].astype(object)

results_master_df['RANK_WE'] = []
results_master_df['RANK_WE'] = results_master_df['RANK_WE'].astype(object)

results_master_df['IQR_WE'] = []
results_master_df['MED_WE'] = []

results_master_df['AVG_WE'] = []
results_master_df['STD_WE'] = []


results_master_df['KS_STAT'] = []
results_master_df['KS_PVALUE'] = []

results_master_df['WS_DIST'] = []

results_master_df['KT_STAT'] = []
results_master_df['KT_PVALUE'] = []

for jj in range(0,len(model_name_abbr_list),1):
    
    
    model_name = model_name_list[jj]
    model_name_abbr = model_name_abbr_list[jj]
    
    if(INCLUDE_ENCRYPTION == True):
        results_total_df = pd.read_csv('../Results/Gemini2p5flash_testSO_' + model_name_abbr + '.csv')
    else:
        results_total_df = pd.read_csv('../Results/Gemini2p5flash_testSO_NEC_' + model_name_abbr + '.csv')
        
    #--------------------------------------------------------------------------
    # Update the results_total_df with Kendall's tau stat and p-value for each note
    for ii in range(0,np.size(results_total_df,0),1):
    
        scores_no_enc = results_total_df.loc[ii, 'Top K without encryption']
        scores_enc = results_total_df.loc[ii, 'Top K with encryption']
        
        s_no_enc = pd.read_fwf(StringIO(scores_no_enc))
        s_enc = pd.read_fwf(StringIO(scores_enc))
        
        dis_scores = (
            s_no_enc[["Disease", "Score"]]
            .merge(
                s_enc[["Disease", "Score"]],
                on="Disease",
                suffixes=("_no_enc", "_enc"),
                how="inner"
            )
        )
        
        kt = sc.stats.kendalltau(dis_scores["Score_no_enc"], dis_scores["Score_enc"], variant = 'b')
        
        results_total_df.loc[ii,'KT_STAT'] = kt.statistic
        results_total_df.loc[ii,'KT_PVALUE'] = kt.pvalue        
    #--------------------------------------------------------------------------
    
    #------ Get a summary of the data for a bar plot ---------#
    sum_df = pd.DataFrame([])
    sum_df['DISEASE'] = []
    sum_df['NDCG_WOE'] = []
    sum_df['NDCG_WOE'] = sum_df['NDCG_WOE'].astype(object)
    
    sum_df['RANK_WOE'] = []
    sum_df['RANK_WOE'] = sum_df['RANK_WOE'].astype(object)
    
    sum_df['IQR_WOE'] = []
    sum_df['MED_WOE'] = []
    sum_df['AVG_WOE'] = []
    sum_df['STD_WOE'] = []


    sum_df['NDCG_WE'] = []
    sum_df['NDCG_WE'] = sum_df['NDCG_WE'].astype(object)
    
    sum_df['RANK_WE'] = []
    sum_df['RANK_WE'] = sum_df['RANK_WE'].astype(object)
    
    sum_df['IQR_WE'] = []
    sum_df['MED_WE'] = []
    sum_df['AVG_WE'] = []
    sum_df['STD_WE'] = []


    sum_df['KS_STAT'] = []
    sum_df['KS_PVALUE'] = []
    
    sum_df['WS_DIST'] = []
    
    sum_df['KT_STAT'] = []
    sum_df['KT_PVALUE'] = []
    

    for ii in range(0,len(disease_list),1):
        
        dis = disease_list[ii]
        
        sum_df.loc[ii,'DISEASE'] = dis
        dis_chk = results_total_df['Disease name'] == dis
        ndcg_list = results_total_df.loc[dis_chk, 'NDCG without encryption'].tolist() 
        sum_df.at[ii,'NDCG_WOE'] = ndcg_list
        
        rank_list = results_total_df.loc[dis_chk, 'Rank without encryption'].tolist()
        sum_df.at[ii,'RANK_WOE'] = rank_list
        
        sum_df.loc[ii,'IQR_WOE'] = sc.stats.iqr(ndcg_list)
        sum_df.loc[ii,'MED_WOE'] = np.median(ndcg_list)
        sum_df.loc[ii,'AVG_WOE'] = np.mean(ndcg_list)
        sum_df.loc[ii,'STD_WOE'] = np.std(ndcg_list) 
        
        enc_ndcg_list = results_total_df.loc[dis_chk, 'NDCG with encryption'].tolist() 
        sum_df.at[ii,'NDCG_WE'] = enc_ndcg_list
        
        enc_rank_list = results_total_df.loc[dis_chk, 'Rank with encryption'].tolist()
        sum_df.at[ii,'RANK_WE'] = enc_rank_list
        
        sum_df.loc[ii,'IQR_WE'] = sc.stats.iqr(enc_ndcg_list)
        sum_df.loc[ii,'MED_WE'] = np.median(enc_ndcg_list)
        sum_df.loc[ii,'AVG_WE'] = np.mean(enc_ndcg_list)
        sum_df.loc[ii,'STD_WE'] = np.std(enc_ndcg_list)     
        
        ks = sc.stats.ks_2samp(ndcg_list, enc_ndcg_list, alternative = 'two_sided')
        sum_df.loc[ii,'KS_STAT'] = ks.statistic
        sum_df.loc[ii,'KS_PVALUE'] = ks.pvalue
        
        ws = sc.stats.wasserstein_distance(ndcg_list, enc_ndcg_list)
        sum_df.loc[ii, 'WS_DIST'] = np.round(ws, 2)
    
        # kt = sc.stats.kendalltau(rank_list, enc_rank_list, variant = 'b')
        # sum_df.loc[ii,'KT_STAT'] = kt.statistic
        # sum_df.loc[ii,'KT_PVALUE'] = kt.pvalue
        
        kt_stat_list = results_total_df.loc[dis_chk, 'KT_STAT'].tolist() 
        kt_pval_list = results_total_df.loc[dis_chk, 'KT_PVALUE'].tolist() 
        sum_df.loc[ii,'KT_STAT'] = np.mean(kt_stat_list)
        sum_df.loc[ii,'KT_PVALUE'] = np.mean(kt_pval_list)   
    
    results_master_df.loc[jj,'MODEL_NAME'] = model_name
    results_master_df.loc[jj,'MODEL_ABBR'] = model_name_abbr
    results_master_df.at[jj, 'RESULTS_SUMMARY'] = sum_df
    
    ndcg_list_total = list(chain.from_iterable(sum_df['NDCG_WOE']))
    results_master_df.at[jj, 'NDCG_WOE'] = ndcg_list_total
    
    rank_list_total = list(chain.from_iterable(sum_df['RANK_WOE']))
    results_master_df.at[jj, 'RANK_WOE'] = rank_list_total
    
    results_master_df.loc[jj,'IQR_WOE'] = sc.stats.iqr(ndcg_list_total)
    results_master_df.loc[jj,'MED_WOE'] = np.median(ndcg_list_total)
    
    results_master_df.loc[jj,'AVG_WOE'] = np.mean(ndcg_list_total)
    results_master_df.loc[jj,'STD_WOE'] = np.std(ndcg_list_total)
    
    enc_ndcg_list_total = list(chain.from_iterable(sum_df['NDCG_WE']))
    results_master_df.at[jj, 'NDCG_WE'] = enc_ndcg_list_total
    
    enc_rank_list_total = list(chain.from_iterable(sum_df['RANK_WE']))
    results_master_df.at[jj, 'RANK_WE'] = enc_rank_list_total
    
    results_master_df.loc[jj,'IQR_WE'] = sc.stats.iqr(enc_ndcg_list_total)
    results_master_df.loc[jj,'MED_WE'] = np.median(enc_ndcg_list_total)
    
    results_master_df.loc[jj,'AVG_WE'] = np.mean(enc_ndcg_list_total)
    results_master_df.loc[jj,'STD_WE'] = np.std(enc_ndcg_list_total)
    
    
    ks_total = sc.stats.ks_2samp(ndcg_list_total, enc_ndcg_list_total, alternative = 'two_sided') 
    results_master_df.loc[jj,'KS_STAT'] = ks_total.statistic
    results_master_df.loc[jj,'KS_PVALUE'] = ks_total.pvalue    
    
    ws_total = sc.stats.wasserstein_distance(ndcg_list_total, enc_ndcg_list_total)
    results_master_df.loc[jj,'WS_DIST'] = ws_total#np.round(ws_total, 3)
    
    # kt_total = sc.stats.kendalltau(rank_list_total, enc_rank_list_total, variant = 'b') 
    # results_master_df.loc[jj,'KT_STAT'] = kt_total.statistic
    # results_master_df.loc[jj,'KT_PVALUE'] = kt_total.pvalue  
    
    
    kt_total_stat_list = results_total_df['KT_STAT'].to_list()
    kt_total_pval_list = results_total_df['KT_PVALUE'].to_list()
    
    results_master_df.loc[jj,'KT_STAT'] = np.nanmean(kt_total_stat_list)
    results_master_df.loc[jj,'KT_PVALUE'] = np.nanmean(kt_total_pval_list)      
    
#-----------------------------------------------------------------------------#

    
#%%

#-----------------------------------------------------------------------------#
#-- Bar plot for normalized Discounted Cumulative Gain for retrieval ---------#
# Plot fonts
plot_fonts = {'font.family':'sans-serif',
              'font.size' : 24,
              'figure.figsize':(18,12),
    }
plt.rcParams.update(plot_fonts)
plt.figure()
color_list = ['#77AADD', '#EE8866', '#EEDD88', '#FFAABB', '#99DDFF',
            '#44BB99', '#BBCC33', '#AAAA00', '#DDDDDD'] # Ptol light except black


_sup = str.maketrans("0123456789-+", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")

def format_p_scientific(p, sci_cut=1e-3, floor=1e-6, precision=2):
    """
    - If p < floor: return '< 10⁻⁶' (clean interpretation)
    - Else if p < sci_cut: return mantissa×10^exponent (unicode, robust)
    - Else: return decimal (e.g., 0.012)
    """
    if p < floor:
        return " < 10⁻⁶"
    if p < sci_cut:
        mantissa, exponent = f"{p:.{precision}e}".split("e")
        exp_sup = str(int(exponent)).translate(_sup)
        return f"= {mantissa}×10{exp_sup}"
    return f"= {p:.3f}"

def add_compline(plt, x1, x2, y, h, text):
    
    plt.plot([x1, x1, x2, x2], [y+h/2, y+h, y+h, y+h/2], lw = 1.2, c = 'k')
    
    plt.text((x1 + x2)/2, y + h, text, ha='center', va='bottom')
    
    return()

positions = np.arange(1, (len(model_name_abbr_list) + 1)*1, step = 1)*1.5
pos_offset = 0.15
bp_width = 0.25

# nDCG without Encryption
bplot = plt.boxplot(results_master_df['NDCG_WOE'], 
            vert= True, 
            patch_artist = True, 
            positions = positions-pos_offset,
            widths = bp_width,
            notch = False, 
            labels = x_model_names)

for patch in bplot['boxes']:
    patch.set_facecolor(color_list[0]) 
    
for ii in range(0,len(model_name_abbr_list),1):
    med_val = results_master_df.loc[ii,'MED_WOE']
    iqr_val = results_master_df.loc[ii,'IQR_WOE']
    avg_val = results_master_df.loc[ii,'AVG_WOE']
    std_val = results_master_df.loc[ii,'STD_WOE']
    
    texty_val = np.percentile(sum_df.loc[ii,'NDCG_WOE'],25)-(0.02*((-1)**ii))
    wstext_y =  np.percentile(sum_df.loc[ii,'NDCG_WOE'],95)
    
    # plt.text(positions[ii]-5*pos_offset, med_val+0.25, f'Med.: {med_val:.2f}', color = 'navy')
    # plt.text(positions[ii]-5*pos_offset, med_val+0.20, f'  IQR: {iqr_val:.2f}', color = 'navy')

    plt.text(positions[ii]-6.5*pos_offset, avg_val+0.20-(0.05*(-1)**ii), f'Mean: {avg_val:.2f}', color = 'navy')
    plt.text(positions[ii]-6.5*pos_offset, avg_val+0.15-(0.05*(-1)**ii), f'Std.: {std_val:.2f}', color = 'navy')


# nDCG with Encryption
bplot = plt.boxplot(results_master_df['NDCG_WE'], 
            vert= True, 
            patch_artist = True, 
            positions = positions+pos_offset,
            widths = bp_width,
            notch = False, 
            labels = x_model_names)

for patch in bplot['boxes']:
    patch.set_facecolor(color_list[1])
    patch.set_hatch('/')
    
    
    
upper_whisker = lambda x: np.max(x[x <= np.percentile(x,75) + 1.5*(np.percentile(x,75)-np.percentile(x,25))])
for ii in range(0,len(model_name_abbr_list),1):
    med_val = results_master_df.loc[ii,'MED_WE']
    iqr_val = results_master_df.loc[ii,'IQR_WE']
    avg_val = results_master_df.loc[ii,'AVG_WE']
    std_val = results_master_df.loc[ii,'STD_WE']
    
    texty_val = np.percentile(sum_df.loc[ii,'NDCG_WE'],25)-(0.02*((-1)**ii))
    wstext_y =  upper_whisker(np.array(results_master_df.loc[ii, 'NDCG_WE'])) #np.percentile(sum_df.loc[ii,'NDCG_WE'],95
    
    # plt.text(positions[ii]+pos_offset, med_val+0.15, f'Med.: {med_val:.2f}', color = 'maroon')
    # plt.text(positions[ii]+pos_offset, med_val+0.10, f'  IQR: {iqr_val:.2f}', color = 'maroon')

    plt.text(positions[ii]+1*pos_offset, avg_val+0.20-(0.05*(-1)**ii), f'Mean: {avg_val:.2f}', color = 'maroon')
    plt.text(positions[ii]+1*pos_offset, avg_val+0.15-(0.05*(-1)**ii), f' Std.: {std_val:.2f}', color = 'maroon')
    
    # add_compline(plt, positions[ii]-pos_offset, positions[ii]+pos_offset, wstext_y, 0.05,
                 # text = 'WD: ' + "{:.3f}".format(results_master_df.loc[ii, 'WS_DIST']))

    add_compline(plt, positions[ii]-pos_offset, positions[ii]+pos_offset, wstext_y, 0.05,
                 text = (r"$\tau$: {:.2f}" "\n" r"$p$ {}").format(results_master_df.loc[ii, 'KT_STAT'],
                                                                   format_p_scientific(results_master_df.loc[ii, 'KT_PVALUE'])))



plt.xticks(positions)
              
# Create custom legend handles
legend_elements = [Patch(facecolor=color_list[0], edgecolor='k', label='Without Encryption'),
                   Patch(facecolor=color_list[1], edgecolor='k', hatch = '/', label='With Encryption')]

plt.legend(handles=legend_elements, loc='upper left')    
          
plt.tight_layout()
plt.ylabel('nDCG')
plt.ylim([0,1.18])
plt.grid()
# plt.title('Encryption on different LLMs')

UObj.folder_check('Plots/')
plt.savefig('Plots/Fig2_retrieval_LLMs.jpg')
plt.savefig('Plots/Fig2_retrieval_LLMs.svg')  
#-----------------------------------------------------------------------------#        

            
            
#-----------------------------------------------------------------------------#

#=============================================================================#

