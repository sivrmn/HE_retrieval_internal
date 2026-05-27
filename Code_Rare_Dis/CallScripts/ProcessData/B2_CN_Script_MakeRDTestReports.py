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

# Script for preparing a DataFrame of RD test clinical notes with labels 

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


import sys
sys.path.append('../../')
import os
# Internal Files

from APIs.Utils.Utils import Utils
# =============================================================================


UObj = Utils()

save_path = "../../../Processed_Data/"

path_case_reports = "../../../Raw_Data/Case_Reports/"

path_gemini_cn_so = path_case_reports + "clinical_notes_gemini2p5flash/so_sections/"

#%%


rd_curated_df = pd.DataFrame([])

rd_curated_df['FILE_NAME'] = []
rd_curated_df['DISEASE_ABBR'] = []
rd_curated_df['DISEASE_OP_LABEL'] = []
rd_curated_df['CLINICAL_NOTE_SO'] = []

# Dictionary linking disease abbreviation with OrphaCode
dis_dict = {'ALS' : 803,
            'FOP' : 337,
            'IPF' : 2032,
            'SC' : 171,
            'TN' : 221091}

df_cnt = 0

dis_list = os.listdir(path_gemini_cn_so)


# Cycle through each disease folder
for ii in range(0,len(dis_list),1):
    
    dis_abbr = dis_list[ii]
    
    path_dis = path_gemini_cn_so + dis_abbr + '/'
    
    dis_file_list = os.listdir(path_dis)
    
    
    # Cycle through each case report
    for jj in range(0,len(dis_file_list),1):
    
        path_dis_file = path_dis + dis_file_list[jj] 
        
        with open(path_dis_file, "r", encoding="utf-8") as f:
            cn_so_sections = f.read()                   
            

        rd_curated_df.loc[df_cnt, 'FILE_NAME'] = dis_file_list[jj] 
        rd_curated_df.loc[df_cnt,'DISEASE_ABBR'] = dis_abbr
        rd_curated_df.loc[df_cnt,'DISEASE_OP_LABEL'] = dis_dict[dis_abbr]
        rd_curated_df.loc[df_cnt,'CLINICAL_NOTE_SO'] = cn_so_sections

        df_cnt = df_cnt + 1

rd_curated_df.to_csv(save_path + "rd_synthetic_SO_sections.csv", index = False)

