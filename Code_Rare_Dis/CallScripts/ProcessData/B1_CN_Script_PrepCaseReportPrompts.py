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

# Script for preparing the prompts for creating synthetic clinical notes 
# from case reports on rare diseases using LLMs. 

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


path_case_reports = "../../../Raw_Data/Case_Reports/"

path_excerpts = path_case_reports + "case_reports_excerpts/"

path_prompts = path_case_reports + "case_reports_prompts/"

path_gemini_cn_so = path_case_reports + "clinical_notes_gemini2p5flash/so_sections_w_artificial_names/"

#%%
prompt_header = "Please generate a mock clinical note from the following case report with clearly marked SOAP sections and randomly assign a realistic patient name: \n\n"

#%%

dis_list = os.listdir(path_excerpts)


# Cycle through each disease folder
for ii in range(0,len(dis_list),1):
    
    dis_abbr = dis_list[ii]
    
    path_dis = path_excerpts + dis_abbr + '/'
    
    dis_file_list = os.listdir(path_dis)
    
    # Cycle through each case report
    for jj in range(0,len(dis_file_list),1):
    
        path_dis_file = path_dis + dis_file_list[jj] 
        
        file_name = dis_file_list[jj].replace('_report_excerpt','')
        file_name = file_name.replace('.txt','')
        
        with open(path_dis_file, "r", encoding="utf-8") as f:
            case_report = f.read()
        
        
        prompt = prompt_header + '\n' + case_report
        
        UObj.folder_check(path_prompts +dis_abbr+ '/')
        with open(path_prompts + dis_abbr + '/' + file_name + '_prompt.txt', "w", encoding="utf-8") as f:
            f.write(prompt)
        
        
        
            
        # Create SO sections placeholder text files if none exist
        UObj.folder_check(path_gemini_cn_so + dis_abbr + '/')
        path_so_sections = path_gemini_cn_so + dis_abbr + '/'  + file_name + '_gemini2p5flash_SO.txt'
        if not os.path.exists(path_so_sections):
            with open(path_so_sections, "w", encoding="utf-8"):
                pass
            print("File created.")
        else:
            print("File already exists, unchanged.")
            
        
        


































