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

# Code for Utility functions

# =============================================================================
 
 
# =============================================================================
# Imports
# ============================================================================= 
import numpy as np
import os

import sys
sys.path.append('../../')

# =============================================================================
#%%
# =============================================================================
# Class containing utility functions
# =============================================================================
class Utils():
    """
    Class containging useful utility methods. 
    """
    #--------------------------------------------------------------------------        
    # Init
    #--------------------------------------------------------------------------  
    def __init__(self): 
        
        a = 1
    
    #--------------------------------------------------------------------------        

    #--------------------------------------------------------------------------        
    # Check if directory exists
    #--------------------------------------------------------------------------  
    def folder_check(self, folder_path): 
        """
        Check if folder exists, else create one. 

        <span style="color: darkred;">Parameters</span>
        ----------
        **`folder_path`** : `str`
            Folder path to create or check existence of.

        <span style="color: darkblue;">Returns</span>
        -------
        None.
        """        
        if(os.path.exists(folder_path) == False):
            # Create target Directory
            os.makedirs(folder_path)
            
        return()
    #--------------------------------------------------------------------------        

# =============================================================================