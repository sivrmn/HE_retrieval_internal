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

# Helper functions for encrypted RAG pipeline

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

# Encryption Scheme
import tenseal as ts


# Data
import datasets as ds

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
# =============================================================================
class Enc_RAG_Embeddings():
    """
    Class containing helper methods for creating embeddings for the encryption RAG 
    pipeline.  
    """    
    #--------------------------------------------------------------------------        
    # Init
    #--------------------------------------------------------------------------  
    def __init__(self): 
        
        a = 1
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    
    #--------------------------------------------------------------------------        



    #--------------------------------------------------------------------------        
    # Function to identify the appropriate acceleration hardware
    #--------------------------------------------------------------------------  
    def get_device(self):
        """
        Find the relevant GPU available.   
        
        <span style="color: darkred;">Parameters</span>
        ----------
        None  
        
        <span style="color: darkblue;">Returns</span>   
        -------  
        **`DEVICE`** : `str`  
            Deveice type.  

        """
        if torch.backends.mps.is_available():
            DEVICE = "mps"
        elif torch.cuda.is_available():
            DEVICE = "cuda"
        else:
            DEVICE = "cpu"
        
        return(DEVICE)
    #-------------------------------------------------------------------------- 
    
    

    #--------------------------------------------------------------------------        
    # Function to clean up clinical notes text before generating embeddings
    #--------------------------------------------------------------------------  
    def get_clean_text(self, text):
        """        
        Clean up raw text for LLM embedddings.  
        <span style="color: darkred;">Parameters</span>
        ----------
        **`text`**: `str`  
            Raw text to be cleaned.  

        <span style="color: darkblue;">Returns</span>   
        -------  
        **`clean_text`** : `str`
            Cleaned text.  

        """
        
        # Replace multiple spaces or line breaks with a single space
        clean_text = re.sub(r'\s+', ' ', text).strip()
        clean_text = clean_text.lower()
        return(clean_text)
    #--------------------------------------------------------------------------        
    
    
    
    
    
    #--------------------------------------------------------------------------        
    # Function to select a SentenceTransformer model
    #--------------------------------------------------------------------------
    def select_st_model(self, model_name):
        """
        <span style="color: darkred;">Parameters</span>
        ----------
        model_name : 

        <span style="color: darkblue;">Returns</span> 
        -------
        None.

        """
        
        model = st.SentenceTransformer(model_name)
        
        return(model)
    #--------------------------------------------------------------------------  



    #--------------------------------------------------------------------------        
    # Function to generate embeddings using LLMs from Sentence Transformers wrapper
    #--------------------------------------------------------------------------  
    def get_st_embeddings(self, model, text, convert_to_numpy = True, 
                          normalize_embeddings = True, DEVICE = 'cpu',
                          BATCH_SIZE = 32, PRECISION = "float32"):
        """
        Helper function to generate embeddings using LLMs from the Sentence
        Transformers library.  
         <span style="color: darkred;">Parameters</span>
        ----------
        **`model`** : `st.SentenceTransformer`  
            Model to be used for embedding.  
        **`text`** : `str` or `list[str]`  
            Raw text to be embedded.  
        **`convert_to_numpy`** : `bool`, optional  
            Converts embeddings to numpy array if True. The default is True.  
        **`normalize_embeddings`** : `bool`, optional  
            Normalizes embedding vector to have length 1. The default is True.  
        **`DEVICE`** : `str`, optional  
            Computation device used for embedding generation. The default is 'cpu'.  
        **`BATCH_SIZE`** : `int`, optional  
            Batch size for embedding generation. The default is 32.  
        **`PRECISION`** : `str`, optional  
            Numerical precision for embedding array. The default is "float32".  

         <span style="color: darkblue;">Returns</span>
        -------
        **`embeddings`** : `numpy.ndarray` or `torch.Tensor`.  
            LLM embeddings of raw text.

        """
        embeddings = model.encode(text,
                                  convert_to_numpy = convert_to_numpy,
                                  normalize_embeddings = normalize_embeddings,
                                  device = DEVICE, 
                                  batch_size = BATCH_SIZE,
                                  show_progress_bar = True,
                                  precision = PRECISION)
        
        return(embeddings)
    #-------------------------------------------------------------------------- 



    #--------------------------------------------------------------------------        
    # Chunking the input text to fit model input size limits
    #--------------------------------------------------------------------------     
    def chunk_text_by_words(self, text, chunk_size, overlap_percent = 0, strip_whitespace = True):
        """
        Splits `text` into chunks of `chunk_size` words with a given overlap percentage.  
        **Notes:**  
            - Overlap is applied in terms of words. For example, with chunk_size=10 and
              overlap_percent=20, each chunk will overlap the next by floor(10*0.20)=2 words.  
            - Step size is max(1, chunk_size - overlap_words) to avoid infinite loops.  
            
        <span style="color: darkred;">Parameters</span>
        ----------
        **`text`** : `str`  
            Raw text to be chunked.  
        **`chunk_size`** : `int`  
            Number of words per chunk (must be >= 1).  
        **`overlap_percent`** : `float`, optional  
            Percent overlap between chunked text. The default is 0.  
        **`strip_whitespace`** : `bool`, optional  
            Removes leading and trailing white spaces in text. The default is True.  

        Raises  
        ------  
        **`ValueError`**  
            Overlap percent is out of bounds.  

        <span style="color: darkblue;">Returns</span>  
        -------  
        **`chunks`** : `list[str]`
            List of chunked raw text.

        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
            
        
        # Normalize overlap_percent; accept values like 25 or 25.0 (meaning 25%)
        if overlap_percent < 0:
            raise ValueError("overlap_percent must be >= 0")
        if overlap_percent > 100:
            raise ValueError("overlap_percent must be <= 100")
        
        words = text.split()
        if not words:
            chunks = []
            
        else:
            overlap_words = np.floor(chunk_size * (overlap_percent / 100.0)).astype(int)
            step = max(1, chunk_size - overlap_words)
            
            chunks = []
            ii = 0
            n = len(words)
            while ii < n:
                chunk_words = words[ii:ii + chunk_size]
                chunk = " ".join(chunk_words)
                chunks.append(chunk.strip() if strip_whitespace else chunk)
                if ii + chunk_size >= n:
                    break
                ii += step
    
        return(chunks)
    #-------------------------------------------------------------------------- 
    
    
    
    #--------------------------------------------------------------------------        
    # split text to limit to word limits ensuring whole sentences in chunks
    #--------------------------------------------------------------------------  
    def chunk_text_sentences(self, text, max_words=50):
        """
        Splits `text` into chunks of whole sentences with total chunk size words   
        with less than or equal to `max_words`.  

        <span style="color: darkred;">Parameters</span>  
        ----------
        **`text`** : `str`  
            Raw text to be chunked.  
        **`max_words`** : `int`, optional  
            Maximum length of chunks in words. The default is 50.  

        <span style="color: darkblue;">Returns</span>  
        -------  
        **`chunks`** : `list[str]`  
            List of chunked raw text.  
        """
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks, current_chunk, current_count = [], [], 0
    
        for sent in sentences:
            words = sent.split()
            n_words = len(words)
            if current_count + n_words <= max_words:
                current_chunk.append(sent)
                current_count += n_words
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = [sent]
                current_count = n_words
    
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return(chunks)
    #--------------------------------------------------------------------------     

# =============================================================================



# =============================================================================
# Class containing functions for encryption and decryption for the encryption  
# RAG pipeline
# =============================================================================
class Enc_RAG_Encryption():
    """
    Class containing helper methods for encryption and decryption for the RAG 
    pipeline.  
    """
    #--------------------------------------------------------------------------        
    # Init
    #--------------------------------------------------------------------------  
    def __init__(self): 
        
        a = 1
    
    #--------------------------------------------------------------------------        



    #--------------------------------------------------------------------------        
    # Generate tenseal CKKS context
    #--------------------------------------------------------------------------  
    def get_ts_ckks_context(self):
        """
        Generates the CKKS TenSEAL context for encryption and decryption.  
        
        <span style="color: darkred;">Parameters</span>  
        ----------
        None  

        <span style="color: darkblue;">Returns</span>
        -------
        **`context`** : `TenSEAL.context`  
            CKKS TenSEAL context for encryption.  
        """
        context = ts.context(ts.SCHEME_TYPE.CKKS, 
                             poly_modulus_degree=8192, 
                             coeff_mod_bit_sizes=[60, 40, 40, 60])
        context.global_scale = pow(2, 40)
        context.generate_galois_keys()
        return(context)

    #--------------------------------------------------------------------------        
    
    

# =============================================================================




# =============================================================================
# Class containing functions for preparing the orphanet database 
# Temporary - will be moved to dedicated API file
# =============================================================================
class RD_Orphanet_DB():
    """
    Class containing helper methods for preparing the Orphadata science dataset 
    for analysis.  
    """
    #--------------------------------------------------------------------------        
    # Init
    #--------------------------------------------------------------------------  
    def __init__(self): 
        
        a = 1
    
    #--------------------------------------------------------------------------        

    #--------------------------------------------------------------------------        
    # Aggregate Orphanet HPO terms to disease 
    #--------------------------------------------------------------------------      
    def aggregate_hpo_terms(self, df, frequency_filter=None):
        """
        Aggregates HPO_Term by DisorderName with an optional FrequencyName filter.  

        <span style="color: darkred;">Parameters</span>
        ----------
        **`df`** : `pd.DataFrame`
            DataFrame of Orphadata HPO terms.
        frequency_filter : `list[str]`, optional
            List of disease frequencies to be included. The default is None.

        <span style="color: darkblue;">Returns</span>  
        -------
        **`agg_df`** : `pd.DataFrame`  
            DataFrame with DisorderName and aggregated list of HPO_Term.

        """
        # Apply filter if provided
        if frequency_filter is not None:
            if isinstance(frequency_filter, str):
                freq_list = [frequency_filter]
            else:
                freq_list = frequency_filter
            filtered_df = df[df['FrequencyName'].isin(freq_list)]
        else:
            filtered_df = df.copy()
    
        # Group by DisorderName and aggregate HPO_Term into lists
        agg_df = filtered_df.groupby('DisorderName')['HPO_Term'].apply(list).reset_index()
    
        return(agg_df)
    #--------------------------------------------------------------------------        
     

# =============================================================================





