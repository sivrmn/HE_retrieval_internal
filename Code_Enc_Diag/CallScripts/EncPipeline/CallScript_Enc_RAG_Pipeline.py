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

# Main script for demonstrating the encrypted RAG pipeline on synthetic clinical 
# notes with Orphadata knowledgebase

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
from APIs.EncPipeline.Enc_RAG_Pipeline import Enc_RAG_Embeddings, Enc_RAG_Encryption
from APIs.Validation.Validation_Enc_Diag import Enc_RAG_Validation
from APIs.Utils.Utils import Utils
# =============================================================================


#=============================================================================#
#========================= Instantiate Objects ===============================#
UObj = Utils()

EMBObj = Enc_RAG_Embeddings()
ENCObj = Enc_RAG_Encryption()

VALObj = Enc_RAG_Validation()


#=============================================================================#

#%%
#=============================================================================#
#========================= Load the data ==================================#


#-----------------------------------------------------------------------------#
#----------------------Orphanet Knowledgebase---------------------------------#
# Preparing the Orphanet corpus
op_df = pd.read_csv("../../../Processed_Data/knowledgebase_Orphadata_merged.csv")
corpus = list(op_df['Summary'])


disease_names = list(op_df['DisorderName'])
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
#---------------------------Get synthetic clinical notes----------------------#
rd_curated_df = pd.read_csv("../../../Processed_Data/rd_synthetic_SO_sections.csv")
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# Get meta data of RD curated SO sections
for ii in range(0,np.size(rd_curated_df,0),1):
    
    c_note = rd_curated_df.loc[ii,'CLINICAL_NOTE_SO']
    
    c_note = EMBObj.get_clean_text(c_note)
    
    c_note_word_size = len(c_note.split())

    rd_curated_df.loc[ii,'NOTE_SIZE_WORDS'] = c_note_word_size
#-----------------------------------------------------------------------------#    
    

#-----------------------------------------------------------------------------#
# Get meta data of the orphanet RD data corpus
meta_corpus_df = pd.DataFrame([])
meta_corpus_df['CORPUS_TEXT'] = []
meta_corpus_df['SIZE_WORDS'] = []

 
for ii in range(0,np.size(corpus,0),1):

    dis_note = corpus[ii]    
    
    meta_corpus_df.loc[ii,'CORPUS_TEXT'] = dis_note
    meta_corpus_df.loc[ii,'SIZE_WORDS'] = len(dis_note.split())
#-----------------------------------------------------------------------------#

#=============================================================================#


#%%
#=============================================================================#
#========================= Run Experiment ====================================#

# Parameters and constants
DEVICE = EMBObj.get_device()
BATCH_SIZE = 1
PRECISION = "float32" 

# Experiment Parameters
NORMALIZE_EMBEDDINGS = True
TOP_K = 20
INCLUDE_ENCRYPTION = True



#-----------------------------------------------------------------------------#
# Select model

# model_name = 'sentence-transformers/all-MiniLM-L6-v2'
# model_name_abbr = 'all_minim_l6_v2'

# model_name = 'sentence-transformers/all-mpnet-base-v2'
# model_name_abbr = 'all_mpnet_base_v2'

# model_name = 'pritamdeka/S-PubMedBert-MS-MARCO'
# model_name_abbr = 'pritamdeka_pubmedbert'

# model_name = 'sentence-t5-xxl'
# model_name_abbr = 'sentence_t5_xxl'

model_name = 'Qwen/Qwen3-Embedding-8B'
model_name_abbr = 'Qwen3_8B'
#-----------------------------------------------------------------------------#


# Download/load model
embed_model = EMBObj.select_st_model(model_name)
print('Max input sequence length: ' + str(embed_model.get_max_seq_length()))

#%% 
#-----------------------------------------------------------------------------#
#--- PERFORMED BY SERVER ONCE ---#
# Get the knowledgebase embeddings
#-----------------------------------------------------------------------------#
time_start = time.time()

emb_corpus = embed_model.encode_document(corpus, 
                                      normalize_embeddings = NORMALIZE_EMBEDDINGS,
                                      device = DEVICE,
                                      batch_size = BATCH_SIZE,
                                      precision = PRECISION,
                                      show_progress_bar = True)
 

time_end = time.time()
print('Time for corpus embeddings = ' + str(time_end - time_start))
#-----------------------------------------------------------------------------#

#%%
#-------------------------- Retrieval Pipeline -------------------------------#

#-----------------------------------------------------------------------------#
#--- PERFORMED BY CLIENT ---#
# CKKS encryption settings
#-----------------------------------------------------------------------------#
if(INCLUDE_ENCRYPTION == True):
    ts_context = ENCObj.get_ts_ckks_context()
    secret_key = ts_context.secret_key()
    ts_context.make_context_public()
#-----------------------------------------------------------------------------#



#-----------------------------------------------------------------------------#
#--- PERFORMED BY SERVER ---#
# Encrypt the corpus embeddings 
#-----------------------------------------------------------------------------#
if(INCLUDE_ENCRYPTION == True):
    emb_corpus_enc = []
    
    for ii in range(0,np.size(emb_corpus,0),1):
        vec = emb_corpus[ii,:]
        emb_corpus_enc.append(ts.ckks_vector(ts_context, vec))
#-----------------------------------------------------------------------------#


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
# Cycle through all clinical notes
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
results_total_df = pd.DataFrame([])
results_total_df['Test File ID'] = []
results_total_df['Query'] = []
results_total_df['Top K with encryption'] = []
results_total_df['Top K without encryption'] = []

for file_cnt in tqdm(range(0,np.size(rd_curated_df,0),1)):
    
    try:
            
    
        #-----------------------------------------------------------------------------#
        # Define the query
        #-----------------------------------------------------------------------------#
        query = EMBObj.get_clean_text(rd_curated_df.loc[file_cnt,'CLINICAL_NOTE_SO'])
        #-----------------------------------------------------------------------------#
        
        
        #-----------------------------------------------------------------------------#
        # Embed the query
        #-----------------------------------------------------------------------------#
        emb_query = embed_model.encode_query(query,
                                 normalize_embeddings = NORMALIZE_EMBEDDINGS,
                                 device = DEVICE,
                                 batch_size = BATCH_SIZE,
                                 precision = PRECISION,
                                 show_progress_bar = False)
        #-----------------------------------------------------------------------------#
        
        
        #=============================================================================#
        # Retrieval without encryption 
        #=============================================================================#
        
        #-----------------------------------------------------------------------------#
        # Compute dot product similarity
        #-----------------------------------------------------------------------------#
        similarities = np.dot(emb_corpus, emb_query)
        #-----------------------------------------------------------------------------#
        
        
        #-----------------------------------------------------------------------------#
        # Rank the disease similarities
        #-----------------------------------------------------------------------------#
        cosine_similarity_ranking = []
        for ii in range(len(similarities)):
            cosine_similarity_ranking.append([disease_names[ii], round(float(abs(similarities[ii])), 2)])
            
        search_results = pd.DataFrame(cosine_similarity_ranking, columns = ['Disease', 'Score'])
        search_results = search_results.sort_values('Score', ascending = False)
        #-----------------------------------------------------------------------------#
        
        #-----------------------------------------------------------------------------#
        # Retrieve top K most similar results
        #-----------------------------------------------------------------------------#
        print("\n\n Retrieval without encryption")
        print("\nQuery:", query)
        print("\n Top K retrieval results:\n", search_results[0:TOP_K].to_string(index = False))
        #-----------------------------------------------------------------------------#
        
        #=============================================================================#
        
        
        #=============================================================================#
        # Retrieval with encryption
        #=============================================================================#
        if(INCLUDE_ENCRYPTION == True):
        
            #-----------------------------------------------------------------------------#
            #--- PERFORMED BY CLIENT ---#
            # Encrypt the query
            #-----------------------------------------------------------------------------#
            emb_query_enc = ts.ckks_vector(ts_context, emb_query)
            #-----------------------------------------------------------------------------#
            
            
            #-----------------------------------------------------------------------------#
            #--- PERFORMED BY SERVER ---#
            # Compute dot product similarity
            #-----------------------------------------------------------------------------#
            enc_similarities = []
            
            for ii in range(0,np.size(emb_corpus_enc,0),1):
                vec = emb_corpus_enc[ii]
                enc_similarities.append(emb_query_enc.dot(vec))
            #-----------------------------------------------------------------------------#
            
            
            #-----------------------------------------------------------------------------#
            #--- PERFORMED BY CLIENT ---#
            # Decrypt similarity and find the top K vectors
            #-----------------------------------------------------------------------------#
            decrypted_similarities = []
            
            for ii in range(0,np.size(enc_similarities,0),1):
                decrypted_similarities.append(enc_similarities[ii].decrypt(secret_key))
            #-----------------------------------------------------------------------------#
            
            
            #-----------------------------------------------------------------------------#
            #--- PERFORMED BY SERVER ---#
            # Rank the disease similarities
            #-----------------------------------------------------------------------------#
            enc_cosine_similarity_ranking = []
            for ii in range(len(decrypted_similarities)):
                enc_cosine_similarity_ranking.append([disease_names[ii], round(decrypted_similarities[ii][0], 2)])
                
            
            enc_search_results = pd.DataFrame(enc_cosine_similarity_ranking, columns = ['Disease', 'Score'])
            enc_search_results = enc_search_results.sort_values('Score', ascending = False)
            #-----------------------------------------------------------------------------#
            
            
            #-----------------------------------------------------------------------------#
            #--- PERFORMED BY SERVER ---#
            # Retrieve top K most similar results
            #-----------------------------------------------------------------------------#
            print("\n\n Retrieval with encryption")
            print("\n Top K retrieval results:\n", enc_search_results[0:TOP_K].to_string(index = False))
            #-----------------------------------------------------------------------------#
            
        
        #-----------------------------------------------------------------------------#
        # Save results for analysis
        #-----------------------------------------------------------------------------#
        disease_abbr = rd_curated_df.loc[file_cnt, 'DISEASE_ABBR']
        
        results_total_df.loc[file_cnt, 'Test File ID'] = rd_curated_df.loc[file_cnt, 'FILE_NAME']
        results_total_df.loc[file_cnt, 'Query'] = query
        results_total_df.loc[file_cnt, 'Disease name'] = disease_abbr
        
        
        # Results with encryption
        if(INCLUDE_ENCRYPTION == True):
            
            enc_srch_df = enc_search_results.reset_index()
            
            enc_disease_rank = VALObj.get_dis_rank(enc_srch_df, disease_abbr)
            enc_ndcg = VALObj.get_ndcg_single_output(enc_disease_rank, TOP_K)
            
            results_total_df.loc[file_cnt, 'Top K with encryption'] = enc_search_results[0:TOP_K].to_string(index = False)
            results_total_df.loc[file_cnt, 'Rank with encryption'] = enc_disease_rank
            results_total_df.loc[file_cnt, 'NDCG with encryption'] = enc_ndcg
            
        else:
            
            results_total_df.loc[file_cnt, 'Top K with encryption'] = np.nan
            results_total_df.loc[file_cnt, 'Rank with encryption'] = np.nan
            results_total_df.loc[file_cnt, 'NDCG with encryption'] = np.nan
        
        # Results without encryption
        srch_df = search_results.reset_index()
        
        disease_rank = VALObj.get_dis_rank(srch_df, disease_abbr)
        ndcg = VALObj.get_ndcg_single_output(disease_rank, TOP_K)
        
        results_total_df.loc[file_cnt, 'Top K without encryption'] = search_results[0:TOP_K].to_string(index = False)
        results_total_df.loc[file_cnt, 'Rank without encryption'] = disease_rank
        results_total_df.loc[file_cnt, 'NDCG without encryption'] = ndcg
        
    except:
        print('Error, moving to next case report.')
        
    #-----------------------------------------------------------------------------#

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#





#%%

UObj.folder_check('../Results/')

if(INCLUDE_ENCRYPTION == True):
    results_total_df.to_csv('../Results/Gemini2p5flash_testSO_v4_' + model_name_abbr + '.csv')
else:
    results_total_df.to_csv('../Results/Gemini2p5flash_testSO_v4_NEC_' + model_name_abbr + '.csv')


#=============================================================================#















