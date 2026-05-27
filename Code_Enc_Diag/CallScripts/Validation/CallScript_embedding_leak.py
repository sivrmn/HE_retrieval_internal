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

# Code for extracting information from LLM embeddings by embedding inversion

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
import vec2text, torch
from vec2text.models import InversionModel, CorrectorEncoderModel
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel



# General
import sys
import os
sys.path.append('../../')
import re


# Internal Files
from APIs.EncPipeline.Enc_RAG_Pipeline import Enc_RAG_Embeddings, Enc_RAG_Encryption, RD_Orphanet_DB
from APIs.Validation.Validation_Enc_Diag import Enc_RAG_Validation
from APIs.Plots.Enc_Diag_Plots import Enc_RAG_Plots
from APIs.Utils.Utils import Utils
# =============================================================================


#=============================================================================#
#========================= Instantiate Objects ===============================#
UObj = Utils()

EMBObj = Enc_RAG_Embeddings()
ENCObj = Enc_RAG_Encryption()
RDObj = RD_Orphanet_DB()

VALObj = Enc_RAG_Validation()
PLTObj = Enc_RAG_Plots()

#=============================================================================#

#%%
#=============================================================================#
#== Create embeddings and show how they can be recovered =====================#

DEVICE = EMBObj.get_device()

rd_curated_df = pd.read_csv("../../../Raw_data/rd_curated_df.csv")


#-----------------------------------------------------------------------------#
# Clean up text for processing
#-----------------------------------------------------------------------------#
def get_clean_text(text):
    
    # Remove control, zero-width, and NBSP characters
    text = re.sub(r"[\u200B-\u200D\uFEFF\u00A0]", " ", text)
    
    # Normalize quotes/dashes
    replacements = {
    "“": '"', "”": '"', "‘": "'", "’": "'", "–": "-", "—": "-",
    "…": "...", "′": "'", "″": '"'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    # Replace multiple spaces or line breaks with a single space
    clean_text = re.sub(r'\s+', ' ', text).strip()
    

    return(clean_text)
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Create embeddings for vec2text
#-----------------------------------------------------------------------------#
def get_gtr_embeddings(text_list,
                       encoder: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer,
                       device: str = "cpu"):

    inputs = tokenizer(text_list,
                       return_tensors="pt",
                       max_length=512,
                       truncation=True,
                       padding="max_length",).to(DEVICE)

    with torch.no_grad():
        model_output = encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        hidden_state = model_output.last_hidden_state
        embeddings = vec2text.models.model_utils.mean_pool(hidden_state, inputs['attention_mask'])

    return(embeddings)
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Select model

model_name = 'gtr-t5-base'
model_name_abbr = 'gtr_base'

#-----------------------------------------------------------------------------#



#-----------------------------------------------------------------------------#
# Load model and corrector model

embed_model = st.SentenceTransformer(model_name)  

encoder = AutoModel.from_pretrained('sentence-transformers/' + model_name).encoder.to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/' + model_name)

corrector = vec2text.load_pretrained_corrector("gtr-base")
#-----------------------------------------------------------------------------#




#-----------------------------------------------------------------------------#
# Run through the example files to show recovery of embeddings
df_leak = pd.DataFrame([]) 
df_leak['CLINICAL_EXCERPT'] = []

df_leak['EMBEDDING'] = []
df_leak['EMBEDDING'] = df_leak['EMBEDDING'].astype(object)
df_leak['RECON_ORIGINAL'] = []

df_leak['EMBEDDING_ENC'] = []
df_leak['EMBEDDING_ENC'] = df_leak['EMBEDDING_ENC'].astype(object)
df_leak['RECON_ENC'] = []

df_leak['EMBEDDING_DEC'] = []
df_leak['EMBEDDING_DEC'] = df_leak['EMBEDDING_DEC'].astype(object)
df_leak['RECON_DEC'] = []



lcl_cnt = 0


for file_cnt in tqdm(range(0,np.size(rd_curated_df,0),1)):
    
    
    
    #----- Select a clinical note ----#
    cn = get_clean_text(rd_curated_df.loc[file_cnt,'CLINICAL_NOTE_SO'])
    
    chunks = EMBObj.chunk_text_sentences(cn, 30)
    
    cn_excerpt = chunks[0]
    #---------------------------------#
    
    
    #--- Embed the query ---#
    # cn_embedding = embed_model.encode(cn_excerpt, 
    #                                      convert_to_numpy=True, 
    #                                      normalize_embeddings=False,
    #                                      device = DEVICE)
    #---------------------------------#
    

    
    #----- Generate embeddings using llm ----#
    embeddings = get_gtr_embeddings([cn_excerpt], encoder, tokenizer, device = DEVICE)
    
    #---------------------------------#
    
    #------ Encryption with CKKS --------#
    ts_context = ENCObj.get_ts_ckks_context()
    secret_key = ts_context.secret_key()
    ts_context.make_context_public()
    #------------------------------------#
    
    
    #----- Reconstruction using original embedding ----#
    recon_original = vec2text.invert_embeddings(embeddings = embeddings,
                                                corrector = corrector,
                                                num_steps = 15)
    #--------------------------------------#
    
    
    #----- Encrypting the embedding ----#
    emb_cpu = embeddings.detach().to("cpu").contiguous()
    if emb_cpu.ndim == 1:
         emb_cpu = emb_cpu.unsqueeze(0)  
    
    embeddings_vec = np.array(emb_cpu[0])
    
    enc_vec = ts.ckks_vector(ts_context, embeddings_vec.T)
    serialized = enc_vec.serialize()
    #---------------------------------#
    
    

    #----- Reconstruction using naive interpretation ----#
    ser_bytes = []
    for a in serialized: ser_bytes.append(a)
    byte_data = bytes(ser_bytes)
    target_dim = 768
    
        
    arr = (np.frombuffer(byte_data[0:target_dim], dtype=np.uint8).astype(np.float32))/255
    
    naive_embedding = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    naive_embedding = naive_embedding.to(DEVICE)
    
    
    recon_encrypted = vec2text.invert_embeddings(embeddings = naive_embedding,
                                                corrector = corrector,
                                                num_steps = 15)
    #---------------------------------#


    #----- Reconstruction using decrypted embedding ----#
    decrypted_vec = enc_vec.decrypt(secret_key)
    
    embeddings_decrypted = torch.from_numpy(np.asarray(decrypted_vec, dtype=np.float32)).unsqueeze(0).to(DEVICE)
    
    
    recon_decrypted = vec2text.invert_embeddings(embeddings = embeddings_decrypted,
                                                corrector = corrector,
                                                num_steps = 15)
    #---------------------------------#



    df_leak.loc[lcl_cnt, 'CLINICAL_EXCERPT'] = cn_excerpt
    
    df_leak.at[lcl_cnt, 'EMBEDDING'] = embeddings
    df_leak.loc[lcl_cnt, 'RECON_ORIGINAL'] = recon_original[0]
    
    df_leak.at[lcl_cnt, 'EMBEDDING_ENC'] = naive_embedding
    df_leak.loc[lcl_cnt, 'RECON_ENC'] = recon_encrypted[0]
    
        
    df_leak.at[lcl_cnt, 'EMBEDDING_DEC'] = embeddings_decrypted
    df_leak.loc[lcl_cnt, 'RECON_DEC'] = recon_decrypted[0]
      
    
    lcl_cnt = lcl_cnt + 1


    UObj.folder_check('../Results/DataLeak/')

    df_leak.to_csv('../Results/DataLeak/'+model_name_abbr+'_data_leak_v2.csv')

#=============================================================================#


