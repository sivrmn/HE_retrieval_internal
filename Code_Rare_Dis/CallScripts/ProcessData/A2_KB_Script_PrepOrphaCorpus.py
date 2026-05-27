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

# Script for preparing an RD knowledgebase with disease profiles using the 
# data extracted from Orphadata Science XMLs

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

# Data analysis
import xml.etree.ElementTree as ET

import sys
sys.path.append('../../')
import os

# Internal Files
from APIs.Utils.Utils import Utils
# =============================================================================

#=============================================================================#
#========================= Instantiate Objects ===============================#
UObj = Utils()
#=============================================================================#


#=============================================================================#
#============================= Define paths ==================================#
load_path = "../../../Raw_data/OrphaData/"

save_path = "../../../Processed_Data/"
UObj.folder_check(save_path)
#=============================================================================#



#=============================================================================#
#============================= Load Data =====================================#

df_op_terms = pd.read_csv(load_path + 'df_Orphadata_terminology.csv')

df_op_lin = pd.read_csv(load_path + 'df_Orphadata_linearization.csv')

df_op_func = pd.read_csv(load_path + 'df_Orphadata_functional_consequences.csv')

df_op_pheno = pd.read_csv(load_path + 'df_Orphadata_phenotypes.csv')

df_op_gene = pd.read_csv(load_path + 'df_Orphadata_genes.csv')

df_op_nat_hist = pd.read_csv(load_path + 'df_Orphadata_natural_histories.csv')

df_op_epid = pd.read_csv(load_path + 'df_Orphadata_epidemiology.csv')

#=============================================================================#

#%%
#=============================================================================#
#======= Create a knowledge base from Orpha Data components===================#


df = df_op_pheno
#------------------------------------------------------------------------------
# Function to create knowledgebase text for orphadata phenotypes
#------------------------------------------------------------------------------
#----------------------------------
# Clean and prepare data
columns_used = [
    "OrphaCode",
    "DisorderName",
    "HPO_Term",
    "FrequencyName",
    "DiagnosticCriteria"
]

df = df[columns_used].copy()

for col in ["DisorderName", "HPO_Term", "FrequencyName", "DiagnosticCriteria"]:
    df[col] = df[col].astype(str).str.strip()
#----------------------------------


#----------------------------------
# Define frequency order
freq_order = {
    "Obligate (100%)": 0,
    "Very frequent (99-80%)": 1,
    "Frequent (79-30%)": 2,
    "Occasional (29-5%)": 3,
    "Very rare (<4-1%)": 4,
    "Excluded (0%)": 5
}
#----------------------------------

#----------------------------------
# Define dictionary for phenotype frequency term in setences
freq_dict_pheno = {
    "Obligate (100%)":  "obligately (100%) associated",
    "Very frequent (99-80%)": "very frequently (99-80%) associated",
    "Frequent (79-30%)": "frequently (79-30%) associated",
    "Occasional (29-5%)": "occasionally (29-5%) associated",
    "Very rare (<4-1%)": "very rarely (<4-1%) associated",
    "Excluded (0%)": "not associated (0%) associated"
}
#----------------------------------



#----------------------------------
# Initialize results list
records = []

#------------------------------------------------------------------------------
# Iterate through each unique (OrphaCode, DisorderName)
#------------------------------------------------------------------------------
unique_disorders = df[["OrphaCode", "DisorderName"]].drop_duplicates()
unique_disorders = unique_disorders.reset_index(drop = True)

for ii in range(0, np.size(unique_disorders,0), 1):
    
    #--------------------------------------------------
    # Identify current disease
    #--------------------------------------------------
    row = unique_disorders.loc[ii, :]
    
    current_orphacode = row["OrphaCode"]
    current_disorder = row["DisorderName"]
    
    #--------------------------------------------------
    # Extract all rows for this disorder
    #--------------------------------------------------
    disorder_df = df[
        (df["OrphaCode"] == current_orphacode) &
        (df["DisorderName"] == current_disorder)
    ]
    
    #--------------------------------------------------
    # Create phenotype sentences by frequency
    #--------------------------------------------------
    pheno_sentences = []
    
    # Get list of unique frequencies for this disorder
    freq_list = sorted(
        disorder_df["FrequencyName"].dropna().unique(),
        key=lambda f: freq_order.get(f, 999)
    )
    
    for freq_name in freq_list:
        
        # Filter for this frequency
        freq_df = disorder_df[disorder_df["FrequencyName"] == freq_name]
        
        # Collect unique HPO terms
        hpo_terms = sorted(set(
            term for term in freq_df["HPO_Term"].dropna()
            if term and term.lower() != "nan"
        ))
        
        if len(hpo_terms) == 0:
            continue
        
        # Create readable text
        if len(hpo_terms) == 1:
            terms_text = hpo_terms[0]
        elif len(hpo_terms) == 2:
            # terms_text = f"{hpo_terms[0]} and {hpo_terms[1]}" 
            terms_text = str(hpo_terms[0]) +" and " + str(hpo_terms[1]) 
        else:
            # terms_text = ", ".join(hpo_terms[:-1]) + f", and {hpo_terms[-1]}"
            terms_text = ", ".join(hpo_terms[:-1]) + ", and " + str(hpo_terms[-1]) 
        
        freq_term = freq_dict_pheno.get(freq_name, "associated with unknown frequency")
        
        sentence = f"{current_disorder} is {freq_term} with {terms_text}."
        pheno_sentences.append(sentence)
    
    #--------------------------------------------------
    # Add diagnostic criteria (if any)
    #--------------------------------------------------
    # diag_list = sorted(set(
    #     crit for crit in disorder_df["DiagnosticCriteria"].dropna()
    #     if crit and crit.lower() != "nan"
    # ))
    
    # if len(diag_list) > 0:
    #     diag_text = "; ".join(diag_list)
    #     pheno_sentences.append(f"Diagnostic criteria include: {diag_text}.")
    
    #--------------------------------------------------
    # Combine everything into final text
    #--------------------------------------------------
    phenotype_summary = " ".join(pheno_sentences).strip()
    
    #--------------------------------------------------
    # Append to results
    #--------------------------------------------------
    records.append({
        "OrphaCode": current_orphacode,
        "DisorderName": current_disorder,
        "Summary": phenotype_summary
    })

#------------------------------------------------------------------------------
# Create final DataFrame and save
#------------------------------------------------------------------------------
kb_op_pheno = pd.DataFrame(records)
kb_op_pheno.to_csv(save_path + 'knowledgebase_Orphadata_phenotypes_only.csv')


#==============================================================================



#%% Functional Consequences
#==============================================================================
# Build a simple functional knowledge base from df_op_func.csv
#   - One row per (OrphaCode, DisorderName)
#   - Each summary mentions DisabilityName + FrequencyName + TemporalityName + SeverityName
#==============================================================================



#------------------------------------------------------------------------------
# Load source data
#------------------------------------------------------------------------------

df = df_op_func

#------------------------------------------------------------------------------
# Keep only needed columns and clean whitespace
#------------------------------------------------------------------------------
columns_used = [
    "OrphaCode",
    "DisorderName",
    "DisabilityName",
    "FrequencyName",
    "TemporalityName",
    "SeverityName"
]
df = df[columns_used].copy()

for col in columns_used:
    df[col] = df[col].astype(str).str.strip()

#------------------------------------------------------------------------------
# Optional: smoother wording for frequency
#------------------------------------------------------------------------------
freq_phrase = {
    "Obligate (100%)":  "obligately (100%)",
    "Very frequent (99-80%)": "very frequently (99-80%)",
    "Frequent (79-30%)": "frequently (79-30%)",
    "Occasional (29-5%)": "occasionally (29-5%)",
    "Very rare (<4-1%)": "very rarely (<4-1%)",
    "Excluded (0%)": "not associated (0%)"
}

#------------------------------------------------------------------------------
# Build summaries explicitly (no groupby)
#------------------------------------------------------------------------------
records = []

unique_disorders = df[["OrphaCode", "DisorderName"]].drop_duplicates()

for _, row in unique_disorders.iterrows():
    current_orphacode = row["OrphaCode"]
    current_disorder  = row["DisorderName"]

    # Rows for this disorder
    sub = df[(df["OrphaCode"] == current_orphacode) & (df["DisorderName"] == current_disorder)].copy()

    # De-duplicate identical combos and sort for deterministic output
    sub = sub.drop_duplicates(subset=["DisabilityName", "FrequencyName", "TemporalityName", "SeverityName"])
    sub = sub.sort_values(["DisabilityName", "FrequencyName", "TemporalityName", "SeverityName"])

    sentences = []

    for _, r in sub.iterrows():
        disability  = r["DisabilityName"]
        freq        = r["FrequencyName"]
        temporality = r["TemporalityName"]
        severity    = r["SeverityName"]

        # Smooth frequency phrase with fallback
        freq_text = freq_phrase.get(freq, f"with {freq.lower()}")

        parts = []
        parts.append(f"{disability} occurs {freq_text}")

        if temporality and temporality.lower() not in ("nan", "none"):
            parts.append(f"and is {temporality.lower()}")

        if severity and severity.lower() not in ("nan", "none"):
            parts.append(f"with {severity.lower()} severity")

        sentence = " ".join(parts) + "."
        if len(sentences) == 0:
            sentence = f"For '{current_disorder}', " + sentence

        sentences.append(sentence)

    summary_text = " ".join(sentences).strip()

    records.append({
        "OrphaCode": current_orphacode,
        "DisorderName": current_disorder,
        "Summary": summary_text
    })

kb_op_func = pd.DataFrame.from_records(records)


kb_op_func.to_csv(save_path + 'knowledgebase_Orphadata_functional_consequences_only.csv')


#=============================================================================#



#%% 
#=============================================================================#

#------------------------------------------------------------------------------
# Load your Orphanet gene component CSV
#------------------------------------------------------------------------------
# Example: change the path if your file lives elsewhere


#------------------------------------------------------------------------------
# Function to create knowledgebase text for OrphaData gene associations
#------------------------------------------------------------------------------
#----------------------------------
# Clean and prepare data
columns_used = [
    "OrphaCode",
    "DisorderName",
    "AssociationType",
    "GeneName",
    "GeneSynonyms"
]

# If your CSV has slightly different column names, map them here
# (Uncomment and adjust as needed)
# rename_map = {
#     "AssociationTypeName": "AssociationType",
#     "GeneSymbol": "GeneName",
#     "Synonyms": "GeneSynonyms"
# }
# df_op_gene = df_op_gene.rename(columns=rename_map)

df = df_op_gene[columns_used].copy()

for col in ["DisorderName", "AssociationType", "GeneName", "GeneSynonyms"]:
    df[col] = df[col].astype(str).str.strip()

# Normalize empty-like strings to proper NaN
df.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NULL": np.nan}, inplace=True)

# Keep only rows that actually have a gene name (cannot summarize without it)
df = df[~df["GeneName"].isna()].copy()
#----------------------------------


#----------------------------------
# Small helpers
#----------------------------------
def split_synonyms(text):
    """
    Split a GeneSynonyms cell into a clean, de-duplicated list.
    Handles common separators: | ; , /
    """
    if pd.isna(text):
        return []
    raw = str(text)
    # Replace various separators with a comma, then split
    for sep in ["|", ";", "/", "•", "·"]:
        raw = raw.replace(sep, ",")
    parts = [p.strip() for p in raw.split(",")]
    parts = [p for p in parts if p]  # remove empties
    # De-duplicate while preserving order
    seen = set()
    uniq = []
    for p in parts:
        if p.lower() not in seen:
            seen.add(p.lower())
            uniq.append(p)
    return uniq

def join_list_readable(items):
    """
    Oxford-comma style join for readability.
    """
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"
#----------------------------------


#----------------------------------
# Initialize results list
records = []

#------------------------------------------------------------------------------
# Iterate through each unique (OrphaCode, DisorderName)
#------------------------------------------------------------------------------
unique_disorders = df[["OrphaCode", "DisorderName"]].drop_duplicates()
unique_disorders = unique_disorders.reset_index(drop=True)

for ii in range(0, np.size(unique_disorders, 0), 1):

    #--------------------------------------------------
    # Identify current disease
    #--------------------------------------------------
    row = unique_disorders.loc[ii, :]
    current_orphacode = row["OrphaCode"]
    current_disorder  = row["DisorderName"]

    #--------------------------------------------------
    # Extract all rows for this disorder
    #--------------------------------------------------
    disorder_df = df[
        (df["OrphaCode"] == current_orphacode) &
        (df["DisorderName"] == current_disorder)
    ]

    #--------------------------------------------------
    # Build association-type specific segments
    #--------------------------------------------------
    assoc_sentences = []

    # Sorted unique association types for stable output
    assoc_types = sorted(
        set(a for a in disorder_df["AssociationType"].dropna()),
        key=lambda x: x.lower()
    )

    # If there is no association type in the data, treat as a single bucket
    if len(assoc_types) == 0:
        assoc_types = [None]

    for assoc in assoc_types:
        # Filter rows for this association
        if assoc is None:
            assoc_df = disorder_df.copy()
        else:
            assoc_df = disorder_df[disorder_df["AssociationType"] == assoc]

        # Collect unique gene phrases: "GeneName (Syn1; Syn2)" or just "GeneName"
        gene_phrases = []
        seen_genes = set()  # de-dupe by gene name (case-insensitive)
        for _, r in assoc_df.iterrows():
            gene = str(r["GeneName"]).strip()
            if gene.lower() in seen_genes:
                continue
            seen_genes.add(gene.lower())

            syns = split_synonyms(r.get("GeneSynonyms", np.nan))
            if len(syns) > 0:
                # Join synonyms with "; " to reduce commas in the main list
                syn_str = "; ".join(syns)
                phrase = f"{gene} ({syn_str})"
            else:
                phrase = gene
            gene_phrases.append(phrase)

        # Skip empty buckets
        if len(gene_phrases) == 0:
            continue

        # Human-friendly join of gene phrases
        genes_text = join_list_readable(gene_phrases)

        # Segment text per association type
        if assoc is None or str(assoc).strip() == "" or str(assoc).lower() == "nan":
            seg = f"{genes_text}"
        else:
            seg = f"{assoc} {genes_text}"

        assoc_sentences.append(seg)

    #--------------------------------------------------
    # Combine everything into final text
    #--------------------------------------------------
    if len(assoc_sentences) > 0:
        # Use semicolons between different association types to keep things tidy
        assoc_text = "; ".join(assoc_sentences)
        gene_summary = f"{current_disorder} is associated with {assoc_text}."
    else:
        gene_summary = ""  # no usable content

    #--------------------------------------------------
    # Append to results
    #--------------------------------------------------
    records.append({
        "OrphaCode": current_orphacode,
        "DisorderName": current_disorder,
        "Summary": gene_summary
    })

#------------------------------------------------------------------------------
# Create final DataFrame and (optionally) save
#------------------------------------------------------------------------------
kb_op_gene = pd.DataFrame(records).sort_values(["OrphaCode", "DisorderName"]).reset_index(drop=True)

# Optional: write to disk
# Set your save_path (ensure trailing slash if used)
kb_op_gene.to_csv(save_path + "knowledgebase_Orphadata_genes_only.csv", index=False)


#==============================================================================



#%%
# Pheno and Gene

#Merging the knowledgebases


kb_merged = pd.DataFrame([])

kb_merged['OrphaCode'] = []
kb_merged['DisorderName'] = []
kb_merged['Summary'] = []

op_codes = np.unique(kb_op_gene['OrphaCode']).astype(int)

for ii in range(0,np.size(kb_op_pheno,0),1):
    
    
    kb_merged.loc[ii,'OrphaCode'] = kb_op_pheno.loc[ii,'OrphaCode']
    kb_merged.loc[ii,'DisorderName'] = kb_op_pheno.loc[ii,'DisorderName']
    kb_merged.loc[ii,'Summary'] = kb_op_pheno.loc[ii,'Summary']
    
    
    curr_op_code = int(kb_merged.loc[ii,'OrphaCode'])
    
    if(curr_op_code in op_codes):
        
        op_chk = kb_op_gene['OrphaCode'].astype(int) == curr_op_code
        summary_2 = kb_op_gene.loc[op_chk,'Summary']

        kb_merged.loc[ii,'Summary'] = kb_merged.loc[ii,'Summary'] + summary_2.iloc[0]

kb_merged.to_csv(save_path + 'knowledgebase_Orphadata_merged.csv')




#%%
# Func and pheno
'''
#Merging the knowledgebases


kb_merged = pd.DataFrame([])

kb_merged['OrphaCode'] = []
kb_merged['DisorderName'] = []
kb_merged['Summary'] = []

op_codes_func = np.unique(kb_op_func['OrphaCode']).astype(int)

for ii in range(0,np.size(kb_op_pheno,0),1):
    
    
    kb_merged.loc[ii,'OrphaCode'] = kb_op_pheno.loc[ii,'OrphaCode']
    kb_merged.loc[ii,'DisorderName'] = kb_op_pheno.loc[ii,'DisorderName']
    kb_merged.loc[ii,'Summary'] = kb_op_pheno.loc[ii,'Summary']
    
    
    curr_op_code = int(kb_merged.loc[ii,'OrphaCode'])
    
    if(curr_op_code in op_codes_func):
        
        op_chk = kb_op_func['OrphaCode'].astype(int) == curr_op_code
        func_summary = kb_op_func.loc[op_chk,'Summary']

        kb_merged.loc[ii,'Summary'] = kb_merged.loc[ii,'Summary'] + func_summary.iloc[0]

kb_merged.to_csv(save_path + 'kb_op_merged_func_pheno.csv')


'''






