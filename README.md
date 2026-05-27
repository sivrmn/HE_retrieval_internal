# HE Retrieval: Privacy-Preserving Rare Disease Diagnosis

`HE Retrieval` is a privacy-preserving diagnostic retrieval framework that combines large language model (LLM) embeddings with homomorphic encryption (HE). The project demonstrates how a clinical note can be converted into an embedding, encrypted on the client side, compared against a rare disease knowledge base in encrypted space, and then ranked after decryption by the client.

The motivating use case is rare disease diagnosis, where patient histories may contain protected health information (PHI). Instead of sending raw clinical notes or the corresponding embeddings to an external service, the framework sends encrypted vectors and performs similarity search without exposing the underlying clinical text.


## Project Structure

```
project-root/
├──Code_Enc_Diag/---------------------------------------# Code for encrypted retrieval, validation, and result replication
│   ├──APIs---------------------------------------------# Reusable modules and fuctions for encrypted retrieval (see Code_Enc_Diag/docs/index.html for details)
│   ├──CallScripts                                 
│   │   ├──EncPipeline 
│   │   │   ├──CallScript_Enc_RAG_pipeline.py-----------# Main encrypted retrieval pipeline script  
│   │   ├──Plots                              
│   │   │   ├──Callscript_Enc_Diag_Plot1_Dis.py---------# Plots manuscript Figure 3
│   │   │   ├──CallScript_Enc_Diag_Plot2_LLMs.py--------# Plots manuscript Figure 4
│   │   │   ├──CallScript_Enc_Diag_Plot3_EncTradeoff.py-# Plots manuscript Figure 5
│   │   ├──Results--------------------------------------# Data from experiments
│   │   ├──Validation
│   │   │   ├──Callscript_embedding_leak.py-------------# Validation experiment demonstrating embedding leaks                 
│   │   │   ├──Callscript_enc_tradeoffs.py--------------# Validation experiment testing ecryption parameters
│   ├──docs                                        
│       ├──index.html-----------------------------------# Details of encrypted retrieval helper modules
├──Code_Rare_Dis----------------------------------------# Code for processing data
│   ├──APIs---------------------------------------------# Reusable modules and functions
│   ├──CallScripts
│       ├──ProcessData
│   │   │   ├──A1_KB_Script_PrepOrphaData.py------------# Script for creating RD knowledge base using Orphadata Science (part 1 of 2)
│   │   │   ├──A2_KB_Script_PrepOrphaCorpus.py----------# Script for creating RD knowledge base using Orphadata Science (part 2 of 2)
│   │   │   ├──B1_CN_Script_PrepCaseReportPrompts.py----# Script for creating syntethic RD SO sections - prompts from PubMed Excerpts (part 1 of 2)
│   │   │   ├──B2_CN_Script_MakeRDTestReports.py--------# Script for creating syntethic RD SO sections - LLM clinical notes to test DataFrame (part 2 of 2)
├──Processed_Data
│   ├──knowledgebase_Orphadata_merged.csv---------------# RD knowledge base creating using Orphadata Science
│   ├──rd_synthetic_SO_sections.csv---------------------# RD sythetic SO sections for testing pipeline
├──Raw_Data
│   ├──Case_Reports-------------------------------------# Raw data for synthetic SO sections (case report excerpts, prompts, LLM outputs)
│   ├──OrphaData----------------------------------------# Raw data from Orphadata Science 
├──HE_retrieval_install.yml-----------------------------# Configuration file for Conda environment
```

## Installation and Setup

### Prerequisites

Python 3.10 or higher is recommended.

### Clone the Repository

```bash
git clone https://github.com/sivrmn/HE_retrieval.git
cd HE_retrieval
```

### Install Dependencies

Create a Conda environment using the provided [`HE_retrieval_install.yml`](HE_retrieval_install.yml) file:
```bash
conda env create -f HE_retrieval_install.yml -n your_environment_name
```

Then activate the environment:

```bash
conda activate your_environment_name
```

## Data

The manuscript uses Orphadata Science derived rare disease information to build disease profiles. 
Synthetic test notes are used for evaluation and should not be treated as real patient records.

## Citation

If you use this project, please cite the associated manuscript:

```text
Rajaganapathy, Sivaraman, Jennifer St. Sauver, Filippo Pinto e Vairo, Prasad G. Iyer, Hongfang Liu, and Jungwei W. Fan. "Hide and Seek: Privacy-Preserving Artificial Intelligence with a Feasibility Study in Rare Disease Diagnosis." medRxiv (2026): 2026-01.
```

## Authors

- Sivaraman Rajaganapathy
- Jennifer St. Sauver
- Filippo Pinto e Vairo
- Prasad G. Iyer
- Hongfang Liu
- Jungwei W. Fan (Corresponding Author)


## Funding

This work was supported by the U.S. National Institutes of Health grants UL1TR002377 and R01HG012748.


## Project Status

Research prototype. The framework is intended for feasibility testing and methodological development. 
It is not intended for clinical or medical use. It has not been reviewed or approved by any medical or regulatory authorities. 
