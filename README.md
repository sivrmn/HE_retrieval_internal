# HE Retrieval: Privacy-Preserving Rare Disease Diagnosis

`HE Retrieval` is a privacy-preserving diagnostic retrieval framework that combines large language model (LLM) embeddings with homomorphic encryption (HE). The project demonstrates how a clinical note can be converted into an embedding, encrypted on the client side, compared against a rare disease knowledge base in encrypted space, and then ranked after decryption by the client.

The motivating use case is rare disease diagnosis, where patient histories may contain protected health information (PHI). Instead of sending raw clinical notes or the corresponding embeddings to an external service, the framework sends encrypted vectors and performs similarity search without exposing the underlying clinical text.


## Project Structure

project-root/
+---Code_Enc_Diag/              # Code for encrypted retrieval and associated helper functions
¦   +---APIs                    # Reusable modules and fuctions
¦   ¦   +---EncPipeline         #
¦   ¦   +---Plots               #
¦   ¦   +---Utils               #
¦   ¦   +---Validation          #
¦   +---CallScripts             #
¦   ¦   +---EncPipeline         #
¦   ¦   +---Plots               #
¦   ¦   ¦   +---Plots           #
¦   ¦   +---Results             #
¦   ¦   ¦   +---DataLeak        #
¦   ¦   ¦   +---EncTradeoffs    #
¦   ¦   +---Validation          #
¦   +---docs                    #
¦       +---index.html          #
+---Code_Rare_Dis               #
¦   +---APIs                    #
¦   ¦   +---Utils               #
¦   +---CallScripts             #
¦       +---ProcessData         #
+---Processed_Data              #
+---Raw_Data                    #
    +---Case_Reports
    ¦   +---case_reports_excerpts
    ¦   +---case_reports_prompts
    ¦   +---clinical_notes_gemini2p5flash
    +---OrphaData
        +---XML


## Installation and Setup

### Prerequisites

Python 3.10 or higher is recommended.

This project uses Python packages for embeddings, encrypted tensor operations, numerical computation, and evaluation. The core dependencies are expected to include:

- `sentence-transformers`
- `tenseal`
- `numpy`
- `pandas`


### Clone the Repository

```bash
git clone <repository-url>
cd HE_retrieval
```

### Install Dependencies



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

## License

TBD

## Project Status

Research prototype. The framework is intended for feasibility testing and methodological development. 
It is not intended for clinical or medical use. It has not been reviewed or approved by any medical or regulatory authorities. 
