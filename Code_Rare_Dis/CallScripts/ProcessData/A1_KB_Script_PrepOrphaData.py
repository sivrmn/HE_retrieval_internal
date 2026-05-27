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

# Script for extracting relevant data from Orphadata Science  XMLs

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
save_path = "../../../Raw_data/OrphaData/"
UObj.folder_check(save_path)

# Point to the folder with Orphadata Science XML data available here: https://sciences.orphadata.com/orphanet-scientific-knowledge-files/
raw_orpha_path = "../../../Raw_Data/OrphaData/XML/" 


#%%


# Rare diseases and alignment with terminologies and databases
path_op_terms = raw_orpha_path + 'en_product1.xml'


#-----------------------------------------------------------------------------#
# Rare diseases and functional consequences
path_op_func = raw_orpha_path + 'en_funct_consequences.xml'
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Phenotypes Associated with Rare Disorders
path_op_pheno = raw_orpha_path + 'en_product4.xml'
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Genes associated with rare diseases
path_op_gene = raw_orpha_path + 'en_product6.xml'
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Linearization of rare diseases
path_op_lin = raw_orpha_path + 'en_product7.xml'
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Natural history of rare diseases
path_op_nat_hist = raw_orpha_path + 'en_product9_ages.xml'
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
# Epidemiology of Rare Diseases
path_op_epid = raw_orpha_path + 'en_product9_prev.xml'
#-----------------------------------------------------------------------------#

#=============================================================================#

#%%

#=============================================================================#
#======================== Extract RD Terminology Data ========================#
tree_op_terms = ET.parse(path_op_terms)
root_op_terms = tree_op_terms.getroot()

data = []

# Iterate through each Disorder
for disorder in root_op_terms.findall('.//Disorder'):
    # -------- Disorder context --------
    disorder_id   = disorder.attrib.get('id')
    orpha_code    = disorder.findtext('OrphaCode')
    expert_link   = disorder.findtext('ExpertLink')
    dname_el      = disorder.find('Name')
    disorder_name = dname_el.text if dname_el is not None else None

    disorder_type_name  = disorder.findtext('.//DisorderType/Name')
    disorder_group_name = disorder.findtext('.//DisorderGroup/Name')

    # Flags (multiple possible): collect "id:value:label" 
    flag_parts = []
    for flag in disorder.findall('.//DisorderFlagList/DisorderFlag'):
        fid = flag.attrib.get('id')
        fval = flag.findtext('Value')
        flab = flag.findtext('Label') or ""
        flag_parts.append(f"{fid}:{fval}:{flab}")
    flags_joined = "|".join(flag_parts) if flag_parts else None

    # Synonyms 
    syns = [s.text.strip() for s in disorder.findall('.//SynonymList/Synonym') if s.text]
    synonyms_joined = "|".join(syns) if syns else None

    # Definition text 
    definition = None
    for ts in disorder.findall('.//SummaryInformationList/SummaryInformation/TextSectionList/TextSection'):
        tstype = ts.findtext('TextSectionType/Name') or ""
        if tstype.strip().lower() == "definition":
            definition = ts.findtext('Contents')
            break

    # -------- External references (one row per reference) --------
    exts = disorder.findall('.//ExternalReferenceList/ExternalReference')
    if not exts:
        # If no external refs, still emit a disorder-level row (mapping fields None)
        data.append({
            'DisorderID': disorder_id,
            'OrphaCode': orpha_code,
            'DisorderName': disorder_name,
            'ExpertLink': expert_link,
            'DisorderType': disorder_type_name,
            'DisorderGroup': disorder_group_name,
            'Flags': flags_joined,
            'Synonyms': synonyms_joined,
            'Definition': definition,
            'ExternalRefID': None,
            'ExternalSource': None,
            'ExternalReference': None,
            'MappingRelationID': None,
            'MappingRelationName': None,
            'ICDRelationID': None,
            'ICDRelationName': None,
            'ValidationStatusID': None,
            'ValidationStatusName': None,
            'ICDRefUrl': None,
            'ICDRefUri': None
        })
        continue

    for ext in exts:
        ext_id   = ext.attrib.get('id')
        ext_src  = ext.findtext('Source')
        ext_ref  = ext.findtext('Reference')

        # Mapping relation
        mrel     = ext.find('DisorderMappingRelation')
        mrel_id  = mrel.attrib.get('id') if mrel is not None else None
        mrel_nm  = mrel.findtext('Name') if mrel is not None else None

        # ICD relation (may be empty/self-closing)
        icdrel     = ext.find('DisorderMappingICDRelation')
        icdrel_id  = icdrel.attrib.get('id') if (icdrel is not None and icdrel.attrib) else None
        icdrel_nm  = icdrel.findtext('Name') if icdrel is not None else None

        # Validation status
        vstat     = ext.find('DisorderMappingValidationStatus')
        vstat_id  = vstat.attrib.get('id') if vstat is not None else None
        vstat_nm  = vstat.findtext('Name') if vstat is not None else None

        # ICD ref links (may be empty)
        icd_url = ext.findtext('DisorderMappingICDRefUrl')
        icd_uri = ext.findtext('DisorderMappingICDRefUri')

        data.append({
            'DisorderID': disorder_id,
            'OrphaCode': orpha_code,
            'DisorderName': disorder_name,
            'ExpertLink': expert_link,
            'DisorderType': disorder_type_name,
            'DisorderGroup': disorder_group_name,
            'Flags': flags_joined,
            'Synonyms': synonyms_joined,
            'Definition': definition,
            'ExternalRefID': ext_id,
            'ExternalSource': ext_src,
            'ExternalReference': ext_ref,
            'MappingRelationID': mrel_id,
            'MappingRelationName': mrel_nm,
            'ICDRelationID': icdrel_id,
            'ICDRelationName': icdrel_nm,
            'ValidationStatusID': vstat_id,
            'ValidationStatusName': vstat_nm,
            'ICDRefUrl': icd_url,
            'ICDRefUri': icd_uri
        })


# Build DataFrame and save    
df_op_terms = pd.DataFrame(data)
print(df_op_terms.head())
df_op_terms.to_csv(save_path + "df_Orphadata_terminology.csv", index = False)
#=============================================================================#


#%%
#=============================================================================#
#=================== Extract RD Functional Consequences Data =================#
tree_op_func = ET.parse(path_op_func)
root_op_func = tree_op_func.getroot()

data = []

# Iterate through each DisorderDisabilityRelevance block
for ddr in root_op_func.findall('.//DisorderDisabilityRelevance'):
    relevance_id = ddr.attrib.get('id')

    # Disorder-level
    disorder = ddr.find('.//Disorder')
    disorder_id   = disorder.attrib.get('id') if disorder is not None else None
    orpha_code    = disorder.findtext('OrphaCode') if disorder is not None else None
    expert_link   = disorder.findtext('ExpertLink') if disorder is not None else None
    dname_el      = disorder.find('Name') if disorder is not None else None
    disorder_name = dname_el.text if dname_el is not None else None

    disorder_type_name  = disorder.findtext('.//DisorderType/Name') if disorder is not None else None
    disorder_group_name = disorder.findtext('.//DisorderGroup/Name') if disorder is not None else None

    # Each DisabilityDisorderAssociation under this disorder
    for assoc in ddr.findall('.//DisabilityDisorderAssociationList/DisabilityDisorderAssociation'):
        assoc_id = assoc.attrib.get('id')

        # Disability node
        disab = assoc.find('Disability')
        disability_id   = disab.attrib.get('id') if disab is not None else None
        disability_name = disab.findtext('Name') if disab is not None else None

        # Frequency
        freq = assoc.find('FrequenceDisability')
        freq_id   = freq.attrib.get('id') if freq is not None else None
        freq_name = freq.findtext('Name') if freq is not None else None

        # Temporality
        temp = assoc.find('TemporalityDisability')
        temp_id   = temp.attrib.get('id') if temp is not None else None
        temp_name = temp.findtext('Name') if temp is not None else None

        # Severity
        sev = assoc.find('SeverityDisability')
        sev_id   = sev.attrib.get('id') if sev is not None else None
        sev_name = sev.findtext('Name') if sev is not None else None

        # Other simple flags
        loss_of_ability = assoc.findtext('LossOfAbility')
        assoc_type      = assoc.findtext('Type')
        defined         = assoc.findtext('Defined')

        data.append({
            # Context
            'RelevanceID': relevance_id,
            'DisorderID': disorder_id,
            'OrphaCode': orpha_code,
            'ExpertLink': expert_link,
            'DisorderName': disorder_name,
            'DisorderType': disorder_type_name,
            'DisorderGroup': disorder_group_name,
            # Association
            'AssociationID': assoc_id,
            'DisabilityID': disability_id,
            'DisabilityName': disability_name,
            'FrequencyID': freq_id,
            'FrequencyName': freq_name,
            'TemporalityID': temp_id,
            'TemporalityName': temp_name,
            'SeverityID': sev_id,
            'SeverityName': sev_name,
            'LossOfAbility': loss_of_ability,
            'AssociationType': assoc_type,
            'Defined': defined
        })




# Build DataFrame and save    
df_op_func = pd.DataFrame(data)
print(df_op_func.head())
df_op_func.to_csv(save_path + "df_Orphadata_functional_consequences.csv", index = False)
#=============================================================================#

#%%
#=============================================================================#
#======================== Extract RD Phenotype Data===========================#
tree_op_pheno = ET.parse(path_op_pheno)
root_op_pheno = tree_op_pheno.getroot()

data = []

# Iterate through HPODisorderSetStatus elements
for status in root_op_pheno.findall('.//HPODisorderSetStatus'):
    disorder = status.find('Disorder')
    disorder_id = disorder.attrib.get('id')
    orpha_code = disorder.findtext('OrphaCode')
    expert_link = disorder.findtext('ExpertLink')
    disorder_name = disorder.find('Name').text if disorder.find('Name') is not None else None
    
    # DisorderType and DisorderGroup names
    disorder_type = disorder.find('.//DisorderType/Name')
    disorder_group = disorder.find('.//DisorderGroup/Name')
    disorder_type_name = disorder_type.text if disorder_type is not None else None
    disorder_group_name = disorder_group.text if disorder_group is not None else None
    
    # Iterate over each HPODisorderAssociation
    for assoc in disorder.findall('.//HPODisorderAssociation'):
        hpo = assoc.find('HPO')
        hpo_id = hpo.attrib.get('id') if hpo is not None else None
        hpo_code = hpo.findtext('HPOId') if hpo is not None else None
        hpo_term = hpo.findtext('HPOTerm') if hpo is not None else None
        
        diag_crit = assoc.find('DiagnosticCriteria') 
        diagnostic_criteria = diag_crit.findtext('Name') if diag_crit is not None else None
        
        freq = assoc.find('HPOFrequency')
        freq_id = freq.attrib.get('id') if freq is not None else None
        freq_name = freq.findtext('Name') if freq is not None else None

        data.append({
            'DisorderID': disorder_id,
            'OrphaCode': orpha_code,
            'ExpertLink': expert_link,
            'DisorderName': disorder_name,
            'DisorderType': disorder_type_name,
            'DisorderGroup': disorder_group_name,
            'HPO_ID': hpo_id,
            'HPO_Code': hpo_code,
            'HPO_Term': hpo_term,
            'DiagnosticCriteria': diagnostic_criteria,
            'FrequencyID': freq_id,
            'FrequencyName': freq_name
        })
    
# Build DataFrame and save    
df_op_pheno = pd.DataFrame(data)
print(df_op_pheno.head())
df_op_pheno.to_csv(save_path + "df_Orphadata_phenotypes.csv", index = False)
#=============================================================================#

#%%

#=============================================================================#
#======================== Extract RD Gene Data  ==============================#
tree_op_gene = ET.parse(path_op_gene)
root_op_gene = tree_op_gene.getroot()

data = []

# Iterate through each Disorder
for disorder in root_op_gene.findall('.//Disorder'):
    disorder_id = disorder.attrib.get('id')
    orpha_code  = disorder.findtext('OrphaCode')
    expert_link = disorder.findtext('ExpertLink')
    dname_el    = disorder.find('Name')
    disorder_name = dname_el.text if dname_el is not None else None

    # DisorderType and DisorderGroup names
    dtype_name = disorder.findtext('.//DisorderType/Name')
    dgroup_name = disorder.findtext('.//DisorderGroup/Name')

    # Iterate over each DisorderGeneAssociation
    for assoc in disorder.findall('.//DisorderGeneAssociationList/DisorderGeneAssociation'):
        # Association meta
        validation_source = assoc.findtext('SourceOfValidation')
        assoc_type_name   = assoc.findtext('DisorderGeneAssociationType/Name')
        assoc_status_name = assoc.findtext('DisorderGeneAssociationStatus/Name')

        # Gene block (optional)
        gene = assoc.find('Gene')
        gene_id     = gene.attrib.get('id') if gene is not None else None
        gene_name   = gene.findtext('Name') if gene is not None else None
        gene_symbol = gene.findtext('Symbol') if gene is not None else None

        # Synonyms (flattened as pipe-separated)
        synonyms = []
        if gene is not None:
            for syn in gene.findall('.//SynonymList/Synonym'):
                if syn.text:
                    synonyms.append(syn.text.strip())
        gene_synonyms = "|".join(synonyms) if synonyms else None

        # External references (Source:Reference; pipe-separated)
        ext_refs = []
        if gene is not None:
            for ext in gene.findall('.//ExternalReferenceList/ExternalReference'):
                src = ext.findtext('Source')
                ref = ext.findtext('Reference')
                if src or ref:
                    ext_refs.append(f"{src}:{ref}")
        gene_external_refs = "|".join(ext_refs) if ext_refs else None

        # Loci (GeneLocus values; pipe-separated)
        loci = []
        if gene is not None:
            for loc in gene.findall('.//LocusList/Locus'):
                gl = loc.findtext('GeneLocus')
                if gl:
                    loci.append(gl.strip())
        gene_locus = "|".join(loci) if loci else None

        data.append({
            'DisorderID': disorder_id,
            'OrphaCode': orpha_code,
            'ExpertLink': expert_link,
            'DisorderName': disorder_name,
            'DisorderType': dtype_name,
            'DisorderGroup': dgroup_name,
            'ValidationSource': validation_source,
            'AssociationType': assoc_type_name,
            'AssociationStatus': assoc_status_name,
            'GeneID': gene_id,
            'GeneSymbol': gene_symbol,
            'GeneName': gene_name,
            'GeneSynonyms': gene_synonyms,
            'GeneExternalRefs': gene_external_refs,
            'GeneLocus': gene_locus
        })



df_op_gene = pd.DataFrame(data)
print(df_op_gene.head())
df_op_gene.to_csv(save_path + "df_Orphadata_genes.csv", index = False)
#=============================================================================#


#%%

#=============================================================================#
#======================== Extract RD Linearization Data ======================#
tree_op_lin = ET.parse(path_op_lin)
root_op_lin = tree_op_lin.getroot()

data = []


# Iterate through each Disorder (source context)
for disorder in root_op_lin.findall('.//Disorder'):
    # Disorder context
    disorder_id   = disorder.attrib.get('id')
    orpha_code    = disorder.findtext('OrphaCode')
    expert_link   = disorder.findtext('ExpertLink')
    dname_el      = disorder.find('Name')
    disorder_name = dname_el.text if dname_el is not None else None

    # Each Disorder→Disorder association
    for assoc in disorder.findall('.//DisorderDisorderAssociationList/DisorderDisorderAssociation'):
        # TargetDisorder block
        target = assoc.find('TargetDisorder')
        target_id    = target.attrib.get('id') if target is not None else None
        target_code  = target.findtext('OrphaCode') if target is not None else None
        target_name  = target.findtext('Name') if target is not None else None

        # RootDisorder attributes (usually mirrors current disorder, includes cycle flag)
        rootd = assoc.find('RootDisorder')
        root_id   = rootd.attrib.get('id') if rootd is not None else None
        root_cycle = rootd.attrib.get('cycle') if rootd is not None else None  # "true"/"false" or None

        # Association type
        atype = assoc.find('DisorderDisorderAssociationType')
        atype_id = atype.attrib.get('id') if atype is not None else None
        atype_nm = atype.findtext('Name') if atype is not None else None

        data.append({
            # Source disorder context
            'DisorderID': disorder_id,
            'OrphaCode': orpha_code,
            'DisorderName': disorder_name,
            'ExpertLink': expert_link,
            # Association details
            'TargetDisorderID': target_id,
            'TargetOrphaCode': target_code,
            'TargetName': target_name,
            'RootDisorderID': root_id,
            'RootCycle': root_cycle,
            'AssociationTypeID': atype_id,
            'AssociationTypeName': atype_nm
        })



# Build DataFrame and save    
df_op_lin = pd.DataFrame(data)
print(df_op_lin.head())
df_op_lin.to_csv(save_path + "df_Orphadata_linearization.csv", index = False)
#=============================================================================#

#%%
#=============================================================================#
#======================== Extract RD Natural History Data=====================#
tree_op_nat_hist = ET.parse(path_op_nat_hist)
root_op_nat_hist = tree_op_nat_hist.getroot()

data = []


# Iterate through each Disorder
for disorder in root_op_nat_hist.findall('.//Disorder'):
    disorder_id   = disorder.attrib.get('id')
    orpha_code    = disorder.findtext('OrphaCode')
    expert_link   = disorder.findtext('ExpertLink')
    dname_el      = disorder.find('Name')
    disorder_name = dname_el.text if dname_el is not None else None

    # DisorderType and DisorderGroup names
    disorder_type_name  = disorder.findtext('.//DisorderType/Name')
    disorder_group_name = disorder.findtext('.//DisorderGroup/Name')

    # ------- AverageAgeOfOnset list (ids & names) -------
    onset_ids, onset_names = [], []
    for onset in disorder.findall('.//AverageAgeOfOnsetList/AverageAgeOfOnset'):
        oid = onset.attrib.get('id')
        onm = onset.findtext('Name')
        if oid: onset_ids.append(oid)
        if onm: onset_names.append(onm.strip())
    age_onset_ids   = "|".join(onset_ids)   if onset_ids   else None
    age_onset_names = "|".join(onset_names) if onset_names else None

    # ------- TypeOfInheritance list (ids & names) -------
    inh_ids, inh_names = [], []
    for inh in disorder.findall('.//TypeOfInheritanceList/TypeOfInheritance'):
        iid = inh.attrib.get('id')
        inm = inh.findtext('Name')
        if iid: inh_ids.append(iid)
        if inm: inh_names.append(inm.strip())
    inh_type_ids   = "|".join(inh_ids)   if inh_ids   else None
    inh_type_names = "|".join(inh_names) if inh_names else None

    data.append({
        'DisorderID': disorder_id,
        'OrphaCode': orpha_code,
        'ExpertLink': expert_link,
        'DisorderName': disorder_name,
        'DisorderType': disorder_type_name,
        'DisorderGroup': disorder_group_name,
        'AgeOfOnsetIDs': age_onset_ids,
        'AgeOfOnsetNames': age_onset_names,
        'InheritanceTypeIDs': inh_type_ids,
        'InheritanceTypeNames': inh_type_names
    })


# Build DataFrame and save    
df_op_nat_hist = pd.DataFrame(data)
print(df_op_nat_hist.head())
df_op_nat_hist.to_csv(save_path + "df_Orphadata_natural_histories.csv", index = False)
#=============================================================================#


#%%

#=============================================================================#
#======================== Extract RD Epidemiology Data =======================#
tree_op_epid = ET.parse(path_op_epid)
root_op_epid = tree_op_epid.getroot()

data = []

# Iterate through each Disorder
for disorder in root_op_epid.findall('.//Disorder'):
    # Disorder context
    disorder_id   = disorder.attrib.get('id')
    orpha_code    = disorder.findtext('OrphaCode')
    expert_link   = disorder.findtext('ExpertLink')
    dname_el      = disorder.find('Name')
    disorder_name = dname_el.text if dname_el is not None else None
    disorder_type_name  = disorder.findtext('.//DisorderType/Name')
    disorder_group_name = disorder.findtext('.//DisorderGroup/Name')

    # Each Prevalence entry under this disorder
    for prev in disorder.findall('.//PrevalenceList/Prevalence'):
        prev_id   = prev.attrib.get('id')
        source    = prev.findtext('Source')

        # PrevalenceType
        ptype     = prev.find('PrevalenceType')
        ptype_id  = ptype.attrib.get('id') if ptype is not None else None
        ptype_nm  = ptype.findtext('Name') if ptype is not None else None

        # Qualification
        qual      = prev.find('PrevalenceQualification')
        qual_id   = qual.attrib.get('id') if qual is not None else None
        qual_nm   = qual.findtext('Name') if qual is not None else None

        # Class (may be empty/self-closing)
        pclass    = prev.find('PrevalenceClass')
        pclass_id = pclass.attrib.get('id') if (pclass is not None and pclass.attrib) else None
        pclass_nm = pclass.findtext('Name') if pclass is not None else None

        # Mean value
        val_moy = prev.findtext('ValMoy')

        # Geography
        geo      = prev.find('PrevalenceGeographic')
        geo_id   = geo.attrib.get('id') if geo is not None else None
        geo_nm   = geo.findtext('Name') if geo is not None else None

        # Validation status
        valid    = prev.find('PrevalenceValidationStatus')
        valid_id = valid.attrib.get('id') if valid is not None else None
        valid_nm = valid.findtext('Name') if valid is not None else None

        data.append({
            # Disorder context
            'DisorderID': disorder_id,
            'OrphaCode': orpha_code,
            'ExpertLink': expert_link,
            'DisorderName': disorder_name,
            'DisorderType': disorder_type_name,
            'DisorderGroup': disorder_group_name,
            # Prevalence details
            'PrevalenceID': prev_id,
            'Source': source,
            'PrevalenceTypeID': ptype_id,
            'PrevalenceTypeName': ptype_nm,
            'QualificationID': qual_id,
            'QualificationName': qual_nm,
            'PrevalenceClassID': pclass_id,
            'PrevalenceClassName': pclass_nm,
            'ValMoy': val_moy,
            'GeographicID': geo_id,
            'GeographicName': geo_nm,
            'ValidationStatusID': valid_id,
            'ValidationStatusName': valid_nm
        })

# Build DataFrame and save    
df_op_epid = pd.DataFrame(data)
print(df_op_epid.head())
df_op_epid.to_csv(save_path + "df_Orphadata_epidemiology.csv", index = False)
#=============================================================================#







