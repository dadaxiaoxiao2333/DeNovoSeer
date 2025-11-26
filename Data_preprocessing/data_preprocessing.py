# Xiyu Rao
# 2025-10-15

import argparse, sys, re
import numpy  as np
import pandas as pd
from sklearn.impute import KNNImputer
import uuid


def preprocess_data(annovar_path, varseeker_path):
    annovar_dat = pd.read_csv(annovar_path, na_values=".")
    varseeker_dat = pd.read_csv(varseeker_path, dtype='str', na_values=".")

    index_generator = lambda df: df.apply(
        lambda x: f'chr{x["Chr"]}_start{x["Start"]}_end{x["End"]}_ref{x["Ref"]}_alt{x["Alt"]}',
        axis=1
    )

    annovar_dat['index'] = index_generator(annovar_dat)
    varseeker_dat['index'] = varseeker_dat.apply(
        lambda x: f'chr{x["Chr"]}_start{x["Start"]}_end{x["End"]}_ref{x["Ref"]}_alt{x["Alt"]}',
        axis=1
    )

    if not annovar_dat['index'].equals(varseeker_dat['index']):
        print("ERROR: Index mismatch between ANNOVAR and VarSeeker files!")
        sys.exit(1)

    return pd.merge(annovar_dat, varseeker_dat, on='index', suffixes=('_anno', '_var'))


def get_continuous_feature():
    return [
        # conservation
        "Polyphen2_HDIV_score", "Polyphen2_HVAR_score", "LINSIGHT", "GERP++_NR", "GERP++_RS", "phyloP100way_vertebrate",
        "phyloP470way_mammalian", "phyloP17way_primate", "phastCons100way_vertebrate", "phastCons470way_mammalian",
        "phastCons17way_primate", "SiPhy_29way_logOdds",

        # coding features
        "ReVe", "SIFT_score", "SIFT4G_score", "LRT_score", "LRT_Omega", "MutationTaster_score",
        "MutationAssessor_score",
        "FATHMM_score", "PROVEAN_score", "VEST4_score", "MetaSVM_score", "MetaLR_score", "MetaRNN_score", "M-CAP_score",
        "REVEL_score", "MutPred_score", "MutPred2_score", "MVP_score", "gMVP_score", "MPC_score", "PrimateAI_score",
        "DEOGEN2_score", "BayesDel_addAF_score", "BayesDel_noAF_score", "ClinPred_score", "LIST-S2_score",
        "VARITY_R_score", "VARITY_ER_score", "VARITY_R_LOO_score", "VARITY_ER_LOO_score", "ESM1b_score", "EVE_score",
        "AlphaMissense_score", "CADD_raw", "CADD_phred", "DANN_score", "fathmm-MKL_coding_score",
        "fathmm-XF_coding_score", "Eigen-raw_coding", "Eigen-phred_coding", "Eigen-PC-raw_coding",
        "Eigen-PC-phred_coding", "GenoCanyon_score", "integrated_fitCons_score", "integrated_confidence_value",
        "GM12878_fitCons_score", "GM12878_confidence_value", "H1-hESC_fitCons_score", "H1-hESC_confidence_value",
        "HUVEC_fitCons_score", "HUVEC_confidence_value", "bStatistic",

        # noncoding features
        "Caddnoncoding", "Dannnoncoding", "Fathmm_Mklnoncoding", "Funseq2", "Eigennoncoding", "Eigen_Pc",
        "Genocanyonnoncoding", "Fire", "Remm", "Linsight_Noncoding", "Fitconsnoncoding", "Fathmm_Xf", "Cscape", "Cdts",
        "Dvar", "Fitcons2", "Ncer", "Orion", "Pafa", "Regbase_Reg", "Regbase_Can", "Regbase_Pat", "Divan_Tss",
        "Divan_Region", "VARA_Score",

        # splicing effects
        "SpliceAI", "DS_AG", "DP_AG", "DS_AL", "DP_AL", "DS_DG", "DP_DG", "DS_DL", "DP_DL",

        # varseeker
        "Varseeker_score",

        # population-based
        "gnomad41_exome_AF", "gnomad41_exome_AF_raw", "gnomad41_exome_AF_XX", "gnomad41_exome_AF_XY",
        "gnomad41_exome_AF_grpmax", "gnomad41_exome_faf95", "gnomad41_exome_faf99", "gnomad41_exome_fafmax_faf95_max",
        "gnomad41_exome_fafmax_faf99_max", "gnomad41_exome_AF_afr", "gnomad41_exome_AF_amr", "gnomad41_exome_AF_asj",
        "gnomad41_exome_AF_eas", "gnomad41_exome_AF_fin", "gnomad41_exome_AF_mid", "gnomad41_exome_AF_nfe",
        "gnomad41_exome_AF_remaining", "gnomad41_exome_AF_sas", "gnomad41_genome_AF", "gnomad41_genome_AF_raw",
        "gnomad41_genome_AF_XX", "gnomad41_genome_AF_XY", "gnomad41_genome_AF_grpmax", "gnomad41_genome_faf95",
        "gnomad41_genome_faf99", "gnomad41_genome_fafmax_faf95_max", "gnomad41_genome_fafmax_faf99_max",
        "gnomad41_genome_AF_afr", "gnomad41_genome_AF_ami", "gnomad41_genome_AF_amr", "gnomad41_genome_AF_asj",
        "gnomad41_genome_AF_eas", "gnomad41_genome_AF_fin", "gnomad41_genome_AF_mid", "gnomad41_genome_AF_nfe",
        "gnomad41_genome_AF_remaining", "gnomad41_genome_AF_sas", "ExAC_ALL", "ExAC_AFR", "ExAC_AMR", "ExAC_EAS",
        "ExAC_FIN", "ExAC_NFE", "ExAC_OTH", "ExAC_SAS", "ExAC_nontcga_ALL", "ExAC_nontcga_AFR", "ExAC_nontcga_AMR",
        "ExAC_nontcga_EAS", "ExAC_nontcga_FIN", "ExAC_nontcga_NFE", "ExAC_nontcga_OTH", "ExAC_nontcga_SAS",
        "ExAC_nonpsych_ALL", "ExAC_nonpsych_AFR", "ExAC_nonpsych_AMR", "ExAC_nonpsych_EAS", "ExAC_nonpsych_FIN",
        "ExAC_nonpsych_NFE", "ExAC_nonpsych_OTH", "ExAC_nonpsych_SAS", "Kaviar_AF", "esp6500siv2_all"

    ]


def get_evs_features():
    evs_val = [
        "pvs1", "ps1", "ps2", "ps3", "ps4",
        "pm1", "pm2", "pm3", "pm4", "pm5", "pm6",
        "pp1", "pp2", "pp3", "pp4",
        "ba1", "bs1", "bs2", "bs3", "bs4",
        "bp1", "bp2", "bp3", "bp4", "bp5", "bp7"
    ]

    strength_scores = {
        "Unmet": 0,  # Neutral (no effect)
        "Supporting": 1,  # Pathogenicity:+1 benign:-1
        "Moderate": 2,  # Pathogenicity:+2 benign:-2
        "Strong": 4,  # Pathogenicity:+4 benign:-4
        "VeryStrong": 8,  # Pathogenicity:+8 benign:-8
        "StandAlone": 10  # Pathogenicity:+10 benign:-10
    }

    colnames = []
    for ev in evs_val:
        # Determine the type of evidence (pathogenic p or benign b)
        sign = 1 if ev.startswith("p") else -1

        scores = [0]
        for strength, value in strength_scores.items():
            if strength == "Unmet":
                continue
            scores.append(sign * value)

        unique_scores = sorted(list(set(scores)), reverse=(sign == 1))

        for score in unique_scores:
            colnames.append(f"{ev}_{{{score}}}")

    return colnames


def func_individual_function_dat(varseeker_dat, annovar_dat, fun_name):
    feature_list = []
    for col in fun_name:
        if col in varseeker_dat.columns:
            feature_list.append(varseeker_dat[[col]])
        else:
            feature_list.append(annovar_dat[[col]])

    return pd.concat(feature_list, axis=1).set_index(varseeker_dat['index']).astype(float)


def func_individual_evs_dat(varseeker_dat, evs_name, params):
    # Extract all possible prefixes of evidence categories (such as 'pvs1', 'ps1', ...）
    evs_types = {name.split('_')[0] for name in evs_name}

    rows = []

    for _, row in varseeker_dat.iterrows():
        feature_dict = {name: 0 for name in evs_name}
        for ev in evs_types:
            if ev in row and not pd.isna(row[ev]):
                value = int(row[ev])
                key = f"{ev}_{{{value}}}"
                if key in feature_dict:
                    feature_dict[key] = 1
        rows.append(feature_dict)

    evs_feature = pd.DataFrame(rows, columns=evs_name, index=varseeker_dat['index'])

    # Decide whether to add Gaussian noise based on gaussian_params
    if params[0]:
        noise = np.random.normal(params[1], params[2], evs_feature.shape)
        evs_feature += noise

    return evs_feature

