# DeNovoSeer

**DeNovoSeer** is a semi‑supervised deep learning framework for predicting the pathogenicity of coding‑region *de novo* mutations (DNMs).  
It integrates multi‑source functional annotations with ACMG/AMP clinical evidence and provides SHAP‑based interpretability to support molecular diagnosis and genetic research.

This repository contains the official implementation of the paper:

> Qiu R, Rao X, Jiang H, Li J, Zhao G. **DeNovoSeer: A Deep Learning Framework for Pathogenicity Prediction of De Novo Mutations.** (2026)

---



## Project Structure

DeNovoSeer/
├── best_model/ # Pre-trained model checkpoints (seeds 42–51)
│ ├── best_model_seed42.pth
│ └── ...
├── data/
│ ├── Gene4Denovo/ # Place raw Gene4Denovo input files here
│ └── SPARK/ # Place raw SPARK input files here (optional)
├── Data_preprocessing/
│ └── Data_preprocessing.ipynb # Data cleaning, feature extraction, scaling
├── DeNovoSeer/
│ └── DeNovoSeer.ipynb # Model training and evaluation (10 splits)
├── shap/
│ └── shap_analysis_denovoseer.ipynb # SHAP interpretability analysis
├── Feature categories used in DeNovoSeer.txt # Description of feature groups
├── requirements.txt # Python dependencies
└── README.md



---

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies.

Install dependencies with:

```bash
pip install -r requirements.txt
```



## Data Preparation

DeNovoSeer requires two types of input CSV files for the **Gene4Denovo** dataset (the SPARK cohort can be preprocessed similarly if available).

### 1. ANNOVAR‑annotated file (`annovar_coding_Gene4Denovo.csv`)

Generated based on [ANNOVAR](https://annovar.openbioinformatics.org/) annotation and further enriched with multiple variant effect prediction tools.

In addition to standard ANNOVAR annotations, this file incorporates external functional and pathogenicity scores, including (but not limited to) **VARA, MutPred2, ReVe, and other in silico prediction tools**, providing a comprehensive variant-level feature profile.

It must contain at least the following columns:

- `Chr`, `Start`, `End`, `Ref`, `Alt`
- Functional scores listed in `get_continuous_feature()` (see preprocessing notebook)
- Any other phenotypic labels used later (e.g., `Otherinfo`)

### 2. Clinical evidence file (`Clinical_coding_Gene4Denovo.csv`)

Contains ACMG/AMP evidence strengths (PS2/PM6 etc.) computed by [GeneBe](https://genebe.net/) or a similar tool, and de novo phenotype data.
It must contain the same genomic coordinate columns (`Chr`, `Start`, `End`, `Ref`, `Alt`) plus evidence columns like `pvs1`, `ps2`, `ps3`, ... and a clinical score column (`Clinical_evidence_score`).

**Getting the data:**

- **Gene4Denovo** can be downloaded from http://www.genemed.tech/gene4denovo/download
- **SPARK** data is available to approved researchers via [SFARI Base](https://base.sfari.org/)

Place your raw CSV files inside the `data/Gene4Denovo/` folder (you may need to rename them to match the notebook paths or adjust the paths as indicated below).





## Usage – Step by Step

All main steps are provided as Jupyter notebooks. You can run them cell‑by‑cell in Jupyter Lab, Notebook, or VS Code.

### Step 1: Data Preprocessing

Open `Data_preprocessing/Data_preprocessing.ipynb`.

- **Update file paths:**
  In the second code cell, set the correct file names:

  python

  ```
  coding_annovar_path = '../data/Gene4Denovo/annovar_coding_Gene4Denovo.csv'
  coding_Clinical_path = '../data/Gene4Denovo/Clinical_coding_Gene4Denovo.csv'
  ```

  

- Run all cells. The notebook will:

  - Merge and validate the two inputs.
  - Extract continuous functional features and one‑hot encoded ACMG evidence features.
  - Filter variants with too many missing values (default: max 40% missing per variant, 70% per feature).
  - Apply KNN imputation (`n_neighbors=5`).
  - Min‑Max scale continuous features.
  - Save processed data to `processed_data/processed_coding.csv`.
  - Standard‑scale the data and export the final file: `processed_data/processed_coding_data_scaled.csv`.

The final output (`processed_coding_data_scaled.csv`) will be used by the training and SHAP notebooks.



### Step 2: Model Training & Evaluation

Open `DeNovoSeer/DeNovoSeer.ipynb`.

- **Set the path to the scaled data:**
  The notebook expects the file `processed_coding_data_scaled.csv` in the same directory. You can modify the `DATA_PATH` variable if needed.

  python

  ```
  DATA_PATH = "processed_coding_data_scaled.csv"
  ```

  

- Run all cells. The script will:

  - Perform **10 independent train/validation/test splits** using seeds 42–51, with stratified sampling for labeled variants.
  - For each split:
    - Randomly oversample the labeled (benign/pathogenic) subset to mitigate class imbalance.
    - Train the `SemiSupervisedHybridCNNIDCNN` model with focal loss, reconstruction loss, and contrastive loss.
    - Use early stopping based on validation AUC.
    - Evaluate on the test set (accuracy, precision, recall, F1, AUC, AP).
  - Save per‑split test predictions in `per_split_predictions/`.
  - Save test‑set indices (for reproducibility) in `test_set_index/`.
  - Save a summary of all 10 splits in `results_10_splits_noise_masking.csv`.
  - Build a **supplementary per‑variant prediction table** (`supplementary_variant_predictions.csv`) that aggregates predictions across splits.

**Model checkpoints:**
The pre‑trained models in `best_model/` were produced by exactly this script. You can also train from scratch and your results will be saved in the working directory (you may then move them to replace the existing ones).

------



### Step 3: SHAP Interpretability Analysis

Open `shap/shap_analysis_denovoseer.ipynb`.

- **Update the three key paths** at the top of the `main()` function:

  python

  ```
  data_path = "processed_coding_data_scaled.csv"   # same as training data
  model_path = "best_model_seed42.pth"             # choose any seed
  test_index_path = "test_set_index/coding_test_indices_seed42.npy"  # matching seed
  ```

  

- Run all cells. The notebook will:

  - Reconstruct the train/test split and validate the test index.
  - Load the chosen model.
  - Generate predictions for labeled test variants.
  - Compute **SHAP values** using `DeepExplainer` (with `GradientExplainer` fallback).
  - Produce **global** outputs: summary plot, bar plot, and a CSV ranking features by mean absolute SHAP.
  - Select representative cases (high‑confidence pathogenic, benign, and borderline) and produce **local** explanations:
    - Waterfall plots (PNG)
    - Force plots (HTML)
    - Local SHAP tables (CSV)
  - All results are saved in the `output_dir` folder (default: `shap_outputs_seed42`).

You can repeat this step with different seeds (e.g., 42–51) to assess stability of the explanations.



## Output Files

### Training & Evaluation

- `results_10_splits_noise_masking.csv` – performance metrics for all 10 splits.
- `per_split_predictions/test_predictions_seed{seed}.csv` – per‑variant predictions for each split.
- `supplementary_variant_predictions.csv` – aggregated predictions (mean, std) across splits.
- `test_set_index/coding_test_indices_seed{seed}.npy` – numpy arrays of test indices.



### SHAP Analysis

- `shap_outputs_seed{seed}/global_shap_importance.csv` – global feature importance (mean |SHAP|).
- `shap_outputs_seed{seed}/global_shap_summary_top20.png` – dot summary plot.
- `shap_outputs_seed{seed}/global_shap_bar_top20.png` – bar summary plot.
- `shap_outputs_seed{seed}/case_X_idx_Y_waterfall.png` – waterfall plot for a selected variant.
- `shap_outputs_seed{seed}/case_X_idx_Y_force.html` – interactive force plot.
- `shap_outputs_seed{seed}/selected_case_summary.csv` – overview of selected cases.

------



## Feature Categories

The file `Feature categories used in DeNovoSeer.txt` lists all features grouped by type.

------



## Reproducibility

- All random seeds are fixed per run (set by `set_seed()`).
- Training uses stratified splitting and consistent data loaders.
- The split indices are saved, allowing exact replication of each test set.
- Pre‑computed model checkpoints are provided for each seed (42–51).