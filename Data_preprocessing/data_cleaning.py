# Xiyu Rao
# 2025-10-15

def preprocess_dataset(dataset, dataset_type, cutoff_smp=0.4, cutoff_feat=0.6):
    # Key variant information columns to preserve (if present in dataset)
    key_info_columns = [
        'Chr', 'Start', 'End', 'Ref', 'Alt', 'Otherinfo', 'Phenotype',
        'platform', 'study', 'PMID', 'label', 'ClinVar_label', 'Explanation', 'ReVe'
    ]

    # Check which key information columns are actually available
    available_key_columns = [col for col in key_info_columns if col in dataset.columns]
    print(f"[INFO] Retained key information columns: {available_key_columns}")

    # Extract continuous functional features and EVS (evidence-based) features
    fun_feature = func_individual_function_dat(dataset, dataset, get_continuous_feature())
    evs_feature = func_individual_evs_dat(dataset, get_evs_features(), [False, 0, 0.02])

    # Output feature extraction summary
    print(f"\n[Feature Extraction Summary]")
    print(f"Continuous feature dimensions: {fun_feature.shape}")
    print(f"EVS feature dimensions: {evs_feature.shape}")
    print(f"Number of key information columns: {len(available_key_columns)}")
    print(f"Label dimensions: {dataset['label'].values.shape}")

    # Merge all features
    feats = pd.concat([fun_feature, evs_feature], axis=1)

    # Append retained key information columns
    for col in available_key_columns:
        if col in dataset.columns:
            feats[col] = dataset[col].values

    df = feats.copy()

    # ---------------- Label Distribution Analysis -----------------
    print("\n[Initial Label Distribution]")
    label_counts = df['label'].value_counts()
    print(label_counts)

    # Print overall dataset statistics before filtering
    print(f"\nDataset dimensions before filtering: {df.shape}")
    print(f"Total missing values: {df.isnull().sum().sum()}")

    # Feature column list (excluding key information columns)
    feature_columns = list(fun_feature.columns) + list(evs_feature.columns)
    print(f"Total number of feature columns: {len(feature_columns)}")

    # ---------------- Sample Filtering -----------------
    print(f"\n[Sample Filtering]")
    print(f"Number of samples before filtering: {df.shape[0]}")
    missing_per_row = df[feature_columns].isnull().mean(axis=1)

    # Examine missing values distribution by label
    print("\n[Missing Value Distribution Across Labels]")
    df_temp = df.copy()
    df_temp['missing_percent'] = missing_per_row
    print(df_temp.groupby('label')['missing_percent'].describe())

    # Execute sample filtering based on row-wise missing value threshold
    df = df[missing_per_row <= cutoff_smp]
    print(f"Number of removed samples: {len(missing_per_row[missing_per_row > cutoff_smp])}")
    print(f"Number of remaining samples: {df.shape[0]}")
    print(f"Total missing values after sample filtering: {df[feature_columns].isnull().sum().sum()}")

    # ---------------- Feature Filtering -----------------
    print(f"\n[Feature Filtering]")
    print(f"Number of features before filtering: {df.shape[1]}")

    missing_per_col = df[feature_columns].isnull().mean()
    removed_features = missing_per_col[missing_per_col > cutoff_feat].index.tolist()

    # Output removed features
    print(f"Removed features: {removed_features}")

    # Preserve all columns but remove only selected features
    columns_to_keep = [col for col in df.columns if col not in removed_features]
    df = df[columns_to_keep]

    # -------- Update Feature Column List After Filtering (Important) --------
    feature_columns = [col for col in feature_columns if col in df.columns]
    print(f"Remaining number of feature columns after filtering: {len(feature_columns)}")

    print(f"Number of removed features: {len(removed_features)}")
    print(f"Total number of columns retained: {df.shape[1]} (including {len(available_key_columns)} key info columns)")

    # ---------------- Remaining Feature Categorization -----------------
    continuous_features = [f for f in get_continuous_feature() if f in df.columns]
    evs_features = [f for f in get_evs_features() if f in df.columns]

    print("\n[Remaining Feature Categorization]")
    print(f"Continuous features: {len(continuous_features)}")
    print(f"EVS features: {len(evs_features)}")
    print(f"Key information columns: {len(available_key_columns)}")
    print(f"Label columns: 1")

    # Print overall dataset statistics after filtering
    print(f"\n[Final Dataset] Dimensions: {df.shape}")
    print(f"Remaining total missing values: {df[feature_columns].isnull().sum().sum()}")

    # ---------------- Label Distribution After Filtering -----------------
    print("\n[Label Distribution After Filtering]")
    label_counts_after = df['label'].value_counts()
    print(label_counts_after)

    # Compare label retention ratio
    label_ratio = (label_counts_after / label_counts).fillna(0)
    print("\n[Label Retention Ratio]")
    print(label_ratio)

    # Reset index and store original row index as a column
    df_reset = df.reset_index()
    df_reset = df_reset.rename(columns={'index': 'variant_index'})

    print(f"\n======= {dataset_type} Dataset Preprocessing Completed =======")
    print(f"Final number of variants: {len(df_reset)}")
    print(f"Columns included: {list(df_reset.columns)}")

    return df_reset


outdir = 'new_processed_data'

# Process coding dataset
preprocessed_coding = preprocess_dataset(
    coding_data,
    "coding",
    cutoff_smp=0.3,  # Adjusted sample missing value threshold
    cutoff_feat=0.7  # Adjusted feature missing value threshold
)

preprocessed_coding.to_csv(f"{outdir}/preprocessed_coding_with_variant_info.csv", index=False)


def impute_normalize_dataset(df, dataset_type, outdir):
    print(f"\n======= Starting Imputation & Normalization for {dataset_type} Dataset =======")

    # Record original label distribution
    original_label_dist = df['label'].value_counts().sort_index()
    print("Original label distribution:")
    for label, count in original_label_dist.items():
        print(f"  Label {label}: {count} samples")

    # ==================== Key Revision: Proper Separation of Numeric & Non-Numeric Columns ====================
    # 1. Separate label column
    labels = df['label'].copy()

    # 2. Identify non-numeric (string/object) columns
    non_numeric_columns = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Non-numeric columns: {non_numeric_columns}")

    # 3. Separate non-numeric data
    non_numeric_data = df[non_numeric_columns].copy()

    # 4. Separate numeric feature columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove label column from numeric columns if present
    if 'label' in numeric_columns:
        numeric_columns.remove('label')

    features = df[numeric_columns].copy()

    print(f"Number of numeric feature columns: {len(numeric_columns)}")
    print(f"Number of non-numeric columns: {len(non_numeric_columns)}")
    print(f"Label column has been successfully separated")
    # =====================================================================================================

    # Missing value imputation using KNN (only applied to numeric features)
    print(f"[Missing Value Imputation] Method: KNN")
    imputer = KNNImputer(n_neighbors=5, weights='uniform')
    features_filled = pd.DataFrame(
        imputer.fit_transform(features),
        columns=features.columns,
        index=features.index
    )

    # ==================== Key Revision: Reconstruct the Dataset Properly ====================
    # Reconstruct dataset: numeric features + non-numeric data + labels
    df_filled = pd.concat([features_filled, non_numeric_data, labels], axis=1)
    # =========================================================================================

    # Verify label integrity after imputation
    filled_label_dist = df_filled['label'].value_counts().sort_index()
    print("\nLabel distribution after imputation:")
    for label, count in filled_label_dist.items():
        print(f"  Label {label}: {count} samples")

    # Compare original vs. current label distributions
    print("\n[Label Integrity Check]")
    labels_unchanged = original_label_dist.equals(filled_label_dist)
    if labels_unchanged:
        print("✓ Label integrity preserved")
    else:
        print("✗ Label integrity check failed!")
        print("Original distribution:", dict(original_label_dist))
        print("Current distribution:", dict(filled_label_dist))

        # Identify discrepancies
        for label in set(original_label_dist.index).union(set(filled_label_dist.index)):
            orig_count = original_label_dist.get(label, 0)
            filled_count = filled_label_dist.get(label, 0)
            if orig_count != filled_count:
                print(
                    f"  Label {label}: Original {orig_count} → Now {filled_count} (Difference: {filled_count - orig_count})")

    # Normalize continuous features (excluding labels and non-numeric columns)
    continuous_features = [
        f for f in get_continuous_feature()
        if f in df_filled.columns and f in numeric_columns
    ]
    if continuous_features:
        print(f"\n[Feature Normalization] Number of continuous features: {len(continuous_features)}")
        scaler = MinMaxScaler()
        df_filled[continuous_features] = scaler.fit_transform(df_filled[continuous_features])

    # Save processed data
    output_path = os.path.join(outdir, f'processed_{dataset_type}_data.csv')
    df_filled.to_csv(output_path, index=False)
    print(f"\nProcessing completed for {dataset_type}. Saved to: {output_path}")
    print(f"Final dataset shape: {df_filled.shape}")

    # Final summary
    print("\n[Processing Summary]")
    print(f"Total samples: {len(df_filled)}")
    print(f"Numeric feature count: {len(numeric_columns)}")
    print(f"Non-numeric column count: {len(non_numeric_columns)}")
    print(f"Final label distribution: {dict(df_filled['label'].value_counts().sort_index())}")

    if labels_unchanged:
        print("✓ All processing completed successfully. Data integrity is maintained.")
    else:
        print("⚠ Processing completed, but label inconsistency detected. Manual verification recommended.")

    print(f"\n======= Dataset Processing Finished for {dataset_type} =======\n")
    return df_filled


# Check if KNNImputer is available
print(KNNImputer)

# Process coding dataset
final_coding = impute_normalize_dataset(
    preprocessed_coding,
    "coding",
    outdir
)
