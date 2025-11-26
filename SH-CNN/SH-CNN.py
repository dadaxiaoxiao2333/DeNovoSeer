# Xiyu Rao
# 2025-10-15

# ========================== Hyperparameter Settings ==========================
epochs = 10
batchsize = 64
learning_rate = 0.0005
lambda_recon = 1.0  # Weight for reconstruction loss
lambda_contrast = 0.5  # Weight for contrastive loss

# Check whether GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def handle_imbalanced_data(features, labels, non_numeric_data=None, method='class_weight', random_state=None):
    """
    Handle class imbalance using multiple resampling strategies while preserving unknown labels (-1.0).

    Parameters:
    ----------
    features : numpy array
        Numeric feature matrix.
    labels : numpy array
        Multiclass labels (including -1.0 for unknown labels).
    non_numeric_data : pandas DataFrame (optional)
        Non-numeric columns to be preserved during resampling.
    method : str
        Methods to handle imbalance:
        'class_weight' (no resampling), 'oversample', 'undersample', 'smote'
    random_state : int or None
        Random seed for reproducibility.

    Returns:
    -------
    features_resampled, labels_resampled, non_numeric_resampled (if provided)
    """

    # Display original class distribution
    unique, counts = np.unique(labels, return_counts=True)
    class_dist = dict(zip(unique, counts))
    print(f"Original class distribution: {class_dist}")

    # No change to data when using class weights
    if method == 'class_weight':
        if non_numeric_data is not None:
            return features, labels, non_numeric_data
        else:
            return features, labels

    # --------------------------------- Separate Known vs. Unknown Labels ---------------------------------
    known_mask = labels != -1.0
    unknown_mask = ~known_mask

    X_known = features[known_mask]
    y_known = labels[known_mask]
    X_unknown = features[unknown_mask]
    y_unknown = labels[unknown_mask]

    if non_numeric_data is not None:
        non_numeric_known = non_numeric_data.iloc[known_mask].copy()
        non_numeric_unknown = non_numeric_data.iloc[unknown_mask].copy()

    # ----------------------------------------- Resampling Methods -----------------------------------------
    if method == 'oversample':
        sampler = RandomOverSampler(sampling_strategy='auto', random_state=random_state)
        X_res, y_res = sampler.fit_resample(X_known, y_known)
        if non_numeric_data is not None:
            non_numeric_res = non_numeric_known.iloc[sampler.sample_indices_].copy()

    elif method == 'undersample':
        sampler = RandomUnderSampler(sampling_strategy='auto', random_state=random_state)
        X_res, y_res = sampler.fit_resample(X_known, y_known)
        if non_numeric_data is not None:
            non_numeric_res = non_numeric_known.iloc[sampler.sample_indices_].copy()

    elif method == 'smote':
        sampler = SMOTE(sampling_strategy='auto', random_state=random_state, k_neighbors=5)
        X_res, y_res = sampler.fit_resample(X_known, y_known)

        if non_numeric_data is not None:
            # SMOTE generates new samples — we copy the nearest neighbor’s non-numeric info
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1)
            nn.fit(X_known)
            _, indices = nn.kneighbors(X_res[len(X_known):])
            extended_non_numeric = pd.concat([
                non_numeric_known,
                non_numeric_known.iloc[indices.flatten()].reset_index(drop=True)
            ])
            non_numeric_res = extended_non_numeric

    else:
        raise ValueError(f"Unknown method: {method}")

    # ------------------------- Merge Back Unknown Samples (-1.0) -------------------------
    X_final = np.concatenate([X_res, X_unknown])
    y_final = np.concatenate([y_res, y_unknown])

    if non_numeric_data is not None:
        non_numeric_final = pd.concat([non_numeric_res, non_numeric_unknown], axis=0)
        non_numeric_final = non_numeric_final.reset_index(drop=True)
    else:
        non_numeric_final = None

    # Display resampled distribution
    unique_res, counts_res = np.unique(y_final, return_counts=True)
    print(f"Resampled class distribution ({method}): {dict(zip(unique_res, counts_res))}")

    if non_numeric_data is not None:
        return X_final, y_final, non_numeric_final
    else:
        return X_final, y_final


# ========================== Apply Imbalanced Data Handling ==========================
train_feats, train_labels, train_non_numeric = handle_imbalanced_data(
    train_feats, train_labels, train_non_numeric, method='oversample', random_state=None
)


# ========================== Custom Dataset Class ==========================
class MutDataset(Dataset):
    """
    Custom PyTorch Dataset:
    - Supports both numeric features and non-numeric metadata
    - Unknown labels (-1) preserved
    """

    def __init__(self, data, labels, non_numeric_data=None, feature_names=None):
        labels = np.nan_to_num(labels, -1)
        self.labels = labels
        self.feats = data
        self.feature_names = feature_names
        self.non_numeric_data = non_numeric_data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor([self.feats[idx]], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    def get_non_numeric_info(self, idx):
        """Return non-numeric metadata for a given index"""
        if self.non_numeric_data is not None:
            return self.non_numeric_data.iloc[idx]
        else:
            return None


# ========================== Build Train / Val / Test Datasets ==========================
train_dataset = MutDataset(train_feats, train_labels, train_non_numeric, feature_names=numeric_feature_names)
test_dataset = MutDataset(test_feats, test_labels, test_non_numeric, feature_names=numeric_feature_names)
val_dataset = MutDataset(val_feats, val_labels, val_non_numeric, feature_names=numeric_feature_names)

train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False)


# ========================== Semi-Supervised Hybrid CNN-IDCNN Model ==========================
class SemiSupervised_Hybrid_CNN_IDCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, feature_dim=203):
        super(SemiSupervised_Hybrid_CNN_IDCNN, self).__init__()

        # Shared feature extractor (CNN + IDCNN)
        self.feature_extractor = nn.Sequential(
            # CNN — Local Feature Extraction
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # IDCNN — Capture Global Context
            nn.Conv1d(128, 256, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Supervised Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        # Reconstruction Head (for unlabeled data)
        self.reconstructor = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_channels * feature_dim)
        )

        # Projection Head for Contrastive Learning
        self.projection_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward(self, x, mode='supervised'):
        features = self.feature_extractor(x)

        if mode == 'supervised':
            return self.classifier(features)
        elif mode == 'reconstruction':
            return self.reconstructor(features)
        elif mode == 'contrastive':
            return self.projection_head(features)
        elif mode == 'features':
            return features
        else:
            raise ValueError("Invalid mode. Choose from 'supervised', 'reconstruction', 'contrastive', or 'features'")


# ========================== Loss Function for Imbalance — Focal Loss ==========================
class FocalLoss(nn.Module):
    """
    Focal Loss to handle severe class imbalance.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


# ========================== Data Augmentation for Contrastive Learning ==========================
def augment_data(x):
    """
    Lightweight data augmentation:
    - Add random noise
    - Random scaling
    """
    noise = torch.randn_like(x) * 0.1
    augmented = x + noise
    scale = torch.FloatTensor(1).uniform_(0.8, 1.2).to(device)
    augmented = augmented * scale
    return augmented.clamp(0, 1)
