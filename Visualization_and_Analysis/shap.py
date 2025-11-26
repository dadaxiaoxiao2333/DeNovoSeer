# Xiyu Rao
# 2025-10-15

def compute_shap_values(model, sample_data, device):
    """Compute SHAP values using clustered background samples"""
    try:
        # Wrapper to ensure correct input format for SHAP
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super(ModelWrapper, self).__init__()
                self.model = model

            def forward(self, x):
                # Make sure the input shape is valid
                if len(x.shape) == 2:
                    x = x.reshape(x.shape[0], 1, -1)
                return self.model(x, mode='supervised')

        wrapped_model = ModelWrapper(model).to(device)
        wrapped_model.eval()

        # Flatten data to run KMeans clustering
        flattened_data = sample_data.reshape(sample_data.shape[0], -1)

        # Use up to 300 background samples
        background_size = min(300, len(flattened_data))
        print(f"Computing SHAP with {background_size} background samples (KMeans clusters)")

        # KMeans clustering to generate representative background samples
        kmeans = KMeans(n_clusters=background_size, random_state=42, n_init=10)
        kmeans.fit(flattened_data)
        background = kmeans.cluster_centers_.reshape(background_size, 1, -1)

        # Prepare test samples (max 200 samples)
        test_size = min(200, len(sample_data) - background_size)
        test_data = sample_data[background_size:background_size + test_size]

        # Convert to PyTorch tensors
        background_tensor = torch.tensor(background, dtype=torch.float32).to(device)
        test_tensor = torch.tensor(test_data, dtype=torch.float32).to(device)

        # Compute SHAP values
        explainer = shap.DeepExplainer(
            wrapped_model,
            background_tensor
        )

        print("Computing SHAP values (averaging 2 runs to reduce randomness)...")
        shap_run1 = explainer.shap_values(test_tensor)
        shap_run2 = explainer.shap_values(test_tensor)

        # Average two runs to stabilize SHAP estimates
        if isinstance(shap_run1, list) and isinstance(shap_run2, list):
            avg_shap = []
            for i in range(len(shap_run1)):
                avg_shap.append((np.array(shap_run1[i]) + np.array(shap_run2[i])) / 2)
            return avg_shap
        else:
            return (np.array(shap_run1) + np.array(shap_run2)) / 2

    except Exception as e:
        print(f"Error computing SHAP values: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def plot_shap_summary(shap_values, sample_data, mode, feature_names, top_n=20):
    """Generate SHAP summary plots (global interpretability visualization)"""
    try:
        # Ensure the sample size matches between SHAP output and original input
        if shap_values.shape[0] != sample_data.shape[0]:
            min_samples = min(shap_values.shape[0], sample_data.shape[0])
            shap_values = shap_values[:min_samples]
            sample_data = sample_data[:min_samples]
            print(f"Adjusted sample size for SHAP summary to: {min_samples}")

        # Ensure feature dimension matches
        if shap_values.shape[1] != sample_data.shape[1]:
            min_features = min(shap_values.shape[1], sample_data.shape[1])
            shap_values = shap_values[:, :min_features]
            sample_data = sample_data[:, :min_features]
            print(f"Adjusted feature size for SHAP summary to: {min_features}")

        # Truncate feature names to match dimension
        feature_names = feature_names[:min_features]

        print(f"SHAP summary plot: samples={shap_values.shape[0]}, features={shap_values.shape[1]}")

        # 1. Bar plot of global feature importance
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            sample_data,
            plot_type="bar",
            feature_names=feature_names,
            max_display=top_n,
            show=False,
            plot_size=None,  # Disable SHAP automatic resizing
            color='darkblue'
        )
        plt.title(f'SHAP Feature Importance ({mode.capitalize()})')
        plt.tight_layout()
        plt.savefig(f'{mode}_shap_importance.png', bbox_inches='tight')
        plt.show()

        # 2. Scatter plot of feature contribution distribution
        plt.figure(figsize=(14, 10))
        shap.summary_plot(
            shap_values,
            sample_data,
            feature_names=feature_names,
            max_display=top_n,
            show=False,
            plot_size=None  # Disable default size control
        )
        plt.title(f'SHAP Value Distribution ({mode.capitalize()})')
        plt.tight_layout()
        plt.savefig(f'{mode}_shap_summary.png', bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"Error plotting SHAP summary: {str(e)}")
        import traceback
        traceback.print_exc()
