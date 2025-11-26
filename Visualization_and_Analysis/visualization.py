# Xiyu Rao
# 2025-10-15

def plot_comparison_curves_with_missing(test_csv, tools_config, title_prefix, save_dir="performance_plots"):
    """ Plot ROC and PR curves (handle missing values directly) """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Read the test dataset
    df = pd.read_csv(test_csv)

    # Keep only samples with known labels (0 and 1)
    df = df[df['label'] != -1].copy()
    y_true = df['label'].values

    print(f"Valid samples (known labels): {len(df)}")

    # 1. ROC curve
    plt.figure(figsize=(12, 10))  # Increase figure size for larger font
    colors = plt.cm.tab20(np.linspace(0, 1, len(tools_config)))

    handles, labels = [], []  # Store legend handles and labels

    # Plot other tools first (skip SH-CNN)
    for i, (tool_name, col_info) in enumerate(tools_config.items()):
        if tool_name == 'SH-CNN':
            continue  # Draw SH-CNN last

        col_name, invert = col_info
        valid_mask = df[col_name].notna()
        y_true_tool = y_true[valid_mask]
        scores = df.loc[valid_mask, col_name].values

        if len(y_true_tool) == 0:
            print(f"⚠️ Warning: {tool_name} has no valid samples, skipped")
            continue

        unique_labels = np.unique(y_true_tool)
        if len(unique_labels) < 2:
            print(f"⚠️ Warning: {tool_name} only has one class ({unique_labels[0]}), ROC cannot be computed")
            continue

        if invert:
            scores = 1 - scores

        fpr, tpr, _ = roc_curve(y_true_tool, scores)
        roc_auc = auc(fpr, tpr)
        label = f'{tool_name} (AUC={roc_auc:.3f}, n={len(y_true_tool)})'

        fpr_smooth = np.linspace(0, 1, 200)
        tpr_smooth = np.interp(fpr_smooth, fpr, tpr)
        tpr_smooth[0] = 0.0

        line, = plt.plot(fpr_smooth, tpr_smooth, color=colors[i],
                         lw=2.0, alpha=0.8)  # Increase line width
        handles.append(line)
        labels.append(label)

    # Plot SH-CNN last (on top layer)
    if 'SH-CNN' in tools_config:
        tool_name, col_info = 'SH-CNN', tools_config['SH-CNN']
        col_name, invert = col_info
        valid_mask = df[col_name].notna()
        y_true_tool = y_true[valid_mask]
        scores = df.loc[valid_mask, col_name].values

        if len(y_true_tool) > 0:
            unique_labels = np.unique(y_true_tool)
            if len(unique_labels) >= 2:
                if invert:
                    scores = 1 - scores

                fpr, tpr, _ = roc_curve(y_true_tool, scores)
                roc_auc = auc(fpr, tpr)
                label = f'SH-CNN (AUC={roc_auc:.3f}, n={len(y_true_tool)})'

                fpr_smooth = np.linspace(0, 1, 200)
                tpr_smooth = np.interp(fpr_smooth, fpr, tpr)
                tpr_smooth[0] = 0.0

                line, = plt.plot(fpr_smooth, tpr_smooth,
                                 color=colors[list(tools_config.keys()).index('SH-CNN')],
                                 lw=2.8, alpha=1.0)  # Increase line width
                handles.insert(0, line)  # Insert at first position
                labels.insert(0, label)

    plt.plot([0, 1], [0, 1], 'k--', lw=2.0, alpha=0.7)  # Increase line width
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title(f'{title_prefix} - ROC Curves', fontsize=20, pad=20)
    plt.legend(handles, labels, loc="lower right", fontsize=14)
    plt.tight_layout()

    # Save ROC figure
    roc_path = os.path.join(save_dir, f'{title_prefix}_ROC.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved: {roc_path}")

    # 2. PR curve
    plt.figure(figsize=(12, 10))  # Increase figure size
    handles, labels = [], []  # Store legend handles and labels

    # Plot other tools first (skip SH-CNN)
    for i, (tool_name, col_info) in enumerate(tools_config.items()):
        if tool_name == 'SH-CNN':
            continue  # Draw SH-CNN last

        col_name, invert = col_info
        valid_mask = df[col_name].notna()
        y_true_tool = y_true[valid_mask]
        scores = df.loc[valid_mask, col_name].values

        if len(y_true_tool) == 0:
            continue

        unique_labels = np.unique(y_true_tool)
        if len(unique_labels) < 2:
            print(f"⚠️ Warning: {tool_name} only has one class ({unique_labels[0]}), PR cannot be computed")
            continue

        if invert:
            scores = 1 - scores

        precision, recall, _ = precision_recall_curve(y_true_tool, scores)
        ap = average_precision_score(y_true_tool, scores)
        label = f'{tool_name} (AP={ap:.3f}, n={len(y_true_tool)})'

        recall_smooth = np.linspace(0, 1, 200)
        precision_smooth = np.interp(recall_smooth, recall[::-1], precision[::-1])

        line, = plt.plot(recall_smooth, precision_smooth, color=colors[i],
                         lw=2.0, alpha=0.8)
        handles.append(line)
        labels.append(label)

    # Plot SH-CNN last (on top layer)
    if 'SH-CNN' in tools_config:
        tool_name, col_info = 'SH-CNN', tools_config['SH-CNN']
        col_name, invert = col_info
        valid_mask = df[col_name].notna()
        y_true_tool = y_true[valid_mask]
        scores = df.loc[valid_mask, col_name].values

        if len(y_true_tool) > 0:
            unique_labels = np.unique(y_true_tool)
            if len(unique_labels) >= 2:
                if invert:
                    scores = 1 - scores

                precision, recall, _ = precision_recall_curve(y_true_tool, scores)
                ap = average_precision_score(y_true_tool, scores)
                label = f'SH-CNN (AP={ap:.3f}, n={len(y_true_tool)})'

                recall_smooth = np.linspace(0, 1, 200)
                precision_smooth = np.interp(recall_smooth, recall[::-1], precision[::-1])

                line, = plt.plot(recall_smooth, precision_smooth,
                                 color=colors[list(tools_config.keys()).index('SH-CNN')],
                                 lw=2.8, alpha=1.0)
                handles.insert(0, line)  # Insert at first position
                labels.insert(0, label)

    plt.xlabel('Recall', fontsize=18)
    plt.ylabel('Precision', fontsize=18)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'{title_prefix} - Precision-Recall Curves', fontsize=20, pad=20)
    plt.legend(handles, labels, loc="lower right", fontsize=14)
    plt.tight_layout()

    # Save PR figure
    pr_path = os.path.join(save_dir, f'{title_prefix}_PR.png')
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"PR curve saved: {pr_path}")

    # Generate missing value report
    missing_report = generate_missing_value_report(df, tools_config, save_dir, title_prefix)
    print(missing_report)


def generate_missing_value_report(df, tools_config, save_dir, title_prefix):
    """ Generate missing value analysis report """
    report = "\n=== Missing Value Report ===\n"
    report += f"Total samples: {len(df)}\n"
    report += f"Samples with known labels: {len(df[df['label'] != -1])}\n\n"

    report += "Missing value per tool:\n"
    report += "{:<20} {:<10} {:<10} {:<10}\n".format("Tool", "Missing", "Rate", "Valid")

    for tool_name, col_info in tools_config.items():
        col_name, _ = col_info
        if col_name not in df.columns:
            report += f"{tool_name}: Column '{col_name}' does not exist\n"
            continue

        missing_count = df[col_name].isna().sum()
        missing_rate = missing_count / len(df)
        valid_count = len(df) - missing_count

        report += "{:<20} {:<10} {:<10.2%} {:<10}\n".format(
            tool_name, missing_count, missing_rate, valid_count
        )

    # Save report
    report_path = os.path.join(save_dir, f'{title_prefix}_missing_value_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Missing value report saved: {report_path}")
    return report
