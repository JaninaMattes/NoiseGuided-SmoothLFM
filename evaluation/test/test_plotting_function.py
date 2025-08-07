import json
import random
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def generate_structured_dummy_probe_data(
    save_path="test_outputs/dummy_model_v0_results.json",
):
    model_configs_v0 = [
        {"name": "Beta02x10x_1e4b", "beta": 1e-4, "source_ts": 0.20, "target_ts": 1.00},
        {"name": "Beta02x10x_01b", "beta": 0.1, "source_ts": 0.20, "target_ts": 1.00},
        {"name": "Beta02x10x_05b", "beta": 0.5, "source_ts": 0.20, "target_ts": 1.00},
        {"name": "Beta02x10x_1b", "beta": 1.0, "source_ts": 0.20, "target_ts": 1.00},
        {"name": "Beta02x10x_2b", "beta": 2.0, "source_ts": 0.20, "target_ts": 1.00},
        {"name": "Beta02x10x_3b", "beta": 3.0, "source_ts": 0.20, "target_ts": 1.00},
        {"name": "Beta02x10x_5b", "beta": 5.0, "source_ts": 0.20, "target_ts": 1.00},
    ]

    probe_types = ["Linear", "Two-Layer"]
    pca_values = [2, 3, 5, 9, 15, 20]

    dummy_results = []

    for cfg in model_configs_v0:
        model_data = {
            "Model": cfg["name"],
            "Beta": cfg["beta"],
            "SourceTimestep": cfg["source_ts"],
            "TargetTimestep": cfg["target_ts"],
            "Results": [],
        }

        for probe in probe_types:
            for pca in pca_values:
                val_accuracies = [
                    round(random.uniform(0.05, 0.35), 3)
                    for _ in range(random.randint(6, 12))
                ]
                model_data["Results"].append(
                    {"ProbeType": probe, "PCA": pca, "ValAccuracies": val_accuracies}
                )

        dummy_results.append(model_data)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(dummy_results, f, indent=2)
    print(f"[INFO] Dummy probe result data saved to: {save_path}")
    return save_path


def generate_structured_dummy_probe_data(
    save_path="test_outputs/dummy_model_v0_results.json",
):
    model_configs_v0 = [
        {"name": "Beta02x10x_1e4b", "beta": 1e-4, "source_ts": 0.20, "target_ts": 1.00},
        {"name": "Beta02x10x_01b", "beta": 0.1, "source_ts": 0.20, "target_ts": 1.00},
        {"name": "Beta02x10x_05b", "beta": 0.5, "source_ts": 0.20, "target_ts": 1.00},
        {"name": "Beta02x10x_1b", "beta": 1.0, "source_ts": 0.20, "target_ts": 1.00},
        {"name": "Beta02x10x_2b", "beta": 2.0, "source_ts": 0.20, "target_ts": 1.00},
        {"name": "Beta02x10x_3b", "beta": 3.0, "source_ts": 0.20, "target_ts": 1.00},
        {"name": "Beta02x10x_5b", "beta": 5.0, "source_ts": 0.20, "target_ts": 1.00},
    ]

    probe_types = ["Linear", "Two-Layer"]
    pca_values = [2, 3, 5, 9, 15, 20]

    dummy_results = []

    for cfg in model_configs_v0:
        model_data = {
            "Model": cfg["name"],
            "Beta": cfg["beta"],
            "SourceTimestep": cfg["source_ts"],
            "TargetTimestep": cfg["target_ts"],
            "Results": [],
        }

        for probe in probe_types:
            for pca in pca_values:
                if probe == "Linear":
                    # Low accuracy range
                    val_accuracies = [
                        round(random.uniform(0.01, 0.05), 3)
                        for _ in range(random.randint(6, 12))
                    ]
                else:
                    # High accuracy range
                    val_accuracies = [
                        round(random.uniform(0.6, 0.95), 3)
                        for _ in range(random.randint(6, 12))
                    ]

                model_data["Results"].append(
                    {"ProbeType": probe, "PCA": pca, "ValAccuracies": val_accuracies}
                )

        dummy_results.append(model_data)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(dummy_results, f, indent=2)
    print(
        f"[INFO] Dummy probe result data (with extreme scale difference) saved to: {save_path}"
    )
    return save_path


def generate_dummy_kmeans_results(
    pca_values=[2, 3, 5, 7, 10, 20],
    k_values=[2, 3, 5, 10, 15, 20, 50],
    model_name="dummy_model",
):
    dummy_results = []

    for pca in pca_values:
        for k in k_values:
            dummy_results.append(
                {
                    "Model": model_name,
                    "Beta": 1.0,
                    "PCA": pca,
                    "K": k,
                    "Silhouette": round(random.uniform(0.2, 0.75), 3),
                    "ARI": round(random.uniform(0.1, 0.8), 3),
                    "NMI": round(random.uniform(0.15, 0.9), 3),
                }
            )

    return dummy_results


def generate_dummy_pca_kmeans_data(
    pca_values=[2, 3, 5], k_values=[2, 3, 4], n_samples=500, n_features=20
):
    np.random.seed(42)
    X, y_true = make_blobs(
        n_samples=n_samples, centers=5, n_features=n_features, random_state=42
    )

    all_rows = []
    for pca_dim in pca_values:
        pca = PCA(n_components=pca_dim)
        X_pca = pca.fit_transform(X)

        for k in k_values:
            kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
            labels = kmeans.fit_predict(X_pca)

            # For scatter plot, we only use 2D (first 2 components) for plotting
            if X_pca.shape[1] < 2:
                # pad with zeros
                X_plot = np.hstack(
                    [X_pca, np.zeros((X_pca.shape[0], 2 - X_pca.shape[1]))]
                )
            else:
                X_plot = X_pca[:, :2]

            for i in range(len(X_plot)):
                all_rows.append(
                    {
                        "PCA": pca_dim,
                        "K": k,
                        "x": X_plot[i, 0],
                        "y": X_plot[i, 1],
                        "Cluster": labels[i],
                    }
                )

    return pd.DataFrame(all_rows)


def plot_pca_kmeans_scatter_grid(df, project_name="test_outputs"):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Set global style to match example
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": False,
            "axes.facecolor": "#e8ecf0",
            "axes.edgecolor": "#cccccc",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "lines.linewidth": 1.2,
        }
    )

    # Convert types for plotting
    df["PCA"] = df["PCA"].astype(str)
    df["K"] = df["K"].astype(str)

    g = sns.FacetGrid(df, row="PCA", col="K", margin_titles=True, height=2.8, aspect=1)
    g.map_dataframe(
        sns.scatterplot,
        x="x",
        y="y",
        hue="Cluster",
        palette="tab10",
        s=20,
        linewidth=0,
        alpha=0.8,
        legend=False,  # <- disable legend in individual plots
    )

    g.set_axis_labels("PC 1", "PC 2")
    g.set_titles(row_template="PCA = {row_name}", col_template="K = {col_name}")

    # Do NOT add a legend here
    # g.add_legend(...)

    plt.subplots_adjust(top=0.92)
    g.fig.suptitle("K-Means Clustering Scatter Grid across PCA + K", fontsize=14)

    output_path = Path(project_name) / "scatter_grid_pca_kmeans.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved scatter grid to: {output_path}")
    plt.show()
    plt.close()


# -----------------------------------------------
# Plotting Functions
# -----------------------------------------------


def plot_probe_results_from_json(
    json_path, project_name="test_outputs", probe_filter="Linear"
):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Flatten
    flat_records = []
    for model in data:
        for res in model["Results"]:
            for acc in res["ValAccuracies"]:
                flat_records.append(
                    {
                        "Model": model["Model"],
                        "Beta": model["Beta"],
                        "SourceTimestep": model["SourceTimestep"],
                        "TargetTimestep": model["TargetTimestep"],
                        "ProbeType": res["ProbeType"],
                        "PCA": res["PCA"],
                        "ValAccuracy": acc,
                    }
                )

    df = pd.DataFrame(flat_records)
    df = df[df["ProbeType"] == probe_filter]
    df["Beta"] = df["Beta"].astype(float)
    beta_order = sorted(df["Beta"].unique())

    #
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": False,
            "axes.facecolor": "#e8ecf0",
            "axes.edgecolor": "#cccccc",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "lines.linewidth": 1.2,
        }
    )

    # Setup FacetGrid
    g = sns.FacetGrid(df, col="PCA", col_wrap=3, height=2.8, aspect=1.2, sharey=True)

    def draw_minimalist_boxplot(data, **kwargs):
        ax = plt.gca()
        ax.set_facecolor("#e8ecf0")
        data = data.copy()
        data["Beta"] = data["Beta"].astype(float)

        # Violin background
        sns.violinplot(
            data=data,
            x="Beta",
            y="ValAccuracy",
            ax=ax,
            order=beta_order,
            inner=None,
            linewidth=0,
            color="#a0aab8",
            saturation=0.3,
        )

        # Boxplot overlay
        sns.boxplot(
            data=data,
            x="Beta",
            y="ValAccuracy",
            width=0.3,
            ax=ax,
            order=beta_order,
            color="white",
            fliersize=1.5,
            linewidth=0.7,
            boxprops={"facecolor": "white", "edgecolor": "#333", "zorder": 2},
            whiskerprops={"linewidth": 0.7},
            capprops={"linewidth": 0.7},
            medianprops={"color": "black", "linewidth": 1},
        )

        # Median line — map beta to float *positions* on x-axis
        medians = (
            data.groupby("Beta", observed=True)["ValAccuracy"]
            .median()
            .reindex(beta_order)
        )
        x_vals = list(range(len(beta_order)))  # x positions of boxes
        y_vals = medians.values  # medians
        ax.plot(x_vals, y_vals, color="red", linewidth=1.3, zorder=3)

        # Set correct tick labels
        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{b:.4g}" for b in beta_order])  # optional formatting
        ax.set_xlabel(r"$\beta$", fontsize=10)
        ax.set_ylabel(r"Validation Accuracy", fontsize=10)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(True, linestyle="--", alpha=0.4)

        # Debug
        print(
            f"[PCA {data['PCA'].iloc[0]}] Median line: {list(zip(beta_order, y_vals))}"
        )

    g.map_dataframe(draw_minimalist_boxplot)
    g.set_titles(col_template=r"PCA = {col_name}", size=10)
    plt.suptitle(
        r"Validation Accuracy over Number of PCA Components", fontsize=12, y=1.05
    )

    plt.tight_layout()
    plot_path = Path(f"{project_name}/pca_beta_boxplot_style_matched.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved PCA plot to: {plot_path}")
    plt.show()
    plt.close()


def plot_probe_comparison_grid(json_path, project_name="test_outputs"):
    import json
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pathlib import Path

    with open(json_path, "r") as f:
        data = json.load(f)

    # Flatten JSON structure
    flat_records = []
    for model in data:
        for res in model["Results"]:
            for acc in res["ValAccuracies"]:
                flat_records.append(
                    {
                        "Model": model["Model"],
                        "Beta": model["Beta"],
                        "SourceTimestep": model["SourceTimestep"],
                        "TargetTimestep": model["TargetTimestep"],
                        "ProbeType": res["ProbeType"],
                        "PCA": res["PCA"],
                        "ValAccuracy": acc,
                    }
                )

    df = pd.DataFrame(flat_records)
    df["Beta"] = df["Beta"].astype(float)
    beta_order = sorted(df["Beta"].unique())
    pca_order = sorted(df["PCA"].unique())
    probe_order = ["Linear", "Two-Layer"]

    # Style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": False,
            "axes.facecolor": "#e8ecf0",
            "axes.edgecolor": "#cccccc",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "lines.linewidth": 1.2,
        }
    )

    # Calculate generous KDE-based y-limits per probe type
    y_limits = {}
    for probe in probe_order:
        vals = df[df["ProbeType"] == probe]["ValAccuracy"].values
        if len(vals) > 1:
            kde = gaussian_kde(vals)
            x_range = np.linspace(vals.min(), vals.max(), 1000)
            density = kde(x_range)
            threshold = max(density) * 0.01
            included = x_range[density > threshold]
            y_min = included.min() if len(included) else vals.min()
            y_max = included.max() if len(included) else vals.max()
        else:
            y_min, y_max = vals.min(), vals.max()

        # Expand limits generously
        padding = (y_max - y_min) * 0.2 if y_max > y_min else 0.05
        y_limits[probe] = (max(0.0, y_min - padding), min(1.2, y_max + padding))

    # FacetGrid with relaxed height and spacing
    g = sns.FacetGrid(
        df,
        row="ProbeType",
        col="PCA",
        height=3.2,
        aspect=1.2,
        sharey=False,
        row_order=probe_order,
        col_order=pca_order,
        margin_titles=True,
    )

    def draw_boxplot_with_medians(data, **kwargs):
        ax = plt.gca()
        probe_type = data["ProbeType"].iloc[0]
        ax.set_facecolor("#e8ecf0")

        sns.violinplot(
            data=data,
            x="Beta",
            y="ValAccuracy",
            ax=ax,
            order=beta_order,
            inner=None,
            linewidth=0,
            color="#a0aab8",
            saturation=0.3,
        )

        sns.boxplot(
            data=data,
            x="Beta",
            y="ValAccuracy",
            width=0.3,
            ax=ax,
            order=beta_order,
            color="white",
            fliersize=1.5,
            linewidth=0.7,
            boxprops={"facecolor": "white", "edgecolor": "#333", "zorder": 2},
            whiskerprops={"linewidth": 0.7},
            capprops={"linewidth": 0.7},
            medianprops={"color": "black", "linewidth": 1},
        )

        # Median line
        medians = (
            data.groupby("Beta", observed=True)["ValAccuracy"]
            .median()
            .reindex(beta_order)
        )
        x_vals = list(range(len(beta_order)))
        y_vals = medians.values
        ax.plot(x_vals, y_vals, color="red", linewidth=1.3, zorder=3)

        ax.set_xticks(x_vals)
        ax.set_xticklabels([f"{b:.4g}" for b in beta_order])
        ax.set_xlabel(r"$\beta$", fontsize=10)
        ax.set_ylabel(r"Validation Accuracy", fontsize=10)
        ax.set_ylim(*y_limits[probe_type])
        ax.grid(True, linestyle="--", alpha=0.4)

    g.map_dataframe(draw_boxplot_with_medians)
    g.set_titles(col_template="PCA = {col_name}", row_template="", size=10)

    # Row subtitles
    row_labels = {
        "Linear": "(a) Linear Probe Classifier Evaluation",
        "Two-Layer": "(b) Two-Layer Probe Classifier Evaluation",
    }

    g.fig.subplots_adjust(top=0.9, hspace=0.4)

    for i, probe in enumerate(probe_order):
        row_axes = g.axes[i]
        left = row_axes[0].get_position().x0
        right = row_axes[-1].get_position().x1
        center_x = (left + right) / 2
        bottom_y = min(ax.get_position().y0 for ax in row_axes)
        y_offset = 0.08 if i == len(probe_order) - 1 else 0.035

        g.fig.text(
            center_x,
            bottom_y - y_offset,
            row_labels[probe],
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="top",
        )

    plt.suptitle(
        r"Validation Accuracy over Number of PCA Components", fontsize=14, y=1.04
    )
    plot_path = Path(f"{project_name}/combined_probe_accuracy_grid.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved unified comparison plot to: {plot_path}")
    plt.show()
    plt.close()


def plot_pca_across_betas(
    json_path, project_name="test_outputs", probe_filter="Linear"
):
    with open(json_path, "r") as f:
        data = json.load(f)

    # Flatten records
    flat_records = []
    for model in data:
        for res in model["Results"]:
            for acc in res["ValAccuracies"]:
                flat_records.append(
                    {
                        "Model": model["Model"],
                        "Beta": model["Beta"],
                        "SourceTimestep": model["SourceTimestep"],
                        "TargetTimestep": model["TargetTimestep"],
                        "ProbeType": res["ProbeType"],
                        "PCA": res["PCA"],
                        "ValAccuracy": acc,
                    }
                )

    df = pd.DataFrame(flat_records)
    df = df[df["ProbeType"] == probe_filter]
    df["Beta"] = df["Beta"].astype(float)
    df["PCA"] = df["PCA"].astype(int)

    beta_order = sorted(df["Beta"].unique())
    pca_order = sorted(df["PCA"].unique())

    # Global style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": False,
            "axes.facecolor": "#e8ecf0",
            "axes.edgecolor": "#cccccc",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "lines.linewidth": 1.2,
        }
    )

    # FacetGrid with one subplot per Beta
    g = sns.FacetGrid(
        df,
        col="Beta",
        col_wrap=4,
        height=2.8,
        aspect=1.2,
        sharey=True,
        col_order=beta_order,
    )

    def draw_pca_grouped_boxplot(data, **kwargs):
        ax = plt.gca()
        ax.set_facecolor("#e8ecf0")
        data = data.copy()

        sns.violinplot(
            data=data,
            x="PCA",
            y="ValAccuracy",
            ax=ax,
            order=pca_order,
            inner=None,
            linewidth=0,
            color="#a0aab8",
            saturation=0.3,
        )

        sns.boxplot(
            data=data,
            x="PCA",
            y="ValAccuracy",
            width=0.3,
            ax=ax,
            order=pca_order,
            color="white",
            fliersize=1.5,
            linewidth=0.7,
            boxprops={"facecolor": "white", "edgecolor": "#333", "zorder": 2},
            whiskerprops={"linewidth": 0.7},
            capprops={"linewidth": 0.7},
            medianprops={"color": "black", "linewidth": 1},
        )

        medians = (
            data.groupby("PCA", observed=True)["ValAccuracy"]
            .median()
            .reindex(pca_order)
        )
        x_vals = list(range(len(pca_order)))
        y_vals = medians.values
        ax.plot(x_vals, y_vals, color="red", linewidth=1.3, zorder=3)

        ax.set_xticks(x_vals)
        ax.set_xticklabels(pca_order)
        ax.set_xlabel("PCA Components", fontsize=10)
        ax.set_ylabel("Validation Accuracy", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

        # Debug
        beta_val = data["Beta"].iloc[0]
        print(f"[β={beta_val}] Median line: {list(zip(pca_order, y_vals))}")

    g.map_dataframe(draw_pca_grouped_boxplot)
    g.set_titles(col_template=r"$\beta$ = {col_name}", size=10)
    plt.suptitle(
        f"Validation Accuracy by PCA Dimension\n({probe_filter} Probes)",
        fontsize=14,
        y=1.05,
    )

    plt.tight_layout()
    plot_path = Path(f"{project_name}/pca_over_beta_faceted_by_beta.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved PCA-vs-Beta plot to: {plot_path}")
    plt.show()
    plt.close()


def plot_pca_comparison_by_beta_grid(json_path, project_name="test_outputs"):
    import json
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pathlib import Path

    with open(json_path, "r") as f:
        data = json.load(f)

    # Flatten records
    flat_records = []
    for model in data:
        for res in model["Results"]:
            for acc in res["ValAccuracies"]:
                flat_records.append(
                    {
                        "Model": model["Model"],
                        "Beta": model["Beta"],
                        "SourceTimestep": model["SourceTimestep"],
                        "TargetTimestep": model["TargetTimestep"],
                        "ProbeType": res["ProbeType"],
                        "PCA": res["PCA"],
                        "ValAccuracy": acc,
                    }
                )

    df = pd.DataFrame(flat_records)
    df["Beta"] = df["Beta"].astype(float)
    df["PCA"] = df["PCA"].astype(int)

    probe_order = ["Linear", "Two-Layer"]
    beta_order = sorted(df["Beta"].unique())
    pca_order = sorted(df["PCA"].unique())

    # Set global style
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": False,
            "axes.facecolor": "#e8ecf0",
            "axes.edgecolor": "#cccccc",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "lines.linewidth": 1.2,
        }
    )

    # KDE-based y-limits per probe type to avoid violin cutoffs
    y_limits = {}
    for probe in probe_order:
        y_vals = df[df["ProbeType"] == probe]["ValAccuracy"].values
        if len(y_vals) > 1:
            kde = gaussian_kde(y_vals)
            x_range = np.linspace(y_vals.min(), y_vals.max(), 500)
            density = kde(x_range)
            density_cutoff = x_range[density > max(density) * 0.01]
            y_min = density_cutoff.min() if len(density_cutoff) else y_vals.min()
            y_max = density_cutoff.max() if len(density_cutoff) else y_vals.max()
        else:
            y_min, y_max = y_vals.min(), y_vals.max()

        padding = (y_max - y_min) * 0.25 if y_max > y_min else 0.05
        y_limits[probe] = (max(0.0, y_min - padding), min(1.2, y_max + padding))

    # FacetGrid (with sharey=False to allow custom limits)
    g = sns.FacetGrid(
        df,
        row="ProbeType",
        col="Beta",
        height=3.0,
        aspect=1.2,
        sharey=False,
        row_order=probe_order,
        col_order=beta_order,
        margin_titles=True,
    )

    def draw_pca_vs_accuracy_boxplot(data, **kwargs):
        ax = plt.gca()
        probe_type = data["ProbeType"].iloc[0]
        ax.set_facecolor("#e8ecf0")

        sns.violinplot(
            data=data,
            x="PCA",
            y="ValAccuracy",
            ax=ax,
            order=pca_order,
            inner=None,
            linewidth=0,
            color="#a0aab8",
            saturation=0.3,
        )

        sns.boxplot(
            data=data,
            x="PCA",
            y="ValAccuracy",
            width=0.3,
            ax=ax,
            order=pca_order,
            color="white",
            fliersize=1.5,
            linewidth=0.7,
            boxprops={"facecolor": "white", "edgecolor": "#333", "zorder": 2},
            whiskerprops={"linewidth": 0.7},
            capprops={"linewidth": 0.7},
            medianprops={"color": "black", "linewidth": 1},
        )

        medians = (
            data.groupby("PCA", observed=True)["ValAccuracy"]
            .median()
            .reindex(pca_order)
        )
        x_vals = list(range(len(pca_order)))
        y_vals = medians.values
        ax.plot(x_vals, y_vals, color="red", linewidth=1.3, zorder=3)

        ax.set_xticks(x_vals)
        ax.set_xticklabels(pca_order)
        ax.set_xlabel("PCA Components", fontsize=10)
        ax.set_ylabel("Validation Accuracy", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)

        # Apply custom y-limits
        y_min, y_max = y_limits[probe_type]
        ax.set_ylim(y_min, y_max)

    g.map_dataframe(draw_pca_vs_accuracy_boxplot)
    g.set_titles(col_template=r"$\beta$ = {col_name}", row_template="", size=10)

    # Row annotations
    row_labels = {
        "Linear": "(a) Linear Probe Classifier Evaluation",
        "Two-Layer": "(b) Two-Layer Probe Classifier Evaluation",
    }

    g.fig.subplots_adjust(top=0.92, hspace=0.4)

    for i, probe in enumerate(probe_order):
        row_axes = g.axes[i]
        left = row_axes[0].get_position().x0
        right = row_axes[-1].get_position().x1
        center_x = (left + right) / 2
        bottom_y = min(ax.get_position().y0 for ax in row_axes)
        y_offset = 0.08 if i == len(probe_order) - 1 else 0.035

        g.fig.text(
            center_x,
            bottom_y - y_offset,
            row_labels[probe],
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="top",
        )

    plt.suptitle(
        "Validation Accuracy across PCA Dimensions per Beta Value", fontsize=14, y=1.05
    )
    plot_path = Path(f"{project_name}/pca_vs_beta_combined_grid.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved combined PCA/Beta grid plot to: {plot_path}")
    plt.show()
    plt.close()


def plot_kmeans_grid(
    clustering_results, metric="Silhouette", project_name="test_outputs"
):
    df = pd.DataFrame(clustering_results)
    df["PCA"] = df["PCA"].astype(str)
    df["K"] = df["K"].astype(int)

    if metric not in df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in results. Available: {list(df.columns)}"
        )

    # Set global style to match example
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": False,  # Set to True if you want full LaTeX rendering
            "axes.facecolor": "#e8ecf0",
            "axes.edgecolor": "#cccccc",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "grid.linestyle": "--",
            "grid.alpha": 0.4,
            "lines.linewidth": 1.2,
        }
    )

    g = sns.FacetGrid(df, row="PCA", height=2.5, aspect=1.5, sharey=True)
    g.map_dataframe(sns.barplot, x="K", y=metric, color="steelblue", edgecolor="black")

    g.set_titles(row_template="PCA = {row_name}")
    g.set_axis_labels("K (# Clusters)", f"{metric} Score")
    plt.subplots_adjust(top=0.92)
    g.fig.suptitle(f"K-Means Clustering Performance\nMetric: {metric}", fontsize=13)

    save_path = Path(project_name) / f"kmeans_clustering_grid_{metric.lower()}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved K-Means grid plot to: {save_path}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    json_path = generate_structured_dummy_probe_data()

    # ---- Probe Evaluation Plots ----
    # Per-ProbeType (individual)
    plot_probe_results_from_json(json_path, probe_filter="Linear")
    plot_probe_results_from_json(json_path, probe_filter="Two-Layer")

    plot_pca_across_betas(json_path, probe_filter="Linear")
    plot_pca_across_betas(json_path, probe_filter="Two-Layer")

    # Combined comparison grid
    plot_probe_comparison_grid(json_path)
    plot_pca_comparison_by_beta_grid(json_path)

    # ---- K-Means Clustering Example ----
    df = generate_dummy_pca_kmeans_data(
        pca_values=[2, 3, 5, 10, 20], k_values=[2, 3, 5, 10, 20]
    )
    plot_pca_kmeans_scatter_grid(df)

    # Test K-Means grid plotting with dummy data
    # Generate dummy K-Means results
    dummy_kmeans_results = generate_dummy_kmeans_results()

    # Test plot with dummy metrics
    plot_kmeans_grid(dummy_kmeans_results, metric="Silhouette")
    plot_kmeans_grid(dummy_kmeans_results, metric="ARI")
    plot_kmeans_grid(dummy_kmeans_results, metric="NMI")

# CUDA_VISIBLE_DEVICES=1 python
