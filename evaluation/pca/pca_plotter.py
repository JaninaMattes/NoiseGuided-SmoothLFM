import os
import sys
import json
from pathlib import Path

import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt


# Helper utilities

# Project root path setup
project_root = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../")
)
sys.path.append(project_root)

# Project-specific modules

# Torch precision
torch.set_float32_matmul_precision("high")


def plot_probe_comparison_grid(json_path, log_path="test_outputs"):
    import json
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pathlib import Path

    with open(json_path, "r") as f:
        data = json.load(f)

    # Flatten data
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

    # Create row-wise y-axis limits
    y_limits = df.groupby("ProbeType")["ValAccuracy"].max().to_dict()
    y_limits = {
        k: min(1.2, v * 1.1) for k, v in y_limits.items()
    }  # Add margin, clamp at 1.2

    # FacetGrid (don't share y!)
    g = sns.FacetGrid(
        df,
        row="ProbeType",
        col="PCA",
        height=2.8,
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
        ax.set_ylim(0, y_limits.get(probe_type, 1.0))
        ax.grid(True, linestyle="--", alpha=0.4)

    g.map_dataframe(draw_boxplot_with_medians)
    g.set_titles(col_template="PCA = {col_name}", row_template="", size=10)

    # Row subtitles
    row_labels = {
        "Linear": "(a) Linear Probe Classifier Evaluation",
        "Two-Layer": "(b) Two-Layer Probe Classifier Evaluation",
    }
    row_offsets = {
        "Linear": 0.035,
        "Two-Layer": 0.15,
    }

    g.fig.subplots_adjust(top=0.92, hspace=0.35)
    for i, probe in enumerate(probe_order):
        row_axes = g.axes[i]
        left = row_axes[0].get_position().x0
        right = row_axes[-1].get_position().x1
        center_x = (left + right) / 2
        bottom_y = min(ax.get_position().y0 for ax in row_axes)
        y_offset = row_offsets[probe]
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
        r"Validation Accuracy over Number of PCA Components", fontsize=14, y=1.05
    )
    plot_path = Path(f"{log_path}/combined_probe_accuracy_grid.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved unified comparison plot to: {plot_path}")
    plt.show()
    plt.close()


def plot_pca_comparison_by_beta_grid(json_path, log_path="test_outputs"):
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

    # Set style
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

    # Calculate y-axis limits per ProbeType
    y_limits = {
        probe: (
            df[df["ProbeType"] == probe]["ValAccuracy"].min(),
            df[df["ProbeType"] == probe]["ValAccuracy"].max(),
        )
        for probe in probe_order
    }

    # Use sharey=False for independent scaling
    g = sns.FacetGrid(
        df,
        row="ProbeType",
        col="Beta",
        height=2.8,
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

        group_counts = data.groupby("PCA")["ValAccuracy"].count()
        has_multiple = group_counts.min() > 1

        if has_multiple:
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
        else:
            print(f"[WARN] Only one value per PCA for {probe_type}. Using stripplot.")
            sns.stripplot(
                data=data,
                x="PCA",
                y="ValAccuracy",
                ax=ax,
                order=pca_order,
                color="black",
                size=5,
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

        y_min, y_max = y_limits[probe_type]
        padding = (y_max - y_min) * 0.1 if y_max > y_min else 0.05
        ax.set_ylim(y_min - padding, y_max + padding)

    g.map_dataframe(draw_pca_vs_accuracy_boxplot)
    g.set_titles(col_template=r"$\beta$ = {col_name}", row_template="", size=10)

    # Row titles
    row_labels = {
        "Linear": "(a) Linear Probe Classifier Evaluation",
        "Two-Layer": "(b) Two-Layer Probe Classifier Evaluation",
    }

    g.fig.subplots_adjust(top=0.92, hspace=0.35)

    for i, probe in enumerate(probe_order):
        row_axes = g.axes[i]
        left = row_axes[0].get_position().x0
        right = row_axes[-1].get_position().x1
        center_x = (left + right) / 2
        bottom_y = min(ax.get_position().y0 for ax in row_axes)
        y_offset = 0.1 if i == len(probe_order) - 1 else 0.035

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
    plot_path = Path(f"{log_path}/pca_vs_beta_combined_grid.png")
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"[INFO] Saved combined PCA/Beta grid plot to: {plot_path}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    json_path = "results/pca_evaluation/2025-07-27_14-31-02/PCA_Quantitative_Varying_Beta_Denoising/group_probe_results.json"
    log_path = "results/PCA_Quantitative_Eval/logs"

    mkdir_path = Path(log_path)
    mkdir_path.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    # print(json.dumps(data, indent=2))
    # print("\nTop-level is a list of length:", len(data))
    # print("First item keys:", list(data[0].keys()))
    # print("First item['Results'] type/length:", type(data[0]['Results']), len(data[0]['Results']))
    # print("First Result sample:", data[0]['Results'][0])

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
    df = pd.DataFrame(flat_records)
    df["Beta"] = df["Beta"].astype(float)
    df["PCA"] = df["PCA"].astype(int)
    df["ValAccuracy"] = df["ValAccuracy"].astype(float)

    # Debugging output
    print(df.dtypes)
    print(df.head())
    print(df.describe())

    # # ---- RUN PLOTS ----
    # plot_probe_comparison_grid(json_path, log_path)
    # plot_pca_comparison_by_beta_grid(json_path, log_path)

    # CUDA_VISIBLE_DEVICES=2 python ...
