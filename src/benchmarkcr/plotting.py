import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib import patches
import numpy as np
import pandas as pd
from pathlib import Path
from .utils import dload  
from itertools import combinations
from .logging_config import log


def adjust_text_positions(coords, sizes, min_distance=0.08, max_y=1.0):

    adjusted = []
    text_height = 0.04  # Estimated text height in plot coordinates

    for (x, y, idx) in coords:
        base_offset = np.sqrt(sizes.loc[idx]) * 0.04 if idx in sizes else 0.04
        adj_y = y + base_offset  # Move text upwards
        safety = 0

        # Ensure text stays within plot area
        max_possible_y = max_y - text_height

        while safety < 20:
            conflict = False
            # Check against existing labels
            for (ax, aay, _) in adjusted:
                if abs(x - ax) < 0.01 and abs(adj_y - aay) < min_distance:
                    conflict = True
                    break

            if conflict:
                adj_y += 0.03 + (base_offset * 0.1)  # Move further upwards
                safety += 1

                # If we're going beyond plot area, stop
                if adj_y > max_possible_y:
                    adj_y = max_possible_y
                    break
            else:
                # Clamp final position
                adj_y = min(adj_y, max_possible_y)
                adjusted.append((x, adj_y, idx))
                break

    return adjusted



def plot_precision_recall_curve():
    pra = dload("pra")
    config = dload("config")
    plot_config = config["plotting"]

    # Create figure using rcParams defaults (figsize and dpi are already set)
    fig, ax = plt.subplots()
    ax.set_xscale("log")

    # Color map from rcParams (no need to get from config again)
    cmap = get_cmap()  # Uses rcParams['image.cmap'] by default
    num_colors = len(pra) if isinstance(pra, dict) else 1
    colors = [cmap(float(i) / max(num_colors - 1, 1)) for i in range(num_colors)]

    if isinstance(pra, dict):
        for (key, val), color in zip(pra.items(), colors):
            val = val[val.tp > 10]
            ax.plot(val.tp, val.precision, c=color, label=key, linewidth=2, alpha=0.8)
    else:
        pra = pra[pra.tp > 10]
        ax.plot(pra.tp, pra.precision, c="black", label="Precision Recall Curve", linewidth=2, alpha=0.8)

    # Labels and title (sizes handled by rcParams)
    ax.set(title="Precision-Recall Performance Across Datasets",
           xlabel="Number of True Positives (TP)",
           ylabel="Precision")
    ax.legend(loc="upper right", frameon=True)

    # Grid and spines (styles handled by rcParams)
    ax.grid(True)  # Style comes from rcParams
    # Spines visibility handled by rcParams ('axes.spines.right' and 'axes.spines.top')

    # Save handling (output config still needed)
    if plot_config["save"]["save_plot"]:
        output_type = plot_config["save"]["output_type"]
        output_path = Path(plot_config["save"]["output_folder"]) / f"precision_recall_curve.{output_type}"
        fig.savefig(output_path, bbox_inches="tight", format=output_type)  # dpi comes from rcParams

    plt.show()



def plot_percomplex_scatter(n_top=10):
    config = dload("config")
    plot_config = config["plotting"]
    rdict = dload("pra_percomplex")

    # Ensure there are at least two datasets to compare
    if len(rdict) < 2:
        print("Skipping plot: At least two datasets are required for per-complex scatter plot.")
        return
    
    column_pairs = list(combinations(rdict.keys(), 2))
    df = pd.DataFrame()
    
    # Data loading
    for i, (key, val) in enumerate(rdict.items()):
        val = val.rename(columns={"auc_score": key})
        if i == 0:
            df = val.copy().drop(columns=["Genes", "Length", "used_genes"])
        else:
            df = pd.concat([df, val[key]], axis=1)
    
    # Plotting
    for pair in column_pairs:
        extreme_indices_0 = df[pair[0]].sort_values(ascending=False).head(n_top).index
        extreme_indices_1 = df[pair[1]].sort_values(ascending=False).head(n_top).index
        
        # Figure created with rcParams defaults
        fig, ax = plt.subplots()
        
        # Base scatter plot (keep color overrides)
        sizes = df['n_used_genes'] * 8
        ax.scatter(df[pair[0]], df[pair[1]], 
                 edgecolors="black", 
                 marker='o', 
                 s=sizes, 
                 linewidth=0.7, 
                 zorder=1)
        
        # Highlight significant points
        significant_indices = extreme_indices_0.union(extreme_indices_1)
        sig_sizes = df.loc[significant_indices, 'n_used_genes'] * 8
        ax.scatter(df.loc[significant_indices, pair[0]],
                 df.loc[significant_indices, pair[1]],
                 facecolors='black', 
                 edgecolors='black', 
                 s=sig_sizes, 
                 linewidth=0.1, 
                 zorder=2)

        all_points = list(zip(df[pair[0]], df[pair[1]]))
        coords = sorted([(df.loc[idx, pair[0]], df.loc[idx, pair[1]], idx) 
                       for idx in significant_indices], key=lambda c: (-c[1], -c[0]))
        
        adjusted_coords = adjust_text_positions(coords, sig_sizes)

        # Draw vertical lines and right-aligned text
        for x, adj_y, idx in adjusted_coords:
            y = df.loc[idx, pair[1]]
            ax.plot([x, x], [y, adj_y], 
                   color='black', 
                   linewidth=0.7, 
                   alpha=0.3,
                   zorder=3)
            
            ax.text(x, adj_y + 0.005,
                   df.loc[idx, 'Name'], 
                   fontsize=6, 
                   ha='left', 
                   va='bottom', 
                   linespacing=1.5,
                   zorder=4,
                   bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5)
                   )

        # Reference line and labels
        ax.plot([0, 1], [0, 1], 
              linestyle='-', 
              color='lightgray', 
              alpha=0.4,
              zorder=0)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Labels use rcParams sizes automatically
        ax.set_xlabel(f"{pair[0]} PR-AUC score")
        ax.set_ylabel(f"{pair[1]} PR-AUC score")
        ax.set_title(f"{pair[0]} vs {pair[1]} - Comparison of complex performance")
        
        plt.tight_layout()
        
        # Save handling
        if plot_config["save"]["save_plot"]:
            output_type = plot_config["save"]["output_type"]
            output_path = Path(plot_config["save"]["output_folder"]) / f"percomplex_scatter_{pair[0]}_vs_{pair[1]}.{output_type}"
            fig.savefig(output_path, bbox_inches="tight", format=output_type)
        
        plt.show()



def plot_percomplex_scatter_bysize():
    config = dload("config")
    plot_config = config["plotting"]
    rdict = dload("pra_percomplex")
    
    for key, per_complex in rdict.items():
        sorted_pc = per_complex.sort_values(by="auc_score", ascending=False, na_position="last")
        top_10, rest = sorted_pc.head(10), sorted_pc.iloc[10:]
        
        # Create figure using rcParams defaults
        fig, ax = plt.subplots()
        
        # Base scatter plot (keep color overrides)
        ax.scatter(
            rest.n_used_genes, rest.auc_score,
            edgecolors="black",
            linewidth=0.5, 
            s=rest.n_used_genes * 10, 
            label="Other Complexes"
        )
        
        # Top 10 scatter plot
        ax.scatter(
            top_10.n_used_genes, top_10.auc_score,
            facecolors='black', 
            edgecolors='black',
            linewidth=0.5, 
            s=top_10.n_used_genes * 10, 
            label="Top 10 AUC Scores"
        )

        # Text annotation handling
        coords = [(row.n_used_genes, row.auc_score, idx) for idx, row in top_10.iterrows()]
        sizes = top_10.n_used_genes * 10
        adjusted_coords = adjust_text_positions(coords, sizes)
        
        for x, adj_y, idx in adjusted_coords:
            y = top_10.loc[idx, "auc_score"]
            ax.plot([x, x], [y, adj_y], 
                   color='black', 
                   linewidth=0.7, 
                   alpha=0.3,
                   zorder=3)
            
            ax.text(x, adj_y + 0.005,
                   top_10.loc[idx, 'Name'], 
                   fontsize=6, 
                   ha='left', 
                   va='bottom', 
                   linespacing=1.5,
                   zorder=4,
                   bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1.5))

        # Axis configuration
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.set_xlabel("Number of genes in the complex")  # Font size from rcParams
        ax.set_ylabel("PR-AUC score")                  # Font size from rcParams
        ax.set_title(f"{key} - Complex performance: PR-AUC score across complex size")
        ax.grid(False)

        plt.tight_layout()

        # Save handling (dpi comes from rcParams)
        if plot_config["save"]["save_plot"]:
            output_type = plot_config["save"]["output_type"]
            output_path = Path(plot_config["save"]["output_folder"]) / f"percomplex_scatter_by_complexsize_{key}.{output_type}"
            fig.savefig(output_path, bbox_inches="tight", format=output_type)

        plt.show()
        plt.close(fig)



def plot_complex_contributions(min_pairs=10, min_precision_cutoff=0.2, num_complex_to_show=5, y_lim=None, fig_title=None, fig_labs=['Fraction of TP', 'Precision']):
    config = dload("config")
    plot_config = config["plotting"]
    plot_data_dict = dload("complex_contributions")
    for key, plot_data in plot_data_dict.items():
        s = plot_data.set_index('Name').sum()
        find_last_precision = s[s > min_pairs].index[-1]
        plot_data = plot_data.drop_duplicates(subset='Name')
        cont_stepwise_anno = plot_data['Name']
        cont_stepwise_mat = plot_data.drop(columns=['Name'])
        tmp_TP = cont_stepwise_mat.sum(axis=0)
        Precision_ind = (tmp_TP >= min_pairs)
        cont_stepwise_mat = cont_stepwise_mat.loc[:, Precision_ind]
        tmp = cont_stepwise_mat.columns
        y = np.array([float(col.split('_')[1]) if isinstance(col, str) and '_' in col else col for col in tmp])
        x = cont_stepwise_mat.sum(axis=0)
        mx, nx = cont_stepwise_mat.shape[0], cont_stepwise_mat.shape[1]
        tmp = np.tile(x, (mx, 1))
        x = cont_stepwise_mat.values / tmp
        x_df = pd.DataFrame(x, index=cont_stepwise_anno, columns=cont_stepwise_mat.columns)
        ind_for_mean = y >= (find_last_precision - min_precision_cutoff)
        if sum(ind_for_mean) == 0:
            log.info("No values above 'min.precision.cutoff'")
            return False
        if sum(ind_for_mean) == 1:
            log.info("Only one value above 'min.precision.cutoff', unable to calculate meaningful contribution structure")
            return False
        # Select top complexes
        a = x_df.loc[:, ind_for_mean].mean(axis=1).sort_values()[-num_complex_to_show:]
        subset = x_df.loc[a.index, :]
        # Use the RdYlBu colormap for the top 10 points
        cmap = plt.get_cmap()  # Get default from rcParams
        colors = cmap(np.linspace(0, 1, num_complex_to_show))
        colors = np.vstack(([0.5, 0.5, 0.5, 1.0], colors))
        others = pd.DataFrame(1 - subset.sum(axis=0), columns=['others']).T
        merged = pd.concat([others, subset], ignore_index=False)
        x = merged.to_numpy()
        x1 = np.zeros_like(x)
        x2 = np.zeros_like(x)
        for i in range(x.shape[0]):
            if i == 0:
                x2[i, :] = x[0, :]
            elif i == 1:
                x1[i, :] = x[0, :]
            else:
                x1[i, :] = x[:i, :].sum(axis=0)
            if i > 0:
                x2[i, :] = x[:i + 1, :].sum(axis=0)

        # Dynamically set y-axis limits based on the precision values
        if y_lim is None:
            y_lim = (max(0, min(y) - 0.1), max(y) + 0.1)  
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [5, 1]})
        ax[0].set_xlim(0, 1)
        ax[0].set_ylim(0.05, find_last_precision)
        ax[0].set_xlabel(fig_labs[0])
        ax[0].set_ylabel(fig_labs[1])
        ax[0].set_title(fig_title if fig_title else f"{key} - Contribution of complexes")
        for i in range(x.shape[0]):
            ax[0].fill_betweenx(y, x1[i, :], x2[i, :], color=colors[i], edgecolor='white')

        # Legend handling (keep custom settings)
        legend_labels = [f'{label[:20]}.' for label in merged.index]
        patches_list = [patches.Patch(color=colors[i], label=legend_labels[i]) for i in range(len(legend_labels))]
        ax[1].axis('off')
        ax[1].legend(handles=patches_list, loc='center', ncol=3, frameon=False, title="Complexes")
        plt.tight_layout()

        # Save handling (remove explicit dpi)
        if plot_config["save"]["save_plot"]:
            output_type = plot_config["save"]["output_type"]
            output_folder = Path(plot_config["save"]["output_folder"])
            output_folder.mkdir(parents=True, exist_ok=True)
            output_path = output_folder / f"complex_contributions_{key}.{output_type}"
            fig.savefig(output_path, bbox_inches="tight", format=output_type)
        plt.show()
        plt.close()



def plot_significant_complexes():
    config = dload("config")
    plot_config = config["plotting"]
    pra_percomplex = dload("pra_percomplex")

    # Define thresholds and prepare data
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    datasets = list(pra_percomplex.keys())
    num_datasets = len(datasets)

    # Create a DataFrame to store results
    df = pd.DataFrame(index=thresholds)
    for key, complex_data in pra_percomplex.items():
        df[key] = [complex_data.query(f'auc_score >= {t}').shape[0] for t in thresholds]

    # Create figure
    fig, ax = plt.subplots()

    # Use colormap from rcParams
    cmap = plt.get_cmap()
    colors = [cmap(i / (num_datasets + 1)) for i in range(1, num_datasets + 1)]

    # Plot bars
    bar_width = 0.8 / num_datasets  # Dynamic width based on dataset count
    for i, dataset in enumerate(datasets):
        x = np.arange(len(thresholds)) + i * bar_width
        ax.bar(x, df[dataset], width=bar_width, color=colors[i], edgecolor='black', label=dataset)

    # Customize x-axis labels
    ax.set_xticks(np.arange(len(thresholds)) + (num_datasets - 1) * bar_width / 2)
    ax.set_xticklabels(thresholds, rotation=0, ha='center')

    # Set title and axis labels (handled by rcParams)
    ax.set_title("Number of significant complexes above PR-AUC thresholds")
    ax.set_xlabel("PR-AUC score thresholds")
    ax.set_ylabel("Number of complexes")

    # Add grid (already handled by rcParams, but ensured)
    ax.grid(axis='y')

    # Add legend
    ax.legend(loc='upper right')

    # Adjust layout
    plt.tight_layout()

    # Save figure if required
    if plot_config["save"]["save_plot"]:
        output_type = plot_config["save"]["output_type"]
        output_folder = Path(plot_config["save"]["output_folder"])
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / f"number_of_significant_complexes.{output_type}"
        plt.savefig(output_path, bbox_inches='tight', format=output_type)

    plt.show()
    plt.close()
    return df



def plot_auc_scores():
    config = dload("config")
    plot_config = config["plotting"]
    pra_dict = dload("pr_auc")

    # Prepare data
    datasets = list(pra_dict.keys())
    auc_scores = list(pra_dict.values())

    # Create figure and axis
    fig, ax = plt.subplots()

    # Use colormap from rcParams
    cmap = plt.get_cmap()
    num_datasets = len(datasets)
    colors = [cmap(i / (num_datasets + 1)) for i in range(1, num_datasets + 1)]

    # Plot bars
    bars = ax.bar(datasets, auc_scores, color=colors, edgecolor="black")

    # Set y-axis limits dynamically
    ax.set_ylim(0, max(auc_scores) + 0.01)

    # Set title and labels
    ax.set_title("AUC scores for the datasets")
    ax.set_ylabel("AUC score")

    # Add grid (already handled by rcParams)
    ax.grid(axis='y')

    # Save the figure if required
    if plot_config["save"]["save_plot"]:
        output_type = plot_config["save"]["output_type"]
        output_folder = Path(plot_config["save"]["output_folder"])
        output_folder.mkdir(parents=True, exist_ok=True)
        output_path = output_folder / f"prauc_scores.{output_type}"
        plt.savefig(output_path, bbox_inches='tight', format=output_type)

    plt.show()
    plt.close()
    return pra_dict
