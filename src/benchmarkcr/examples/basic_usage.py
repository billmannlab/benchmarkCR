"""
Basic usage example of the benchmarkCR package.
Demonstrates initialization, data loading, analysis, and plotting.
"""
#%%
import benchmarkcr

inputs = {

    # "depmap all": {
    #     "path": ("../../../../datasets/depmap_geneeffect_all_cellines.csv"), 
    #     "sort": "high"
    # },

    "Melanoma (63 Screens)": {
        "path": benchmarkcr.get_example_data_path("melanoma_cell_lines_500_genes.csv"), 
        "sort": "high"
    },
    # "Liver (24 Screens)": {
    #     "path": benchmarkcr.get_example_data_path("liver_cell_lines_500_genes.csv"), 
    #     "sort": "high"
    # },
    # "Neuroblastoma (37 Screens)": {
    #     "path": benchmarkcr.get_example_data_path("neuroblastoma_cell_lines_500_genes.csv"), 
    #     "sort": "high"
    # },
}

# Initialize logger, config, and output folder
benchmarkcr.initialize()

# Load datasets and gold standard terms
data, _ = benchmarkcr.load_datasets(inputs)
terms, genes_in_terms = benchmarkcr.load_gold_standard()

# Run analysis
for name, dataset in data.items():
    df, pr_auc = benchmarkcr.pra(name, dataset)
    #pc = benchmarkcr.pra_percomplex(name, dataset)
    fpc = benchmarkcr.fast_pra_percomplex(name, dataset) 
    cc = benchmarkcr.complex_contributions(name)

# Generate plots
benchmarkcr.plot_auc_scores()
benchmarkcr.plot_precision_recall_curve()
benchmarkcr.plot_percomplex_scatter()
benchmarkcr.plot_complex_contributions()
benchmarkcr.plot_significant_complexes()


# %%
