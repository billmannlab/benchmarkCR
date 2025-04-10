import numpy as np
import pandas as pd
from sklearn import metrics
from .utils import dsave, dload  
from .preprocessing import filter_matrix_by_genes 
import os
from art import tprint
import matplotlib.pyplot as plt
from .logging_config import log
from bitarray import bitarray
from tqdm.notebook import tqdm
tqdm.pandas()


def deep_update(source, overrides):
    """Recursively update the source dict with the overrides."""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            deep_update(source[key], value)
        else:
            source[key] = value
    return source


def initialize(config={}):

    log.info("******************************************************************")
    log.info("🧬 benchmarkCR: Systematic CRISPR screen benchmarking framework")
    log.info("******************************************************************")
    log.started("Initialization")

    result_file = "result.pkl"
    if os.path.exists(result_file):
        log.info(f"{result_file} already exists. It will be removed and recreated.")
        os.remove(result_file)  # Remove the file

    default_config = {
        "min_complex_size": 3,
        "min_complex_size_for_percomplex": 3,
        "output_folder": "output",
        "gold_standard": "CORUM",
        "color_map": "RdYlBu",  # Options include: 'tab10', 'tab20', 'Set1', etc.
        "jaccard": True,    
        "plotting": {
            "save": {
                "save_plot": True,
                "output_type": "png",
                "output_folder": "./output",
            }
        },
        "preprocessing": {
            "normalize": False,
            "fill_na": False,
            "drop_na": True,
        }
    }
    
    log.progress("Saving configuration settings.")   
    if config is not None:
        config = deep_update(default_config, config)
    else:
        config = default_config
        
    dsave(config, "config")
    log.progress("Updating matplotlib settings.")
    plt.rcParams.update({
        # Font settings
        'font.size': 7,                # General font size
        'axes.titlesize': 10,          # Title size
        'axes.labelsize': 7,           # Axis labels (xlabel/ylabel)
        'legend.fontsize': 7,          # Legend text
        'xtick.labelsize': 6,          # X-axis tick labels
        'ytick.labelsize': 6,          # Y-axis tick labels
        'lines.linewidth': 1.5,        # Line width for plots
        'figure.dpi': 300,             # Figure resolution
        'figure.figsize': (8, 6),      # Default figure size
        'grid.linestyle': '--',        # Grid line style
        'grid.linewidth': 0.5,         # Grid line width
        'grid.alpha': 0.2,             # Grid transparency
        'axes.spines.right': False,    # Hide right spine
        'axes.spines.top': False,      # Hide top spine
        'image.cmap': config['color_map'],        # Default colormap
        'axes.edgecolor': 'black',                # Axis edge color
        'axes.facecolor': 'none',                 # Transparent axes background
        'mathtext.fontset': 'dejavusans',   # ADD THIS TO PREVENT cmsy10
        'text.usetex': False                # Ensure LaTeX is off
    })
    log.done("Matplotlib settings updated.")
    output_folder = config.get("output_folder", "output")
    os.makedirs(output_folder, exist_ok=True)
    log.progress(f"Output folder '{output_folder}' ensured to exist.")
    log.done("Initialization completed. ")
    tprint("benchmarkCR",font="standard")



def pra(dataset_name, matrix, is_corr=False):
    log.info(f"******************** {dataset_name} ********************")
    log.started(f"** Global Precision-Recall Analysis - {dataset_name} **")

    terms_data = dload("tmp", "terms")
    if terms_data is None or not isinstance(terms_data, pd.DataFrame):
        raise ValueError("Expected 'terms' to be a DataFrame, but got None or invalid type.")
    terms = terms_data.reset_index(drop=True)
    genes_present = dload("tmp", "genes_present_in_terms")
    sorting = dload("input", "sorting")
    sort_order = sorting.get(dataset_name, "high")

    if not is_corr:
        matrix = perform_corr(matrix, "numpy")
    matrix = filter_matrix_by_genes(matrix, genes_present)

    log.info(f"Matrix shape: {matrix.shape}")
    df = binary(matrix)
    log.info(f"Pair-wise shape: {df.shape}")
    df = quick_sort(df, ascending=(sort_order == "low"))
    df = df.reset_index(drop=True)

    # Build gold standard: map pair → complex ID
    gold_pair_to_complex = {}
    for idx, row in terms.iterrows():
        genes = row.used_genes
        if len(genes) < 2:
            continue
        for i, g1 in enumerate(genes):
            for g2 in genes[i + 1:]:
                pair = tuple(sorted((g1, g2)))
                gold_pair_to_complex[pair] = idx

    # Label predictions and complex IDs
    complex_ids = []
    predictions = []
    for g1, g2 in zip(df["gene1"], df["gene2"]):
        pair = tuple(sorted((g1, g2)))
        if pair in gold_pair_to_complex:
            predictions.append(1)
            complex_ids.append(gold_pair_to_complex[pair])
        else:
            predictions.append(0)
            complex_ids.append(0)

    df["prediction"] = predictions
    df["complex_id"] = complex_ids

    if df["prediction"].sum() == 0:
        log.info("No true positives found in dataset.")
        pr_auc = np.nan
    else:
        tp = df["prediction"].cumsum()
        df["tp"] = tp
        precision = tp / (np.arange(len(df)) + 1)
        recall = tp / tp.iloc[-1]
        pr_auc = metrics.auc(recall, precision)
        df["precision"] = precision
        df["recall"] = recall

    log.info(f"PR-AUC: {pr_auc:.4f}, Number of true positives: {df['prediction'].sum()}")
    dsave(df, "pra", dataset_name)
    dsave(pr_auc, "pr_auc", dataset_name)
    log.done(f"Global PRA completed for {dataset_name}")
    return df, pr_auc




def fast_pra_percomplex(dataset_name, matrix, is_corr=False):

    log.started(f"*** Per-complex PRA started - {dataset_name} ***")
    config = dload("config")
    terms = dload("tmp", "terms").reset_index(drop=True)
    genes_present = dload("tmp", "genes_present_in_terms")
    sorting = dload("input", "sorting")
    sort_order = sorting.get(dataset_name, "high")

    if not is_corr:
        matrix = perform_corr(matrix, "numpy")
    matrix = filter_matrix_by_genes(matrix, genes_present)
    log.info(f"Matrix shape: {matrix.shape}")
    df = binary(matrix)
    log.info(f"Pair-wise shape: {df.shape}")

    df = quick_sort(df, ascending=(sort_order == "low"))
    pairwise_df = df.reset_index(drop=True)

    pairwise_df['gene1'] = pairwise_df['gene1'].astype("category")
    pairwise_df['gene2'] = pairwise_df['gene2'].astype("category")
    
    # Precompute a mapping from each gene to the row indices in the pairwise DataFrame where it appears.
    gene_to_pair_indices = {}
    for i, (gene_a, gene_b) in enumerate(zip(pairwise_df["gene1"], pairwise_df["gene2"])):
        gene_to_pair_indices.setdefault(gene_a, []).append(i)
        gene_to_pair_indices.setdefault(gene_b, []).append(i)
    
    # Initialize AUC scores (one for each complex) with NaNs.
    auc_scores = np.full(len(terms), np.nan)
    
    # Loop over each gene complex
    for idx, row in tqdm(terms.iterrows()):
        gene_set = set(row.used_genes)
        if len(gene_set) < config["min_complex_size_for_percomplex"]:  # Skip small complexes
            continue

        # Collect all row indices in the pairwise data where either gene belongs to the complex.
        candidate_indices = bitarray(len(pairwise_df))
        for gene in gene_set:
            if gene in gene_to_pair_indices:
                candidate_indices[gene_to_pair_indices[gene]] = True
        
        if not candidate_indices.any():
            continue
        
        # Select only the relevant pairwise comparisons.
        selected_rows = np.unpackbits(candidate_indices).view(bool)[:len(pairwise_df)]
        sub_df = pairwise_df.iloc[selected_rows]
        # A prediction is 1 if both genes in the pair are in the complex; otherwise 0.
        predictions = (sub_df["gene1"].isin(gene_set) & sub_df["gene2"].isin(gene_set)).astype(int)
        
        if predictions.sum() == 0:
            continue

        # Compute cumulative true positives and derive precision and recall.
        true_positive_cumsum = predictions.cumsum()
        precision = true_positive_cumsum / (np.arange(len(predictions)) + 1)
        recall = true_positive_cumsum / true_positive_cumsum.iloc[-1]
        
        if len(recall) < 2 or recall.iloc[-1] == 0:
            continue

        auc_scores[idx] = metrics.auc(recall, precision)
    
    # Add the computed AUC scores to the terms DataFrame.
    terms["auc_score"] = auc_scores
    terms.drop(columns=["ID", "list", "set", "hash"], inplace=True)
    dsave(terms, "pra_percomplex", dataset_name)
    log.done(f"Per-complex PRA completed.")
    return terms



def pra_percomplex(dataset_name, matrix, is_corr=False):
    log.started(f"*** Per-complex PRA started for {dataset_name} ***")
    config = dload("config")
    terms = dload("tmp", "terms").reset_index(drop=True)
    genes_present = dload("tmp", "genes_present_in_terms")
    sorting = dload("input", "sorting")
    sort_order = sorting.get(dataset_name, "high")

    if not is_corr:
        matrix = perform_corr(matrix, "numpy")
    matrix = filter_matrix_by_genes(matrix, genes_present)
    log.info(f"Matrix shape: {matrix.shape}")
    df = binary(matrix)
    log.info(f"Pair-wise shape: {df.shape}")

    df = quick_sort(df, ascending=(sort_order == "low"))
    df = df.reset_index(drop=True)

    # Precompute gene → row indices
    gene_to_rows = {}
    for i, (g1, g2) in enumerate(zip(df["gene1"], df["gene2"])):
        gene_to_rows.setdefault(g1, []).append(i)
        gene_to_rows.setdefault(g2, []).append(i)

    aucs = np.full(len(terms), np.nan)
    N = len(df)

    for idx, row in tqdm(terms.iterrows()):
        genes = set(row.used_genes)
        if len(genes) < config["min_complex_size_for_percomplex"]:  # Skip small complexes
            continue

        # Get all row indices where either gene is in the complex
        candidate_idxs = set()
        for g in genes:
            candidate_idxs.update(gene_to_rows.get(g, []))
        candidate_idxs = sorted(candidate_idxs)

        if not candidate_idxs:
            continue

        # Use only relevant rows for prediction
        sub = df.loc[candidate_idxs]
        preds = (sub["gene1"].isin(genes) & sub["gene2"].isin(genes)).astype(int)

        if preds.sum() == 0:
            continue

        tp = preds.cumsum()
        prec = tp / (np.arange(len(preds)) + 1)
        recall = tp / tp.iloc[-1]

        if len(recall) < 2 or recall.iloc[-1] == 0:
            continue

        aucs[idx] = metrics.auc(recall, prec)

    terms["auc_score"] = aucs
    terms.drop(columns=["ID", "list", "set", "hash"], inplace=True)
    dsave(terms, "pra_percomplex", dataset_name)
    log.done(f"Per-complex PRA completed.")
    return terms


def perform_corr(df, corr_func):
    if corr_func not in {"numpy", "pandas"}:
        raise ValueError("corr_func must be 'numpy' or 'pandas'")

    log.started(f"Performing correlation using '{corr_func}' method.")
    
    if corr_func == "numpy":
        # Compute correlation matrix and diagonal NaN in one pass
        corr_values = np.corrcoef(df.values)
        np.fill_diagonal(corr_values, np.nan)
        corr = pd.DataFrame(corr_values, index=df.index, columns=df.index)
        log.done("Correlation.")
        return corr
    else:
        # Compute correlations and modify diagonal in-place
        corr = df.T.corr()
        np.fill_diagonal(corr.values, np.nan)
        return corr


def is_symmetric(df):
    return np.allclose(df, df.T, equal_nan=True)


def binary(corr):
    log.started("Converting correlation matrix to pair-wise format.")
    if is_symmetric(corr):
        corr = convert_full_to_half_matrix(corr)
    
    stack = corr.stack().rename_axis(index=['gene1', 'gene2']).\
            reset_index().rename(columns={0: 'score'})
    if has_mirror_of_first_pair(stack):
        log.info("Mirror pairs detected. Dropping them to ensure unique gene pairs.")
        stack = drop_mirror_pairs(stack)
    log.done("Pair-wise conversion.")
    return stack


def has_mirror_of_first_pair(df):
    g1, g2 = df.iloc[0]['gene1'], df.iloc[0]['gene2']
    mirror_exists = ((df['gene1'] == g2) & (df['gene2'] == g1)).iloc[1:].any()
    return mirror_exists


def convert_full_to_half_matrix(df):
    if not is_symmetric(df):
        raise ValueError("Matrix must be symmetric to convert to half matrix.")

    log.started("Converting full correlation matrix to upper triangle (half-matrix) format.")
    arr = df.values.copy()
    arr[np.tril_indices_from(arr)] = np.nan  # zero-based lower triangle + diagonal → NaN
    log.done("Matrix conversion.")
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def drop_mirror_pairs(df):
    log.started("Dropping mirror pairs to ensure unique gene pairs (Optimized).")
    gene_pairs = np.sort(df[["gene1", "gene2"]].to_numpy(), axis=1)
    df.loc[:, ["gene1", "gene2"]] = gene_pairs
    df = df.loc[~df.duplicated(subset=["gene1", "gene2"], keep="first")]
    log.done("Mirror pairs are dropped.")
    return df


def quick_sort(df, ascending=False):
    log.started(f"Pair-wise matrix is sorting based on the 'score' column: ascending:{ascending}")
    order = 1 if ascending else -1
    sorted_df = df.iloc[np.argsort(order * df["score"].values)].reset_index(drop=True)
    log.done("Pair-wise matrix sorting.")
    return sorted_df


def complex_contributions(name):
    log.started(f"Computing complex contributions for dataset: {name}")

    # Load data
    pra = dload("pra", name)
    terms = dload("tmp", "terms")
    d = pra.query('prediction == 1').drop(columns=['gene1', 'gene2'])
    
    # Precompute thresholds as a numpy array
    thresholds = np.array([round(i, 2) for i in np.arange(1, 0.0001, -0.025)])
    
    # Group by complex_id and sort precision values in ascending order
    groups = d.groupby('complex_id')['precision'].apply(lambda x: np.sort(x.values))
    
    # Compute counts for each complex using vectorized operations
    results = {}
    for cid, precisions in groups.items():
        indices = np.searchsorted(precisions, thresholds, side='left')
        counts = len(precisions) - indices
        results[cid] = counts.tolist()
    
    # Handle complexes with no entries (initialize with 0s)
    all_cids = terms['ID'].tolist()
    for cid in all_cids:
        if cid not in results:
            results[cid] = [0] * len(thresholds)
    
    # Create DataFrame and format results
    r = pd.DataFrame(results, index=thresholds).T
    t = terms[['ID', 'Name']].set_index('ID')
    r['Name'] = r.index.map(t['Name'])
    r = r[list(reversed(r.columns))]  # Reverse columns to match original order
    r = r.reset_index(drop=True)
    
    dsave(r, "complex_contributions", name)
    log.done(f"Complex contributions computation completed for dataset: {name}")
    return r


### OLD FUNCTIONS



# def generate_gene_pair_hashes(terms):
#     log.info(f"Gene pair hashes generating.")
#     hash_table = {}
#     for term_id, genes in zip(terms["ID"], terms["Genes"].str.split(";")):
#         for gene_pair in itertools.permutations(genes, 2):
#             hash_table[hash(gene_pair)] = term_id

#     log.info(f"Generated {len(hash_table)} gene pair hashes.")
#     return hash_table


# def binary(corr):
#     log.info(f"Converting correlation matrix to binary format.")
#     if is_symmetric(corr):
#         corr = convert_full_to_half_matrix(corr)

#     stack = corr.stack().reset_index()
#     stack.columns = ["gene1", "gene2", "score"]
#     if has_mirror_of_first_pair(stack):
#         stack = drop_mirror_pairs(stack)
#     return stack



# def complex_contributions(name):
#     log.info(f"Computing complex contributions for dataset: {name}")

#     pra = dload("pra", name)
#     terms = dload("tmp", "terms")
#     d = pra.query('prediction == 1').drop(columns=['gene1', 'gene2'])
#     results = {}
#     thresholds = [round(i, 2) for i in np.arange(1, 0.0001, -0.025)]
#     for cid in terms.ID.to_list():
#         arr = []
#         for threshold in thresholds:
#             r = d[d.complex_id == cid].query('precision >= @threshold')
#             arr.append(r.shape[0])
#         results[cid] = arr

#     r = pd.DataFrame(results, index=thresholds).T
#     t = terms[['ID', 'Name']].set_index('ID')
#     r['Name'] = r.index.map(t.Name)
#     r = r[list(reversed(list(r.columns)))]
#     r = r.reset_index(drop=True)
#     dsave(r, "complex_contributions", name)
#     log.info(f"Complex contributions computation completed for dataset: {name}")
#     return r



# def drop_mirror_pairs(df):
#     log.info("Dropping mirror pairs to ensure unique gene pairs.")
#     df[["gene1", "gene2"]] = np.sort(df[["gene1", "gene2"]].values, axis=1)
#     return df.drop_duplicates(subset=["gene1", "gene2"])


# def perform_corr(df, corr_func):
#     if corr_func not in {"numpy", "pandas"}:
#         raise ValueError("corr_func must be 'numpy' or 'pandas'")

#     log.info(f"Performing correlation using '{corr_func}' method.")
#     corr = (
#         pd.DataFrame(np.corrcoef(df.values), index=df.index, columns=df.index)
#         if corr_func == "numpy"
#         else df.T.corr()
#     )
#     np.fill_diagonal(corr.values, np.nan)
#     return corr




# def compute_pra_without_complexes(dataset_name, matrix, is_corr=False, remove_complexes=[]):
#     log.info("PRA computation after removing complexes started.")
#     terms = dload("tmp", "terms")
#     common_genes = dload("tmp", "common_genes")
#     log.info(f"Removing {len(remove_complexes)} specified complexes.")
#     terms = terms[~terms.Name.isin(remove_complexes)]
#     terms["hash"] = terms.used_genes.apply(lambda x: [hash(i) for i in x])
#     hash_table = generate_gene_pair_hashes(terms)
#     genes_present_in_terms = set(terms["list"].explode().unique()) & set(common_genes)
#     dsave(genes_present_in_terms, "tmp", "removed_pra_genes")
#     dsave(hash_table, "tmp", "removed_pra_hash_table")
#     if not is_corr: matrix = perform_corr(matrix, "numpy")
#     matrix = filter_matrix_by_genes(matrix, genes_present_in_terms)
#     stack = binary(matrix)
#     annotated = check_gene_pairs_in_gold_standard(stack, hash_table)
#     ann_sorted = quick_sort(annotated)
#     pra = compute_pra(ann_sorted)
#     dsave(pra, "removed_pra", dataset_name)
#     log.info("PRA computation after removing complexes completed.")
#     return pra



# def convert_full_to_half_matrix(df):
#     if not is_symmetric(df):
#         raise ValueError("Matrix must be symmetric to convert to half matrix.")
#     log.info("Converting full correlation matrix to upper triangle (half-matrix) format.")
#     d = df.copy()
#     mask = np.tril_indices_from(d, k=0)  # includes diagonal
#     d.values[mask] = np.nan
#     return d



# def percomplex_pra(dataset_name, matrix, is_corr=False):
#     log.info(f"Per-complex PRA computation started for {dataset_name}.")
#     terms = dload("tmp", "terms")
#     genes_present_in_terms = dload("tmp", "genes_present_in_terms")
#     sorting_prefs = dload("input", "sorting")
#     sort_order = sorting_prefs.get(dataset_name, "high")  # Default to high
#     if not is_corr:  matrix = perform_corr(matrix, "numpy")
#     matrix = filter_matrix_by_genes(matrix, genes_present_in_terms)
#     matrix_binary = binary(matrix)
#     if sort_order == "low":
#         matrix_binary_sorted = quick_sort(matrix_binary, ascending=True)
#     else:
#         matrix_binary_sorted = quick_sort(matrix_binary)

#     log.info("Hashing gene pairs for efficient lookup.")
#     matrix_binary_sorted[["gene1_hash", "gene2_hash"]] = matrix_binary_sorted[["gene1", "gene2"]].applymap(hash)
#     stack = matrix_binary_sorted.copy()
#     terms["auc_score"] = terms.progress_apply(lambda row: compute_auc_for_complex(stack, row), axis=1)
#     terms.drop(columns=["ID", "list", "set", "hash"], inplace=True)
#     dsave(terms, "pra_percomplex", dataset_name)
#     log.info(f"Per-complex PRA computation completed for {dataset_name} (Sorting: {sort_order}).")
#     return terms


# def compute_auc_for_complex(df, row):
#     config = dload("config")
#     min_complex_size = config.get("min_complex_size_for_percomplex", 3)  # Default to 3 if not specified
#     if row.n_used_genes < min_complex_size:
#         return np.nan
#     # log.info(f"Computing AUC for complex: {row.Name} with {row.n_used_genes} genes.")
#     mask = df["gene1_hash"].isin(row.hash) | df["gene2_hash"].isin(row.hash)
#     df = df.loc[mask].copy()
#     if df.empty:
#         return np.nan
#     df["prediction"] = df[["gene1_hash", "gene2_hash"]].isin(row.hash).all(axis=1).astype(int)
#     df["tp"] = df["prediction"].cumsum()
#     if df.iloc[-1]["tp"] == 0:
#         return np.nan
#     df.reset_index(drop=True, inplace=True)
#     df["precision"] = df["tp"] / (df.index + 1)
#     df["recall"] = df["tp"] / df["tp"].iloc[-1]
#     auc_score = metrics.auc(df["recall"], df["precision"])
#     # log.info(f"AUC computed for complex: {row.Name}. Score: {auc_score:.4f}")
#     return auc_score







# def compute_pra(df):
#     log.info("Calculating precision-recall and AUC score.")
#     if df.empty:
#         log.warning("Empty DataFrame encountered in compute_pra. Returning empty DataFrame.")
#         return df  
#     df["tp"] = df["prediction"].cumsum()
#     df.reset_index(drop=True, inplace=True)
#     df["precision"] = df["tp"] / (df.index + 1)
#     df["recall"] = df["tp"] / df["tp"].iloc[-1]
#     log.info("DONE: Calculating precision-recall AUC score.")
#     return df


# def pra(dataset_name, matrix, is_corr=False):
#     log.info(f"PRA computation started for {dataset_name}.")
#     genes_present_in_terms = dload("tmp", "genes_present_in_terms")
#     #terms_hash_table = dload("tmp", "terms_hash_table")
#     sorting_prefs = dload("input", "sorting")
#     sort_order = sorting_prefs.get(dataset_name, "high") 
#     if not is_corr: matrix = perform_corr(matrix, "numpy")
#     matrix = filter_matrix_by_genes(matrix, genes_present_in_terms)
#     stack = binary(matrix)

#     log.info("Checking gene pairs against the gold standard.")
#     gene_pairs = list(zip(stack["gene1"], stack["gene2"]))
#     hashed_pairs = [hash(pair) for pair in gene_pairs]
#     stack["complex_id"] = [terms_hash_table.get(h, 0) for h in hashed_pairs]
#     stack["prediction"] = [1 if h in terms_hash_table else 0 for h in hashed_pairs]

#     annotated = stack.copy()
#     if sort_order == "low":
#         ann_sorted = quick_sort(annotated, ascending=True) 
#     else:
#         ann_sorted = quick_sort(annotated) 

#     pra = compute_pra(ann_sorted)
#     pr_auc = metrics.auc(pra.recall, pra.precision)
#     dsave(pra, "pra", dataset_name)
#     dsave(pr_auc, "pr_auc", dataset_name)
#     log.info(f"PRA computation completed for {dataset_name} (Sorting: {sort_order}).")
#     return pra, pr_auc










