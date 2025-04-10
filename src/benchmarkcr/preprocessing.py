import os
import pandas as pd
from .utils import dsave, dload
from tqdm import tqdm
from .logging_config import log
from importlib import resources
tqdm.pandas()




def get_example_data_path(filename: str):
    return resources.files("benchmarkcr.data").joinpath("dataset").joinpath(filename)



def load_datasets(files):
    preprocessing = dload("config")["preprocessing"]
    data_dict= {}     

    for filename, meta in files.items():
        filepath = meta["path"]
        ext = os.path.splitext(filepath)[1]

        if ext == ".csv":
            df = pd.read_csv(filepath, index_col=0)
        elif ext == ".xlsx":
            df = pd.read_excel(filepath, index_col=0)
        else:
            df = pd.read_parquet(filepath)

        df.index = df.index.str.split().str[0]

        if preprocessing['normalize']: 
            df = normalize_data(df)

        if preprocessing['drop_na']:
            log.info(f"{filename}: Dropping missing values.")
            df = df.dropna(how="any")
        
        if preprocessing['fill_na'] == 'mean':
            log.info(f"{filename}: Filling missing values with column mean.")
            df = df.T.fillna(df.mean(axis=1)).T
        elif preprocessing['fill_na'] == 'zero':
            log.info(f"{filename}: Filling missing values with zeros.")
            df = df.fillna(0)

        data_dict[filename] = df

    common_genes = get_common_genes(data_dict)
    dsave({"datasets": data_dict, "sorting": {k: v["sort"] for k, v in files.items()}}, "input")
    log.done(f"Datasets loaded.")
    return data_dict, common_genes  


def normalize_data(df):
    log.started("Normalization")
    normalized = (df - df.mean()) / df.std(ddof=0)
    log.done("Normalization")
    return normalized
    

def get_common_genes(datasets):
    log.started("Finding common genes across datasets.")
    gene_sets = [set(df.index) for df in datasets.values()]
    common_genes = set.intersection(*gene_sets)
    log.done(f"Common genes found: {len(common_genes)}")
    dsave(common_genes, "tmp", "common_genes")
    return common_genes


def filter_matrix_by_genes(matrix, genes_present_in_terms):
    log.started("Filtering matrix using genes present in terms.")
    genes = matrix.index.intersection(genes_present_in_terms)
    matrix = matrix.loc[genes, genes]
    log.done(f"Filtering matrix: {matrix.shape}")
    return matrix.loc[genes, genes]
    

def load_gold_standard():
    
    config = dload("config") 
    common_genes = dload("tmp", "common_genes")

    log.started(f"Loading gold standard: {config['gold_standard']}, Min complex size: {config['min_complex_size']}, Jaccard filtering: {config['jaccard']}")
    if not common_genes:
        raise ValueError("Common genes not found.")

    # Define gold standard file paths
    gold_standard_files = {
        "CORUM": "gold_standard/CORUM.p",
        "GOBP": "gold_standard/GOBP.p",
        "PATHWAY": "gold_standard/PATHWAY.p"
    }
    filename = gold_standard_files[config["gold_standard"]]
    filename_path = resources.files("benchmarkcr.data").joinpath(filename)
    if not filename_path:
        raise ValueError("Invalid Gold Standard type.")

    terms = pd.read_parquet(filename_path)  # type: ignore
    common_genes_set = set(common_genes)
    terms["used_genes"] = terms["Genes"].apply(lambda x: list(set(x.split(";")) & common_genes_set))
    terms["n_used_genes"] = terms["used_genes"].apply(len)
    log.info(f"Applying min_complex_size filtering: {config['min_complex_size']}")
    terms = terms[terms["n_used_genes"] >= config['min_complex_size']]
    terms["hash"] = terms["used_genes"].apply(lambda x: [hash(i) for i in x])
    if config['jaccard']:
        log.info("Applying Jaccard filtering. Remove terms with identical gene sets.")
        terms = filter_duplicate_terms(terms)

    # Compute hash values for final terms
    # terms["hash"] = terms["used_genes"].apply(lambda x: [hash(i) for i in x])
    # terms_hash_table = generate_gene_pair_hashes(terms)
    # log.info(f"Generating gene pair hashes.")
    # hash_table = {}
    # for term_id, genes in zip(terms["ID"], terms["Genes"].str.split(";")):
    #     for gene_pair in itertools.permutations(genes, 2):
    #         hash_table[hash(gene_pair)] = term_id
    #dsave(hash_table, "tmp", "terms_hash_table")
    #log.info(f"Generated {len(hash_table)} gene pair hashes.")

    genes_present_in_terms = list(set(terms["used_genes"].explode().unique()) & common_genes_set)
    dsave(terms, "tmp", "terms")
    dsave(genes_present_in_terms, "tmp", "genes_present_in_terms")
    log.done("Gold standard loading completed.")
    terms.reset_index(drop=True, inplace=True)
    return terms, genes_present_in_terms


def filter_duplicate_terms(terms):
    log.started("Filtering duplicate terms using optimized method.")
    
    # Precompute frozen gene sets and hash them
    terms = terms.copy()
    terms["gene_set"] = terms["used_genes"].map(lambda x: frozenset(x))
    
    # Group by identical gene sets
    grouped = terms.groupby("gene_set", sort=False)
    
    # Identify duplicate clusters (groups with >1 term)
    duplicate_clusters = []
    for _, group in grouped:
        if len(group) > 1:
            duplicate_clusters.append(group["ID"].values)
    
    # Determine which IDs to keep (smallest ID in each duplicate cluster)
    keep_ids = set(terms["ID"])
    for cluster in duplicate_clusters:
        sorted_ids = sorted(cluster)
        keep_ids.difference_update(sorted_ids[1:])  # Remove all but smallest ID
    
    # Filter and clean up
    filtered = terms[terms["ID"].isin(keep_ids)].copy()
    filtered.drop(columns=["gene_set"], inplace=True)
    
    log.done(f"{len(terms) - len(filtered)} terms removed due to identical gene sets.")
    return filtered
