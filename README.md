# benchmarkCR

🧬 **benchmarkCR** is a benchmarking toolkit for evaluating CRISPR screen results against biological gold standards. It provides precision-recall analysis using reference gene sets from CORUM protein complexes, Gene Ontology Biological Processes (GO-BP), KEGG pathways, and other curated resources. The toolkit computes gene-level and complex-level performance metrics, helping researchers systematically assess the biological relevance and resolution of their CRISPR screening data.


---

## 🔧 Features

- Precision-recall curve generation for ranked gene lists

- Evaluation using CORUM complexes, GO terms, pathways

- Complex-level resolution analysis and visualization

- Easy integration into CRISPR screen workflows

---

## 📦 Installation

Suggested to use Python version `3.10` with `virtual env`.

Create `venv`

```bash
conda create -n p310 python=3.10
conda activate p310
pip install uv
```

Install benchmarkCR via pip

``` bash
uv pip install benchmarkcr
```

or 

```bash
pip install benchmarkcr
```

or Install benchmarkCR via git (to develop package in local)

```bash
git clone https://github.com/billmannlab/benchmarkCR.git
cd benchmarkcr
uv pip install -e .
```



---

## 🚀 Quickstart

```python
import benchmarkcr

inputs = {
    "Melanoma (63 Screens)": {
        "path": benchmarkcr.get_example_data_path("melanoma_cell_lines_500_genes.csv"), 
        "sort": "high"
    },
    "Liver (24 Screens)": {
        "path": benchmarkcr.get_example_data_path("liver_cell_lines_500_genes.csv"), 
        "sort": "high"
    },
    "Neuroblastoma (37 Screens)": {
        "path": benchmarkcr.get_example_data_path("neuroblastoma_cell_lines_500_genes.csv"), 
        "sort": "high"
    },
}

config = {
    "min_complex_size": 3,
    "min_complex_size_for_percomplex": 3,
    "output_folder": "output",
    "gold_standard": "CORUM",
    "color_map": "RdYlBu",
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

# Initialize logger, config, and output folder
benchmarkcr.initialize(config)

# Load datasets and gold standard terms
data, _ = benchmarkcr.load_datasets(inputs)
terms, genes_in_terms = benchmarkcr.load_gold_standard()

# Run analysis
for name, dataset in data.items():
    df, pr_auc = benchmarkcr.pra(name, dataset)
    pc = benchmarkcr.pra_percomplex(name, dataset)
    #fpc = benchmarkcr.fast_pra_percomplex(name, dataset) # not tested yet.
    cc = benchmarkcr.complex_contributions(name)

# Generate plots
benchmarkcr.plot_auc_scores()
benchmarkcr.plot_precision_recall_curve()
benchmarkcr.plot_percomplex_scatter()
benchmarkcr.plot_complex_contributions()
benchmarkcr.plot_significant_complexes()
```

---

## 📂 Examples

- [examples/basic_usage.py](examples/basic_usage.py)
- [notebooks/demo.ipynb](notebooks/demo.ipynb)

---

## 📃 License

MIT
