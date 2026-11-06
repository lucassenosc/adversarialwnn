# Network intrusion detection experiments

This directory contains the intrusion detection experiments for *Binarization as an Attack Surface in Weightless Neural Networks*. The paths below are relative to `ids/`.

## Scope

The paper's IDS experiments evaluate Standard WiSARD and Bloom WiSARD under binary and multiclass classification. The paper-reported IDS perturbations are:

- **FGSM** (`L∞`, `ε = 0.3`), generated on a differentiable surrogate and transferred to the WNN;
- **Random L∞** (`ε = 0.3`);
- **Random L2** (`ε = 3.0`).

The Random L∞ and Random L2 methods are model-agnostic sensitivity probes.

The repository also contains auxiliary Decision Tree and Random Forest notebooks. These notebooks are retained as exploratory baselines, but they are **not required to reproduce the IDS results reported in the current paper**.

## Repository Structure

```text
.
├── analysis/
│   ├── aggregate_reports.ipynb
│   └── curva_roc.ipynb
│
├── data/                         # Not versioned; downloaded separately
│   ├── botiot/
│   ├── cicids2017/
│   └── unsw_nb15/
│
├── experiments/
│   ├── baselines/
│   │   ├── decision_tree.ipynb
│   │   └── random_forest.ipynb
│   │
│   └── wisard/
│       ├── bloom_wisard.ipynb
│       └── standart_wisard.ipynb
│
├── libs/
│   ├── bloom_filter.py
│   └── wisard.py
│
├── preprocessing/
│   ├── prepare_botiot.ipynb
│   └── prepare_cicids2017.ipynb
│
├── results/
│   ├── aggregated_reports/
│   ├── curves_data/
│   └── reports/
│
├── README.md
├── requirements.txt
└── requirements-lock.txt
```

`__pycache__/` and virtual-environment directories are local artifacts and are intentionally excluded from version control.

## Main Components

### WiSARD experiments

- `experiments/wisard/standart_wisard.ipynb`  
  Standard WiSARD experimental pipeline.

- `experiments/wisard/bloom_wisard.ipynb`  
  Bloom WiSARD experimental pipeline.

Both notebooks handle dataset loading, preprocessing, perturbation generation, binarization, model evaluation, metrics, and result export.

### Data preprocessing

- `preprocessing/prepare_botiot.ipynb`  
  Builds the Bot-IoT training and test files from the reduced Bot-IoT CSV files.

- `preprocessing/prepare_cicids2017.ipynb`  
  Builds the CICIDS2017 training and test files from the cleaned CICIDS2017 dataset.

The UNSW-NB15 train/test files are expected to already be present in `data/unsw_nb15/`.

### Analysis

- `analysis/aggregate_reports.ipynb`  
  Reads individual CSV files from `results/reports/` and writes grouped outputs to `results/aggregated_reports/`.

- `analysis/curva_roc.ipynb`  
  Reads the saved curve data from `results/curves_data/` and generates comparison curves.

### Local libraries

- `libs/wisard.py`  
  Bloom WiSARD implementation used by the Bloom WiSARD notebook.

- `libs/bloom_filter.py`  
  Bloom-filter implementation used by `libs/wisard.py`.

## Reference Environment

The validated IDS reference environment uses:

```text
Python 3.13.12
```

Two dependency files are provided:

- `requirements.txt`: direct project dependencies;
- `requirements-lock.txt`: complete dependency snapshot generated from a clean, validated reproduction environment.

For the closest reproduction of the validated environment, use `requirements-lock.txt`.

## 1. Obtain the Repository

```bash
git clone https://github.com/lucassenosc/adversarialwnn.git
cd adversarialwnn/ids
```

## 2. Create a Virtual Environment

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

The expected Python version is:

```text
Python 3.13.12
```

### Linux / macOS

If Python 3.13 is installed:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python --version
```

The lock file was validated in the Windows reference environment. On another operating system, `requirements.txt` may be preferable if a platform-specific locked dependency cannot be resolved.

## 3. Install Dependencies

Upgrade `pip` first:

```bash
python -m pip install --upgrade pip
```

For the validated locked environment:

```bash
python -m pip install -r requirements-lock.txt
```

Alternatively, install only the declared direct dependencies:

```bash
python -m pip install -r requirements.txt
```

Check the resulting environment:

```bash
python -m pip check
```

A healthy environment should report:

```text
No broken requirements found.
```

## 4. Download the Datasets

Large datasets are intentionally excluded from Git because of their size.
If you downloaded the IDS datasets with the previous layout, move the local `data/` folder from the checkout root to `ids/data/`.

Download the dataset files from:

[Download the datasets from Google Drive](https://drive.google.com/drive/folders/116HwuX547KEJ8spp6mhnufUL7Yq0Jg0i?usp=sharing)

Place the files under:

```text
data/
├── botiot/
├── cicids2017/
└── unsw_nb15/
```

The experimental notebooks expect the prepared splits to follow this organization:

```text
data/
├── botiot/
│   ├── BotIoT_training-set.csv
│   └── BotIoT_testing-set.csv
│
├── cicids2017/
│   ├── CICIDS_training-set.csv
│   └── CICIDS_testing-set.csv
│
└── unsw_nb15/
    ├── UNSW_NB15_training-set.csv
    └── UNSW_NB15_testing-set.csv
```

If the prepared Bot-IoT or CICIDS2017 splits are not available, generate them with the preprocessing notebooks described below.

## 5. Prepare the Data if Necessary

### Bot-IoT

Place the reduced source files in:

```text
data/botiot/
```

Then run:

```text
preprocessing/prepare_botiot.ipynb
```

The notebook generates:

```text
data/botiot/BotIoT_training-set.csv
data/botiot/BotIoT_testing-set.csv
```

### CICIDS2017

Place:

```text
data/cicids2017/cicids2017_cleaned.csv
```

and run:

```text
preprocessing/prepare_cicids2017.ipynb
```

The notebook generates:

```text
data/cicids2017/CICIDS_training-set.csv
data/cicids2017/CICIDS_testing-set.csv
```

## 6. Run the WiSARD Experiments

Open either notebook directly in Jupyter or VS Code:

```text
experiments/wisard/standart_wisard.ipynb
experiments/wisard/bloom_wisard.ipynb
```

The notebooks locate `data/`, `libs/`, and `results/` inside `ids/` when opened from the repository root, `ids/`, or their own subdirectories.

Select the project's `.venv` as the notebook kernel. Before using **Run All**, set the experiment control panel in each notebook to the paper's IDS scope:

```python
DATASETS_TO_RUN = ["Bot-IoT", "UNSW-NB15", "CICIDS"]
RUN_BINARY = True
RUN_MULTICLASS = True
ENCODING_TYPES = ["linear", "gaussian"]
ATTACKS_TO_RUN = ["FGSM", "RANDOM_LINF", "RANDOM_L2"]
```

Use the same selection when computing paper summaries from the archived CSVs. The repository also contains exploratory runs and configurations beyond this selection.

## 7. Results

Individual experiment reports are written to:

```text
results/reports/
```

Curve data used by the analysis notebook are written to:

```text
results/curves_data/
```

Aggregated reports are written to:

```text
results/aggregated_reports/
```

The result directories are intentionally **not ignored** by `.gitignore`, allowing experimental outputs selected for the reproducibility artifact to be versioned.

## 8. Aggregate Reports

After generating individual experiment CSVs, run:

```text
analysis/aggregate_reports.ipynb
```

The aggregation flow is:

```text
results/reports/
        ↓
aggregate_reports.ipynb
        ↓
results/aggregated_reports/
```

The aggregator groups experiment files according to the filename convention used by the experimental notebooks.

## 9. Generate ROC / PR Comparisons

Run:

```text
analysis/curva_roc.ipynb
```

It reads the `.npz` files stored in:

```text
results/curves_data/
```

and generates comparison plots from the stored clean and perturbed predictions/scores.

## 10. Auxiliary Baselines

The repository currently retains:

```text
experiments/baselines/decision_tree.ipynb
experiments/baselines/random_forest.ipynb
```

These are auxiliary exploratory baselines. They are not required for reproduction of the IDS results reported in the current paper.

## Reproducibility Checklist

A reviewer reproducing the IDS experiments should:

1. Clone the public repository.
2. Use Python 3.13.12.
3. Create and activate a fresh virtual environment.
4. Install `requirements-lock.txt` for the validated environment.
5. Run `python -m pip check`.
6. Download the three IDS datasets and place them under `data/`.
7. Generate the Bot-IoT/CICIDS2017 prepared splits if necessary.
8. Select the new virtual environment as the Jupyter kernel.
9. Apply the IDS settings above and run the Standard WiSARD and Bloom WiSARD notebooks.
10. Aggregate generated CSV reports with `analysis/aggregate_reports.ipynb`.
11. Generate comparison curves with `analysis/curva_roc.ipynb` when needed.

## Citation

If this repository is used in academic work, please cite the associated paper:

> *Binarization as an Attack Surface in Weightless Neural Networks.*
