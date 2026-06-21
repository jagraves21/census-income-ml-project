# Submission Overview

This repository contains my submission for a machine learning take-home assignment.

It includes all source code, experiments, analysis notebooks, trained models, and the final written report.

## Repository Structure

### [data/](./data/)
All datasets and associated artifacts are organized here. Each dataset has its own subdirectory containing raw data, processed data, models, results, and figures.

- [Census/](./data/Census) — current dataset used in this project
  - [raw/](./data/Census/raw) &mdash; original provided dataset
  - [processed/](./data/Census/processed) &mdash; cleaned and transformed datasets
  - [figures/](./data/Census/figures) &mdash; figures generated during analysis
  - [models/](./data/Census/models) &mdash; serialized trained models (`.joblib`)
  - [results/](./data/Census/results) &mdash; csv outputs from evaluation and analysis

### [notebooks/](./notebooks)
End-to-end workflow covering exploration, preprocessing, modeling, and reporting.

1. [01-exploratory-data-analysis.ipynb](./notebooks/01-exploratory-data-analysis.ipynb)
2. [02-data-cleaning-and-preprocessing.ipynb](./notebooks/02-data-cleaning-and-preprocessing.ipynb)
3. [03-feature-importance-analysis.ipynb](./notebooks/03-feature-importance-analysis.ipynb)
4. [04-performance-estimation.ipynb](./notebooks/04-performance-estimation.ipynb)
5. [05-model-interpretation.ipynb](./notebooks/05-model-interpretation.ipynb)
6. [06-segmentation-analysis.ipynb](./notebooks/06-segmentation-analysis.ipynb)

### [src/](./src)
Custom Python packages shared across notebooks.

- [ml_utils/](./src/ml_utils) &mdash; evaluation, modeling, preprocessing, and visualization utilities
- [my_datasets/](./src/my_datasets) &mdash; dataset loading and management utilities

### [report/](./report)
Final written report and source files.

- [main.pdf](./report/main.pdf) &mdash; final report
- `*.tex` &mdash; LaTeX source files used to generate the report

### [requirements.txt](./requirements.txt)
Python dependencies required to run the project.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

The notebooks are largely independent, but [02-data-cleaning-and-preprocessing.ipynb](./notebooks/02-data-cleaning-and-preprocessing.ipynb) must be run before any subsequent notebooks. For reproducibility, it is recommended to execute the notebooks in the following order:

1. [01-exploratory-data-analysis.ipynb](./notebooks/01-exploratory-data-analysis.ipynb)
2. [02-data-cleaning-and-preprocessing.ipynb](./notebooks/02-data-cleaning-and-preprocessing.ipynb)
3. [03-feature-importance-analysis.ipynb](./notebooks/03-feature-importance-analysis.ipynb)
4. [04-performance-estimation.ipynb](./notebooks/04-performance-estimation.ipynb)
5. [05-model-interpretation.ipynb](./notebooks/05-model-interpretation.ipynb)
6. [06-segmentation-analysis.ipynb](./notebooks/06-segmentation-analysis.ipynb)

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.