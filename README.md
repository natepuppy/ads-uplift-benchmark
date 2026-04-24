# Benchmarking Uplift Models Under Realistic Confounding

**Paper:** "Benchmarking Uplift Models Under Realistic Confounding: A Controlled Study in Ad Incrementality Estimation"

---

## Overview

This repository contains all code to reproduce the experiments and figures in the paper. The core contribution is a **synthetic benchmark framework** for evaluating uplift/incrementality models under controlled confounding conditions — something impossible with real-world ad data, which lacks ground-truth treatment effects (CATE).

## Project Structure

```
research/
├── data/
│   ├── dgp/             # Data generating processes (DGPs) with known CATE
│   └── criteo/          # Criteo Uplift Dataset download + preprocessing
├── models/              # S, T, X, R-Learners + Uplift Random Forest
├── metrics/             # PEHE, Qini, AUUC, calibration evaluation
├── experiments/         # Experiment runner + YAML configs
├── analysis/            # Figure and LaTeX table generation
├── paper/               # LaTeX source
└── notebooks/           # Exploratory analysis
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full benchmark
python experiments/run_benchmark.py --config experiments/configs/benchmark.yaml

# Generate all paper figures
python analysis/figures.py --results_dir results/

# (Optional) Download Criteo data
bash data/criteo/download.sh
```

## Benchmark Design

We define a family of Data Generating Processes (DGPs) parametrized by:

| Parameter | Description | Range |
|-----------|-------------|-------|
| `alpha` | Confounding strength (selection bias in treatment) | 0.0 – 2.0 |
| `tau_type` | CATE structure | constant, linear, nonlinear, heterogeneous |
| `rho` | Base conversion rate | 0.01 – 0.20 |
| `n` | Sample size | 5k – 100k |

## Models

| Model | Reference |
|-------|-----------|
| S-Learner | Hill (2011) |
| T-Learner | Künzel et al. (2019) |
| X-Learner | Künzel et al. (2019) |
| R-Learner | Nie & Wager (2021) |
| Uplift RF | Rzepakowski & Jaroszewicz (2012) |

Each meta-learner tested with three base learners: Logistic Regression, Random Forest, XGBoost.

## Metrics

- **PEHE** — Precision in Estimation of Heterogeneous Effects (requires synthetic ground truth)
- **Qini coefficient** — ranking-based uplift metric
- **AUUC** — Area Under Uplift Curve
- **Policy value** — expected outcome under top-k% treatment policy
