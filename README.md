# Generalized confounder adjustment for testing and estimation (GCATE)

This repository contains the code for reproducing simulation and real data analysis results of the paper "Simultaneous inference for generalized linear models with unmeasured confounders".


## Files

### Scripts

- ex1: Simulation with Poisson DGP with sample splitting
    - `ex1_generate_data.py`: Generate simulated data.
    - `ex1_run_gcate.py`: Run GCATE.
- ex2: Simulation with Poisson DGP without sample splitting
    - `ex2_generate_data.py`: Generate simulated data.
    - `ex2_run_gcate.py`: Run GCATE.
    - `ex2_run_glm.py`: Run GLM oracle and GLM naive.
    - `ex2_run_cate.R`: Run CATE.
- ex3: Simulation with Splatter simulator
    - `ex3_generate_data.py`: Generate simulated data.
    - `ex3_run_gcate.py`: Run GCATE.
    - `ex3_run_glm.py`: Run GLM naive.
    - `ex3_run_cate.R`: Run CATE.
- ex4: Lupus data
    - `ex4_preprocess_lupus.py`: preprocess the lupus data    
    - `ex4_run_gcate.py`: Run GCATE.
    - `ex4_run_glm.py`: Run GLM oracle and GLM naive.
    - `ex4_run_cate.R`: Run CATE.
    - `ex4_GO.R`: gene ontology analysis

### Jupyter notebooks:
- `Plot_simu.ipynb`: Reproduce the figures and tables for simulation studies.
- `Plot_lupus.ipynb`: Reproduce the figures and tables for the lupus data analysis.


## Requirements

The following packages are required for the reproducibility workflow.


### Python packages

Package | Version
---|---
anndata | 0.9.2 
cvxpy | 1.1.18 
h5py | 3.1.0 
joblib | 1.1.0 
jupyter | 1.0.0
matplotlib | 3.4.3
numba | 0.54.1 
numpy | 1.22.0 
pandas | 1.3.3 
python | 3.8.12
scanpy | 1.9.3 
scikit-learn | 1.1.2 
scipy | 1.10.1 
statsmodels | 0.13.5 
tqdm | 4.62.3

### R packages

Package | Version
---|---
AnnotationDbi | 1.56.2
cate | 1.1.1 
clusterProfiler | 4.2.2
org.Hs.eg.db | 3.14.0
qvalue | 2.26 
R | 3.8.2
reticulate | 1.31 
rrvgo | 1.6.0
tidyverse | 1.3.1



## Reproducibility workflow

For simulation studies, the workflow is as follows:

- Run script `ex1_generate_data.py` to generate simulated data, which will be stored in the folder `/data/ex1/`. The second and the third experiments can be generated by running `ex2_generate_data.py` and `ex3_generate_data.py`, respectively.
- Run scripts of individual methods, and the results will be stored in the folder `result/`.
- Use `Plot_simu.ipynb` to reproduce the figures (Figures 2-6 and F1-F2) and table (Table G2) based on the previous results.

For real data analysis, the workflow is as follows:

- Obtain the h5ad file of the lupus data from the authors of the original paper and store it in the folder `data/lupus/GSE174188_CLUES1_adjusted.h5ad`.
- Run `ex4_preprocess_lupus.py` to preprocess the lupus data.
- Run scripts of individual methods, and the results will be stored in the folder `result/lupus/`.
- Use `Plot_lupus.ipynb` to reproduce the figures (Figures 7 and G3-G7) and tables (Tables G3-G4) based on the previous results.
- Run `ex4_GO.R` to perform gene ontology analysis (Figrue G8).