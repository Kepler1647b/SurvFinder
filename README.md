# SurvFinder

The source code of article 'Multiview deep-learning-enabled histopathology for risk stratification and treatment decision-making in stage II colorectal cancer'.

## System requirements
This code was developed and tested in the following settings. 
### OS
- Ubuntu 20.04
### GPU
- Nvidia GeForce RTX 2080 Ti
### Dependencies
- Python (3.8.12)
- Pytorch install = 1.10.1
- torchvision (0.10.0)
- CUDA (11.3)
## Installation guide

- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers) on your machine (download the distribution that comes with python3).  
  
- After setting up Miniconda, install OpenSlide (3.4.1):  
```
apt-get install openslide-python
```
- Create a conda environment with lgnet.yaml:
```
conda env create -f sfinder.yaml
```  
- Activate the environment:
```
conda activate sfinder
```
- Typical installation time: 1 hour

## Preprocessing
We use the python files to convert the WSI to patches with size 256*256 pixels and taking color normalization for comparison.
### Slide directory structure
```
DATA_ROOT_DIR/
    └──case_id/
        ├── slide_id.svs
        └── ...
    └──case_id/
        ├── slide_id.svs
        └── ...
    ...
```
### Generating patches
- /preprocessing/generate_patch.py
### Color normalization methods
- /preprocessing/StainTools-master/main_staintools.py
## Training and evaluation of SurvFinder

### Training and evaluation of SegNet
Training of TNTM
```
python /ntl/train_ntl.py --Task 'ntl' --TrainFolder "/train_folder"  --Comment 'comment' --Pretrain --FoldN  --BatchSize  --Epoch  --Resume 0 --Stat 'train'
```
Evaluation of TNTM
```
python /ntl/train_ntl.py --Task 'ntl' --TrainFolder "/test_folder"  --Comment 'comment'  --FoldN  --BatchSize  --Epoch  --Resume 0 --Stat 'test'
```

Training of TLSM
```
python /tls3/train_lym3.py --Task 'lym' --TrainFolder "/train_folder"  --Comment 'comment' --Pretrain --FoldN  --BatchSize  --Epoch  --Resume 0 --Stat 'train'
```
Evaluation of TLSM
```
python /tls3/train_lym3.py --Task 'lym' --TrainFolder "/test_folder"  --Comment 'comment'  --FoldN  --BatchSize  --Epoch  --Resume 0 --Stat 'test'
```
### Training and evaluation of MVNet
Training
```
python /surv/main.py
```
Evaluation
```
python /surv/eval.py
```
### Figure_Generation Code

This folder contains R scripts used to generate the main figures in the manuscript.

- `single_roc.R`: Generates ROC curves (e.g., Figure 2).
- `confusion_matrix.R`: Generates confusion matrix plots (e.g., Figure 2).
- `triple_roc.R`: Generates multiple ROC curves for comparison (e.g., Figure 3).
- `kmcurv.R`: Plots Kaplan–Meier survival curves for risk groups (e.g., Figure 4).
- `kmcurv_chemo.R`: Plots Kaplan–Meier curves stratified by adjuvant chemotherapy status (e.g., Figure 5).
- `cox_analysis.R`: Generates forest plots for univariate and multivariate Cox regression analyses.

Each script can be run independently with the appropriate input data. Please ensure that required R packages (e.g., `survival`, `survminer`, `ggplot2`) are installed.