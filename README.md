# SurvFinder

The source code of article 'Multiview deep-learning-enabled histopathology for risk stratification and treatment decision-making in stage II colorectal cancer'.

## System requirements
This code was developed and tested in the following settings. 
### OS
- Ubuntu 20.04
### GPU
- Nvidia GeForce RTX 2080 Ti
### Dependencies
- Python (3.9.6)
- Pytorch install = 1.9
- torchvision (0.6)
- CUDA (10.1)
- openslide_python (1.1.1)
- tensorboardX (2.4)
Other dependencies: Openslide (3.4.1), matplotlib (3.1.1), numpy (1.18.1), opencv-python (4.1.1), pandas (1.0.3), pillow (7.0.0), scikit-learn (0.22.1), scipy (1.3.1)
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

### Training and evaluation of MVNet
Training
```
python /ntl/train_ntl.py --Task 'lym' --TrainFolder "/train_folder"  --Comment 'comment' --Pretrain --FoldN  --BatchSize  --Epoch  --Resume 0 --Stat 'train'
```
Evaluation
```
python /ntl/train_ntl.py --Task 'lym' --TrainFolder "/test_folder"  --Comment 'comment'  --FoldN  --BatchSize  --Epoch  --Resume 0 --Stat 'test'
```

The trained checkpoints are saved in /ckpt, 5 files derived from 5-fold cross validation.




