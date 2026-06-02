# 🧚🏻‍♀️ProGENIE: An interpretable AI framework for transcriptomic inference and therapy-relevant profiling from prostate cancer histology

### Abstract

*Molecular stratification is central to precision prostate cancer management, yet high costs and tissue requirements limit routine clinical implementation. Recent advances in AI have enabled transcriptomic inference from routine histology. However, the biological relevance of inferred transcriptomes remains incompletely characterized. Here, we present ProGENIE, an interpretable framework that integrates spatially aware transformation and multi-head attention pooling to infer genome-scale transcriptomic profiles (16,591 genes) from H&E-stained images. ProGENIE generates spatially resolved transcriptomic heatmaps that recapitulate proliferation, stromal, and immune programs within the tumor microenvironment (TME). Using an independent cohort from South Australian hospitals (SAH), the model identified 3,167 well-predicted genes and demonstrated improved performance relative to representative histology-to-transcriptomics methods. Further, we demonstrate the value of ProGENIE in stratifying recurrence risk, characterizing TME programs, and capturing therapy-relevant biological signals. By enabling genome-scale molecular profiling from routine histology, ProGENIE provides a scalable, cost-effective strategy with the potential to expand access to precision oncology.*

### Overview
<p align="center">
  <img width="746" height="659" alt="image" src="https://github.com/user-attachments/assets/96cb3945-294e-4b9c-bbb0-10fcb683e6bd" />
</p>

## Pre-requisites
- Linux (Tested on Red Hat Enterprise Linux 8.4)
- NVIDIA GPU (Tested on NVIDIA A100 PCIe 40GB)
- Python (Python 3.10.0),
PyTorch==1.13.1+cu116,
Torchvision==0.14.1+cu116,
Torchaudio==0.13.1+cu116,
Matplotlib==3.10.1,
NumPy==1.23.5,
OpenCV-Python==4.11.0.86,
Openslide-Python==1.4.1,
Pandas==2.2.3,
Scikit-Image==0.25.2,
Scikit-Learn==1.6.1,
SciPy==1.15.2,
Seaborn==0.13.2,
Einops==0.8.1,
Transformers==4.49.0,
Timm==1.0.3,
Tensorboard==2.19.0,
TensorboardX==2.6.2.2

## Installation Guide for Linux (using anaconda)
1. Clone this git repository: `git clone https://github.com/ABILiLab/ProGENIE.git`
2. `cd ProGENIE`
3. Create a conda environment: `conda create -n progenie python=3.10.0`
4. `conda activate progenie`
5. Install the required package dependencies: `pip install -r requirements.txt`


## Preparation
1. Prepare the `wsi/` directory, which contains the whole slide images
2. Prepare the reference file: `example/ref_file.csv`

  For example:

| WSI File Name      | Patient ID       | rna_A1BG |  ...       | rna_ZZZ3  | tcga_project |
|-------------------|------------------|----------|------------|------------|---------------|
| TCGA-2A-A8VL-01A  | TCGA-2A-A8VL-01A | 0.0658   |  ...       | 2.4027  | TCGA-PRAD     |
| TCGA-2A-A8VO-01A  | TCGA-2A-A8VO-01A | 0.0243   |  ...       | 2.5807 | TCGA-PRAD     |
| TCGA-2A-A8VT-01A  | TCGA-2A-A8VT-01A | 0.0195   |  ...       | 3.6254 | TCGA-PRAD     |

3. Prepare the ground truth label file: `examples/true_label.csv`
   

## Preprocessing
**1. Create patches from WSIs**
   
   To extract image patches from raw Whole Slide Images (WSIs), run the patch generation script provided in: `pre_processing/create_patches.py`
   
   An example script to run the patch extraction: `scripts/create_patches.sh`
   
**2. Extract Features from Patches**

   To extract features from patches using a pretrained encoder, run the feature extraction script provided in: `pre_processing/extract_patch_features.py`
   
   An example script to run the feature extraction:`scripts/extract_patches_features.sh`

   Note: Pretrained encoder weights (e.g., for UNI, CHIEF, and Prov-GigaPath) can be obtained by referring to the following repositories:
  
   - UNI: https://github.com/mahmoodlab/UNI

   - CHIEF: https://github.com/hms-dbmi/CHIEF

   - Prov-GigaPath: https://github.com/prov-gigapath/prov-gigapath
   
**3. Obtain k-Means Features**

  To compute k-Means features from extracted patch features, run the clustering script provided in: `pre_processing/kmean_features.py`

  An example script to run the k-Means clustering: `scripts/kmean_features.sh`

## Inference on independent dataset
We released the model weights for four pre-trained models on [HuggingFace](https://huggingface.co/ananananxuan/ProGENIE/tree/main "HuggingFace"), please download the weights first.

**1. Prepare the dataset**

To combine the k-Means features with ground truth gene expression profiles for model training, run the dataset preparation script provided in: `pre_processing/prepare_dataset.py`

An example script to run the dataset preparation: `scripts/prepare_dataset.sh`

**2. Inference and evaluation**

To perform model inference on the test set and evaluate performance:

Run the main inference and evaluation script:`inference.py`

You can find an example script here:`scripts/inference.sh`

The output will be saved in: `examples/results`

This includes:

- test_pred_labels.csv: predicted gene expression values

- test_true_labels.csv: ground truth labels

- test_gene_metrics.csv: PCC, RMSE, and R²

## Visualization on spatial 
Generate spatial gene expression heatmaps from spatial transcriptomics slides using a sliding-window inference strategy. Predicted expression values are assigned to the center patch of each window, and optional Gaussian smoothing can be applied to both ground-truth and predicted expression maps for visualization.

An example script to run the visualization:
```
python visualization.py \
  --base_dir spatial \
  --count_file filtered_feature_bc_matrix.h5 \
  --h5_path wsi_features.h5 \
  --model_path model_best.pth \
  --out_dir heatmap \
  --offsets_json wsi_crop.json \
  --gene ACTA2
```


## Reference

Please cite our publication: Han _et al_., "Predicting gene expression from whole slide images in prostate cancer using deep learning". doi: xxx.










