#!/bin/bash



# Run the dataset preparation script
python pre_processing/prepare_dataset.py \
    --base_path examples/features_uni/TCGA-PRAD \
    --label_path examples/true_label.csv \
    --save_path examples/dataset \
    --output_name independent_dataset.pt 
