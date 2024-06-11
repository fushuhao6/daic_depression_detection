# Depression Score Regression using BERT on DAIC-WOZ Dataset

This repository contains the code to train a BERT model to predict depression scores using the DAIC-WOZ dataset.

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)

## Introduction

This project aims to leverage the BERT model for the regression task of predicting depression scores from the DAIC-WOZ dataset. The preprocessing scripts and model training scripts are provided to facilitate the end-to-end pipeline from raw data to model predictions.

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- pandas
- numpy
- sklearn
- matplotlib
- seaborn
- tqdm
- nltk

## Data Preprocessing

To preprocess the DAIC-WOZ dataset, follow these steps:

1. **Download the DAIC-WOZ dataset** to a specific location on your machine, e.g., `/data/DAIC`.
2. **Change the paths** in the `utils/preprocess.py` file to match the location where you have downloaded the dataset.
3. **Run the preprocessing script**:
```
python utils/preprocess.py
```

This will preprocess the data and prepare it for model training.

## Model Training

### Train on the original training set
To train the BERT model for depression score regression, use the following command:
```
python main_regress.py --data_root [your data root]
```
Replace `[your data root]` with the path to the preprocessed dataset.

### Train on the original + synthetic set
To train the BERT model with both the training dataset and the synthetic dataset, use the following command:
```
python main_regress.py --data_root [your data root] --datasets train synthetic
```


### Evaluate trained model on test set
To evaluate the BERT model on the test dataset provided by DAIC-WOZ, simply add the `--eval` tag after the command you
ran for training. e.g., 

```
python main_regress.py --data_root [your data root] --datasets train synthetic --eval
```



