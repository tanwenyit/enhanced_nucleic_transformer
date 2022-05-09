# Enhanced Nucleic Transformer

This is the enhanced version of nucleic transformer proposed by He et al. (2021)

## Kaggle Competition Dataset

You can obtain the datasets from the link below:

1. https://www.kaggle.com/c/stanford-covid-vaccine/data (Original dataset)
2. https://www.kaggle.com/datasets/shujun717/openvaccine-12x-dataset (Augmented dataset)

Afterwards, unzip them all in a single folder "data"

## Environment

The experiment is conducted under virtual environment created using Anaconda.
You can create your own environment and ensure these packages are installed by using ```pip install <package name>``` command.

These are the mandatory packages:
```
torch 1.9.0 or <any version>
tqdm <any version>
ranger-adabelief 0.1.0
adabelief-pytorch 0.2.1
pandas 1.1.2
numpy 1.19.2

```

## Code to reproduce results for the openvaccine dataset

Here I include the hypeparameters that give the best single model.

1. pretrain with all available sequences: ```./pretrain.sh```

2. train on targets: ```./run.sh```

3. to make predictions and generate a csv file for submission: ```./predict.sh``` then you can make a submission at https://www.kaggle.com/c/stanford-covid-vaccine/submissions

## How to run shell script

Run the script under Anaconda Prompt, with the correct environment.

## References
He, S., Gao, B., Sabnis, R., Sun, Q. (2021). Nucleic Transformer: Deep Learning on Nucleic Acids with Self-Attention and Convolutions. 10.1101/2021.01.28.428629.
https://www.biorxiv.org/content/10.1101/2021.01.28.428629v1
Original Code: https://github.com/Shujun-He/Nucleic-Transformer