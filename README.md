# interspeech2021_MTDA
the code for interspeech2021 submission. my interspeech2021 paper name is "Unsupervised Multi-Target Domain Adaptation for Acoustic Scene Classification". <br/>
In this repository, I finish my method MTDA, which proposed by my paper. Furthermore, I also finish other SOTO methods, these methods are not open source. So I write code for them according to thier paper. The details you can refer my paper. <br/>
This repository contains the code for six experiments in my paper
* MTDA: proposed method in my paper.
* DANN: for paper "Domainadaptation  neural  network  for  acoustic  scene  classification  inmismatched conditions"
* MCD: For paper  "Maximumclassifier  discrepancy  for  unsupervised  domain  adaptation"
* UADA: for paper  "Unsupervised adversarial domain adaptation for acoustic sceneclassification"
* W-UADA: For paper  "Unsupervised  adver-sarial  domain  adaptation  based  on  the  wasserstein  distance  foracoustic scene classification"
* MMD: this method have been used in the paper MCD as comparable method, I coding for it according to paper "Learning transferablefeatures with deep adaptation networks"

## Prerequisites

Hardware:
* A GPU

Software:
* Python>= 3.7
* PyTorch (I use 1.6.0)
* numpy, scipy....

## Quick Start

```python
# Clone the repository

# Download the data: I will release logmel feature eatracted from DCASE2020 task1A dataset soon, and you can also get mel spectrum by yourself according to our code in data folder.

# Train a model for MTDA
python Main/main.py            # Needs to run on a GPU

# Train a model for DANN
python Main/DANN_main.py            # Needs to run on a GPU

# Train a model for MCD
python Main/mcd_main.py            # Needs to run on a GPU

# Train a model for MMD
python Main/mmd_main.py            # Needs to run on a GPU

# Train a model for UADA and W-UADA
python Main/wgan_main.py            # Needs to run on a GPU

```

## Organization of the Repository

### model

The `model` directory contains six methods code, you can find them according their name.

### data

the folder contain dataloader for train and test.
### bestmodel
 we release part of our training model results, your can use this model to test directly.

## Citing
this code are based  following code:
https://github.com/hehaodele/CIDA
