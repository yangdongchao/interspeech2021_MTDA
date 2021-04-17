# interspeech2021_MTDA
the code for interspeech2021 submission. my interspeech2021 paper name is "Unsupervised Multi-Target Domain Adaptation for Acoustic Scene Classification". <br/>
In this repository, I finish my method MTDA, which proposed by my paper. Furthermore, I also finish other SOTO methods, many of methods are not open source. So I write code for them according to thier paper. The details you can refer my paper. After the review, I will release my paper as soon as possible on arxiv. <br/>
This repository contains the code for six experiments in my paper
* MTDA: The proposed method in my paper.
* DANN: For paper "Domain adaptation  neural  network  for  acoustic  scene  classification  inmismatched conditions", you can get more details on this paper.
* MCD: For paper  "Maximum classifier  discrepancy  for  unsupervised  domain  adaptation". I only finish part of this paper, this paper utilize KD(kownlegde distillation) to improve source domain accuracy, in order to fairly compare with my method, I donnot use KD technology.
* UADA: for paper  "Unsupervised adversarial domain adaptation for acoustic sceneclassification". The author of paper also release their code, you can also refer thier code, <a>https://github.com/shayangharib/AUDASC</a>, but I still recommend to read our code, because it is easy to understand.
* W-UADA: For paper  "Unsupervised  adver-sarial  domain  adaptation  based  on  the  wasserstein  distance  foracoustic scene classification"
* MMD: this method have been used in the paper MCD as comparable method, I code for it according to paper "Learning transferablefeatures with deep adaptation networks"

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

# Download the data: I will release logmel feature eatracted from DCASE2020 task1A dataset soon, and you can also get mel spectrum by yourself according to our code in data folder. The way of get logmel feature, I refer <a name='dcase2019_task1'>https://github.com/qiuqiangkong/dcase2019_task1</a>. 

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
## TO DO list
I am sorry for these dirty code,  I will add note for these code in future to make you quickly read. <br/>
I will reorganize these code in future, and make it fully.
## Organization of the Repository

### models

The `models` directory contains six methods code, you can find them according their name.

### data

the folder contain dataloader for train and test.
### best_model
 we release part of our training model results, you can use these models to test directly.

## Citing
this code are based  following code <br/>
https://github.com/hehaodele/CIDA
https://github.com/qiuqiangkong/dcase2019_task1
