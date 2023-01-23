# NerCo: A Contrastive Learning based Two-stage Chinese NER Method  

This repo provides the source code & data of our paper "NerCo: A Contrastive Learning based Two-stage Chinese NER Method  " .

### Overview

NerCo is our proposed two-stage learning approach for tackling *Entity Representation Segmentation in Label-semantics*. Unlike traditional sequence labeling methods which lead to the above problem, our approach takes a two-stage NER strategy. In the first stage, we conduct contrastive learning for label-semantics based representations. Then we finetune the learned model in the second stage, equipping it with inner-entity position discrimination for chunk tags and linear mapping to type tags for each token. 

![image-20230123100746577](figures/framework.png)

<p align="center">
    Figure1: A comparison between traditional sequence labeling methods and our proposed method NerCo. 
</p>

![image-20230123114406634](figures/contrast.png)

<p align="center">
    Figure2: Contrastive representation learning as the first stage of NerCo.  
</p>

## Dependencies

```bash
source activate # use conda
conda create --name nerco python=3.7.3 # create a virtual enviroment named nerco
conda activate nerco # activate
pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html # torch
pip3 install -r requirements.txt # requirement file
cp fastnlp_src/* ~/.conda/envs/nerco/lib/python3.7/site-packages/fastNLP/core/. # overwrite fasnlp source
```

## Data Preparation
1. Download the character embeddings and word embeddings.(Provided by [Flat](https://github.com/LeeSureman/Flat-Lattice-Transformer/blob/master/README.md)). Put them into `data/word` subdirectory.

      Character and Bigram embeddings (gigaword_chn.all.a2b.{'uni' or 'bi'}.ite50.vec) : [Google Drive](https://drive.google.com/file/d/1_Zlf0OAZKVdydk7loUpkzD2KPEotUE8u/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)

      Word(Lattice) embeddings: 
      
      yj, (ctb.50d.vec) : [Google Drive](https://drive.google.com/file/d/1K_lG3FlXTgOOf8aQ4brR9g3R40qi1Chv/view?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1pLO6T9D)
      
      ls, (sgns.merge.word.bz2) : [Baidu Pan](https://pan.baidu.com/s/1luy-GlTdqqvJ3j-A4FcIOw)

2. Datasets.
You can either preprocess the datasets(using `preprocess.py`) or download our preprocessed-done [datasets](https://drive.google.com/drive/folders/1efbRAjqIRe5y1meiNEZdR4EzRmY7tOM0?usp=sharing). Put each dataset into `data/datasets/`(e.g. `data/datasets/weibo` for Weibo NER dataset)
## Evaluate
You can either evaluate our trained checkpoints(download [here](https://drive.google.com/drive/folders/17PD3q4Hl77DKq0PjfQp4gmH2715hWZJF?usp=sharing)) or your models trained from scratch(see next section).
```bash
cd evaluate
python weibo.py #taking weibo dataset for evaluation example
```

## Train

```bash
cd train
python ontonotes.py #taking weibo ontonotes for traing example
```
