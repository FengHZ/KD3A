# KD3A: Unsupervised Multi-Source Decentralized Domain Adaptation via Knowledge Distillation
Here is the official implementation of the model `KD3A` in paper ["KD3A: Unsupervised Multi-Source Decentralized Domain Adaptation via Knowledge Distillation"](https://arxiv.org/abs/2011.09757).

## Model Review
* Knowledge Distillation
  
  ![KD](./images/kd.PNG)

* Knowledge Vote
  
  ![KV](./images/kv.PNG)

## Setup
### Install Package Dependencies
```
Python Environment: >= 3.6
torch >= 1.2.0
torchvision >= 0.4.0
tensorbard >= 2.0.0
numpy
yaml
```
### Install Datasets
We need users to declare a `base path` to store the dataset as well as the log of training procedure. The directory structure should be
```
base_path
│       
└───dataset
│   │   DigitFive
│       │   mnist_data.mat
│       │   mnistm_with_label.mat
|       |   svhn_train_32x32.mat  
│       │   ...
│   │   DomainNet
│       │   ...
│   │   OfficeCaltech10
│       │   ...
|   |   Office31
|       |   ...
└───trained_model_1
│   │	parmater
│   │	runs
└───trained_model_2
│   │	parmater
│   │	runs
...
└───trained_model_n
│   │	parmater
│   │	runs    
```
Our framework now support four multi-source domain adaptation datasets: ```DigitFive, DomainNet, OfficeCaltech10 and Office31```.

* DigitFive
  
  The DigitFive dataset can be accessed in [Google Drive](https://drive.google.com/file/d/1QvC6mDVN25VArmTuSHqgd7Cf9CoiHvVt/view?usp=sharing).
* DomainNet
  
  [VisDA2019](http://ai.bu.edu/M3SDA/) provides the DomainNet dataset.

### Unsupervised Multi-source Domain Adaptation
The configuration files can be found under the folder  `./config`, and we provide four config files with the format `.yaml`. To perform the unsupervised multi-source decentralized domain adaptation on the specific dataset (e.g., DomainNet), please use the following commands:

```python
python main.py --config DomainNet.yaml --target-domain clipart -bp base_path
```

The training process for DomainNet is as follows.

  ![top1](./images/DomainNet-Top1.svg)

  ![top5](./images/DomainNet-Top5.svg)

### Negative Transfer

In training process, our model will record the domain weights as well as the accuracy for target domain as 
```
Source Domains  :['infograph', 'painting', 'quickdraw', 'real', 'sketch']

Domain Weight : [0.1044, 0.3263, 0.0068, 0.2531, 0.2832]

Target Domain clipart Accuracy Top1 : 0.726 Top5: 0.902
```
* Irrelevant Domains
  
  We view quickdraw as the irrelevant domain, and the K3DA assigns low weights to it in training process.

* Malicious Domains
  
  We use the poisoning attack with level $m\%$ to create malicious domains. The related settings in the configuration files is as follows:
  ```
  UMDAConfig:
      malicious:
        attack_domain: "real"
        attack_level: 0.3
  ```
  With this setting, we will perform poisoning attack in the source domain `real` with $30\%$ mislabeled samples.

### Communication Rounds

We also provide the settings in `.yaml` config files to perform model aggregation with communication rounds $r$ as follows:
```
UMDAConfig:
    communication_rounds: 1
```
The communication rounds can be set into $[0.2, 0.5 , 1 , ... , N]$.

## Reference

If you find this useful in your work please consider citing:
```
@article{DBLP:journals/corr/abs-2011-09757,
  author    = {Haozhe Feng and
               Zhaoyang You and
               Minghao Chen and
               Tianye Zhang and
               Minfeng Zhu and
               Fei Wu and
               Chao Wu and
               Wei Chen},
  title     = {{KD3A:} Unsupervised Multi-Source Decentralized Domain Adaptation
               via Knowledge Distillation},
  journal   = {CoRR},
  volume    = {abs/2011.09757},
  year      = {2020},
  url       = {https://arxiv.org/abs/2011.09757},
  archivePrefix = {arXiv},
  eprint    = {2011.09757}
}
```





  
