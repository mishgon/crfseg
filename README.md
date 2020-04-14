# crfseg: CRF layer for segmentation in PyTorch

Conditional random field (CRF) is a classical graphical model which allows to make structured predictions 
in such tasks as image semantic segmentation or sequence labeling.

You can learn about it in papers:
* [Efficient Inference in Fully Connected CRFs with
Gaussian Edge Potentials](https://arxiv.org/pdf/1210.5644.pdf)
* [Conditional Random Fields as Recurrent Neural Networks](https://arxiv.org/pdf/1502.03240.pdf)

## Installation
`pip install crfseg`

## Usage
Can be easily used as differentiable (and moreover learnable) postprocessing layer of your NN for segmentation.
```angular2html
import torch
import torch.nn as nn
from crfseg import CRF

model = nn.Sequential(
    nn.Identity(),  # your NN
    CRF(n_spatial_dims=2)
)

batch_size, n_channels, spatial = 10, 1, (100, 100)
x = torch.zeros(batch_size, n_channels, *spatial)
log_proba = model(x)
```
