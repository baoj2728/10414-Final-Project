# 10-414 Final Project
Public repository and stub/testing code for Final Project


## Major File Edits
src/ndarray_backend_cuda.cu\
python/needle/ops/ops_flash_attention.py (added file)\
python/needle/ops/__init__.py\
python/needle/backend_ndarray/ndarray.py\
python/needle/nn/nn_transformer.py\
apps/models.py

## Jupyter Notebook Script for Testing
```python
import sys
sys.path.append('./python')
```
```python
# Download the PTB dataset
import urllib.request
import os

!mkdir -p './data/ptb'
# Download Penn Treebank dataset
ptb_data = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb."
for f in ['train.txt', 'test.txt', 'valid.txt']:
    if not os.path.exists(os.path.join('./data/ptb', f)):
        urllib.request.urlretrieve(ptb_data + f, os.path.join('./data/ptb', f))

```
```python
# Training Cycle
import needle as ndl
sys.path.append('./apps')
from models import LanguageModel
from simple_ml import train_ptb, evaluate_ptb

device = ndl.cuda()
corpus = ndl.data.Corpus("data/ptb")
train_data = ndl.data.batchify(corpus.train, batch_size=32, device=device, dtype="float32")
model = LanguageModel(20, len(corpus.dictionary), hidden_size=32, num_layers=1, seq_model='transformer', seq_len=20, device=device, flash=False)
train_ptb(model, train_data, seq_len=20, n_epochs=10, device=device, lr=0.003, optimizer=ndl.optim.Adam)
evaluate_ptb(model, train_data, seq_len=20, device=device)
```
