# Text Classification with Capsule Network
Implementation of our paper 
["Investigating Capsule Networks with Dynamic Routing for Text Classification"](https://arxiv.org/pdf/1804.00538.pdf) which is accepted by EMNLP-18.

Requirements: Code is written in Python (2.7) and requires Tensorflow (1.4.1).

# Data Preparation
The reuters_process.py provides functions to clean the raw data and generate Reuters-Multilabel and Reuters-Full datasets. For quick start, please refer to [downloadDataset](https://drive.google.com/open?id=1a4rB6B1FDf7epZZlwXIppaSA7Nr8wSpt) for the Reuters Multi-label dataset.

# More explanation 
The utils.py includes several wrapped and fundamental functions such as _conv2d_wrapper, _separable_conv2d_wrapper and _get_variable_wrapper etc. to make programming efficiently.

The layers.py implements capsule network including Primary Capsule Layer, Convolutional Capsule Layer, Capsule Flatten Layer and FC Capsule Layer.

The network.py provides the implementation of two kinds of capsule network as well as baseline models for the comparison.

The loss.py provides the implementation of three kinds of loss function: cross entropy, margin loss and spread loss.

# Quick start

```bash
python ./main.py --loss_type margin_loss --embedding_type static -- model_type CNN --learning_rate 0.0005

python ./main.py --loss_type margin_loss --embedding_type static -- model_type capsule-A --learning_rate 0.001
```

The main functions are already in this repository. The next update will including more detailed instructions. For any questions, you can report issue here.
