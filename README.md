# Text Classification with Capsule Network
Implementation of our paper 
["Investigating Capsule Networks with Dynamic Routing for Text Classification"](https://arxiv.org/pdf/1804.00538.pdf) which is accepted by EMNLP-18.

Requirements: Code is written in Python (2.7) and requires Tensorflow (1.4.1).

# Data Preparation
The reuters_process.py provides functions to clean the raw data and generate Reuters-Multilabel and Reuters-Full datasets. For quick start, please refer to [downloadDataset](https://drive.google.com/open?id=1a4rB6B1FDf7epZZlwXIppaSA7Nr8wSpt) for the Reuters-Multilabel dataset.

# More explanation 
The utils.py includes several wrapped and fundamental functions such as _conv2d_wrapper, _separable_conv2d_wrapper and _get_variable_wrapper etc.

The layers.py implements capsule network including Primary Capsule Layer, Convolutional Capsule Layer, Capsule Flatten Layer and FC Capsule Layer.

The network.py provides the implementation of two kinds of capsule network as well as baseline models for the comparison.

The loss.py provides the implementation of three kinds of loss function: cross entropy, margin loss and spread loss.

# Quick start

```bash
python ./main.py --model_type CNN --learning_rate 0.0005

python ./main.py --model_type capsule-A --learning_rate 0.001
```

# Performance on Reuters-Multilabel dataset

```bash
Capsule-A
Epoch: 1  Val accuracy: 82.9%  Loss: 0.1149
ER: 0.015 Precision: 0.236 Recall: 0.362 F1: 0.255
Epoch: 2  Val accuracy: 88.8%  Loss: 0.0748
ER: 0.172 Precision: 0.466 Recall: 0.500 F1: 0.459
Epoch: 3  Val accuracy: 89.3%  Loss: 0.0601
ER: 0.495 Precision: 0.765 Recall: 0.751 F1: 0.734
Epoch: 4  Val accuracy: 90.5%  Loss: 0.0560
ER: 0.578 Precision: 0.829 Recall: 0.817 F1: 0.802
Epoch: 5  Val accuracy: 90.1%  Loss: 0.0530
ER: 0.609 Precision: 0.841 Recall: 0.838 F1: 0.822
Epoch: 6  Val accuracy: 90.9%  Loss: 0.0505
ER: 0.600 Precision: 0.850 Recall: 0.854 F1: 0.831
Epoch: 7  Val accuracy: 92.0%  Loss: 0.0474
ER: 0.600 Precision: 0.873 Recall: 0.837 F1: 0.833

Capsule-B
Epoch: 1  Val accuracy: 82.7%  Loss: 0.0867
ER: 0.031 Precision: 0.257 Recall: 0.226 F1: 0.235
Epoch: 2  Val accuracy: 90.9%  Loss: 0.0586
ER: 0.458 Precision: 0.752 Recall: 0.663 F1: 0.692
Epoch: 3  Val accuracy: 93.9%  Loss: 0.0431
ER: 0.612 Precision: 0.943 Recall: 0.792 F1: 0.841

CNN:
ER: 0.028 Precision: 0.307 Recall: 0.199 F1: 0.234
Epoch: 2  Val accuracy: 92.0%  Loss: 0.0462
ER: 0.200 Precision: 0.687 Recall: 0.492 F1: 0.555
Epoch: 3  Val accuracy: 94.7%  Loss: 0.0346
ER: 0.265 Precision: 0.876 Recall: 0.589 F1: 0.683
Epoch: 4  Val accuracy: 95.2%  Loss: 0.0310
ER: 0.255 Precision: 0.890 Recall: 0.581 F1: 0.683
Epoch: 5  Val accuracy: 95.4%  Loss: 0.0298
ER: 0.262 Precision: 0.887 Recall: 0.581 F1: 0.682
Epoch: 6  Val accuracy: 95.2%  Loss: 0.0295
ER: 0.262 Precision: 0.884 Recall: 0.577 F1: 0.679
Epoch: 7  Val accuracy: 95.8%  Loss: 0.0294
ER: 0.246 Precision: 0.881 Recall: 0.566 F1: 0.671
```

Notes: Val accuracy and loss are evaluated on dev (single-label), the metrics such as ER and Precision are evaluated on test (multi-label).

The main functions are already in this repository. For any questions, you can report issue here.

# Reference
If you find our source code useful, please consider citing our work.
```
@article{zhao2018investigating,
  title={Investigating Capsule Networks with Dynamic Routing for Text Classification},
  author={Zhao, Wei and Ye, Jianbo and Yang, Min and Lei, Zeyang and Zhang, Suofei and Zhao, Zhou},
  journal={arXiv preprint arXiv:1804.00538},
  year={2018}
}

@article{zhang2018fast,
  title={Fast Dynamic Routing Based on Weighted Kernel Density Estimation},
  author={Zhang, Suofei and Zhao, Wei and Wu, Xiaofu and Zhou, Quan},
  journal={arXiv preprint arXiv:1805.10807},
  year={2018}
}
```

Our second paper makes Capsule Network in relation with Kernel Density Estimation, and provides routing algorithm with explicit objective function to minimize.
