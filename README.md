# guidance-to-fairest-boundary

This repository will contain the official implementation of our ICML 2026 paper:

**Fair Machine Learning, Fairness-Accuracy Trade-off, Fair Bayes-optimal Classifier**


## Usage Instruction
To run experiments on the CelebA dataset, use:
```
python main.py datasets=celeba
```

## Datasets
This repository does not include the datasets or preprocessed files. Please download each dataset from its official source and preprocess it before running the experiments.

For CelebA, our preprocessing pipeline is based on the preprocessing used in **Fair Mixup: Fairness via Interpolation** by Ching-Yao Chuang and Youssef Mroueh. 

For Adult and COMPAS, our preprocessing follows the setup used in **Bayes-Optimal Classifiers under Group Fairness** by Xianli Zeng, Edgar Dobriban, and Guang Cheng.