# Brain-Tumor-Classification

The detection of brain tumors is a task of great importance for the early diagnosis and treatment planning of patients with neurological conditions. However, manual interpretation of brain Magnetic Resonance Image scans requires professional expertise and can be time-consuming. Nowadays, using machine learning and deep learning in specific, it is possible to aid professionals to identify brain tumors more reliably. In this study, a deep learning-based approach is proposed for automated classification using Convolutional Neural Networks, by training the model wich learn the patterns associated with brain tumors from a dataset of labeled MRI scans, is expected to get a classifier capable distinguishing between healthy and unhealthy images. The proposed solution includes preprocessing techniques such as normalization, the design of a deep network with three convolutional layers and two linear layers in order to get a classifier of brain tumors, and the evaluation of the model through accuracy, precision, recall, and F1-score metrics. The ultimate goal is to develop a classifier capable of assisting professionals by providing consistent and accurate tumor detection, improving diagnostic speed and precision, and contributing to more timely treatment planning by analyzing how the CNN works and process medical images to evaluate the accuracy of brain tumor diagnosis. 

## Build on

- Python 3.12.2

## Installation

```bash
pip install -r requirements.txt
```

## Data availability

Te dataset is shared open source. 

Bhuvaji, S,. Kadam, A., Bhumkar, P., Dedge, S., and Kanchan, S. (2020). Brain Tumor Classification (MRI). Kaggle. https://doi.org/10.34740/KAGGLE/DSV/1183165
