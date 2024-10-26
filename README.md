# Brain-Tumor-Classification

The detection of brain tumors is a task of great importance for the early diagnosis and treatment planning of patients with neurological conditions. In this study, a Convolutional Neural Network is proposed for automated classification. By training the model to learn patterns associated with brain tumors from a dataset of labeled MRI scans, the goal is to develop a classifier capable of distinguishing between healthy and unhealthy images.

The model's evaluation is performed using accuracy, precision, recall, and F1-score metrics. The ultimate objective is to assist medical professionals by improving diagnostic speed and precision, thereby contributing to more timely treatment planning.

## Build on

- Python 3.12.2
- PyTorch 2.4.0
- Scikit-learn~=1.5.1
- Numpy~=1.26.4
- matplotlib~=3.8.4

## Installation

To set up the environment, install the required packages by running:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
```bash
git clone https://github.com/Josespa/Brain-Tumor-Classification
cd Brain-Tumor-Classification
```

2. Run the script for train the model and evaluation
```bash
python main.py
```

## Data availability

Te dataset is shared open source. 

Bhuvaji, S,. Kadam, A., Bhumkar, P., Dedge, S., and Kanchan, S. (2020). Brain Tumor Classification (MRI). Kaggle. https://doi.org/10.34740/KAGGLE/DSV/1183165

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
Special thanks to the contributors of the Kaggle dataset used for this project.
Thanks to the open-source community for the tools and libraries that made this project possible.