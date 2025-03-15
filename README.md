DeepLearning-ResNet

Efficient ResNet for CIFAR-10 Image Classification

Overview

This repository contains an optimized ResNet-based deep learning model for CIFAR-10 classification. The model is designed to maintain high accuracy while reducing computational complexity, staying under 5 million parameters.

For a detailed explanation of the model architecture, training strategy, results, and performance analysis, refer to the project report.

Repository Structure

DeepLearning-ResNet/
│── Final_model.ipynb       # Final trained ResNet model
│── submission2.csv         # Kaggle competition submission
│── LICENSE                 # Project license
│── README.md               # Project documentation
│── requirements.txt        # Required dependencies
│
├── Instructions/           # Instructions and project guidelines
│
├── Report/                 # Full project report and analysis
│
├── WorkingFiles/           # Development files and experiments
│   ├── Dataset/            # CIFAR-10 dataset
│   ├── environment.yml     # Conda environment setup
│   ├── Esteban/            # Work files for Esteban
│   ├── Shruti/             # Work files for Shruti
│   ├── Steven/             # Work files for Steven
│   ├── LaTeX/              # Report LaTeX files
│   ├── pytorch-cifar-master/ # Reference PyTorch CIFAR implementation
│   ├── ReferenceProjects/  # Related research projects
│   ├── startup_notebook.ipynb # Initial model development notebook

Installation & Setup

Prerequisites

Ensure you have:
	•	Python 3.8+
	•	PyTorch, torchvision, and CUDA (for GPU support)
	•	pip or Conda for package management

Environment Setup
	1.	Clone the repository:

git clone https://github.com/Esteban-D-Lopez/DeepLearning-ResNet.git
cd DeepLearning-ResNet


	2.	Create a virtual environment (optional but recommended):

conda env create -f WorkingFiles/environment.yml
conda activate resnet-env


	3.	Install dependencies:
The project relies on various Python packages for deep learning, data processing, and visualization, including:
	•	torch
	•	torchvision
	•	numpy
	•	scipy
	•	matplotlib
	•	seaborn
	•	scikit-learn
	•	opencv-python
Install all dependencies using:

pip install -r requirements.txt



Training & Evaluation
	1.	Train the model:

python train.py


	2.	Evaluate model performance:

python evaluate.py --weights final_model.pth


	3.	Generate submission for Kaggle:

python generate_submission.py --output submission2.csv



Model Architecture

This project develops an efficient ResNet variant optimized for CIFAR-10. The final model features:
	•	Increased residual block depth
	•	Squeeze-and-Excitation (SE) blocks for feature recalibration
	•	Depthwise separable convolutions for computational efficiency
	•	Optimized learning rate scheduling

For a detailed breakdown of the architecture, refer to the project report.

Contributors
	•	Esteban Lopez – Model development & optimization
	•	Shruti Karkamar – Training strategy & evaluation
	•	Steven Granaturov – Data preprocessing & augmentation

Citations & References

For all references and citations, see the project report.