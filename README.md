# OCR-Letter-Recognition
PyTorch-based OCR System for A-Z character recognition using a ResNet-18 deep learning architecture.
OCR Letter Recognition

A deep-learning-based Optical Character Recognition (OCR) system for recognizing characters from images using a ResNet18-based model.

Overview

This project implements an OCR pipeline for character recognition. The system processes character images and uses a fine-tuned ResNet18 convolutional neural network to classify the input character.

The repository contains the dataset-processing, label-checking, testing, and evaluation scripts used during development.

Features

* Character image preprocessing
* Dataset preparation and inspection
* Label verification
* ResNet18-based character classification
* Model evaluation
* Testing and prediction
* Confusion-matrix-based analysis
* Pretrained model available through GitHub Releases

Project Structure

OCR-Letter-Recognition/
│
├── OCRdataset.py
├── check.py
├── check_labels.py
├── checkforsimilarfonts(confusionmatrix).py
├── evaluation.py
├── testing.py
├── .gitignore
└── README.md

Model

The project uses a ResNet18 architecture for character recognition.

The pretrained model and supporting NumPy file are distributed separately as GitHub Release assets so that large binary files do not need to be stored directly in the Git repository.

Release Assets

Download the following files from the latest release:

* resnet18_ocr.pth
* characterfont.npz

After downloading them, place them in the project directory as required by the scripts.

Installation

Clone the repository:

git clone https://github.com/ravi15-12/OCR-Letter-Recognition.git
cd OCR-Letter-Recognition

Create a virtual environment:

python -m venv venv

Activate it on macOS/Linux:

source venv/bin/activate

Install the required Python packages:

pip install torch torchvision opencv-python numpy

Usage

After downloading the release assets, place the required files in the location expected by the scripts.

Dataset Processing

python OCRdataset.py

Check Labels

python check_labels.py

Evaluation

python evaluation.py

Testing

python testing.py

The exact execution order may depend on the dataset and paths configured in the scripts.

Evaluation

The model was evaluated using standard classification metrics and confusion-matrix analysis.

The evaluation scripts included in this repository can be used to inspect model performance and identify character classes that may be confused with one another.

Technologies Used

* Python
* PyTorch
* TorchVision
* OpenCV
* NumPy
* ResNet18
* Convolutional Neural Networks (CNN)
* Optical Character Recognition (OCR)

Model Analysis

A dedicated analysis script is included for investigating visually similar characters and examining confusion between character classes:

checkforsimilarfonts(confusionmatrix).py

This helps identify classes that require further analysis or model improvement.

Pretrained Model

The pretrained model is provided through the repository’s GitHub Releases.

Repository:
https://github.com/ravi15-12/OCR-Letter-Recognition

Download the latest release assets before running inference with the pretrained model.

Future Improvements

* Improve recognition of visually similar characters
* Expand the training dataset
* Add additional data augmentation
* Improve preprocessing and normalization
* Add a dedicated inference interface
* Optimize the model for faster deployment
* Extend the system from isolated character recognition toward complete word/document OCR

Author

Ravi Kant

B.Tech — Electronics & Communication Engineering
BIT Sindri
