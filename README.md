# Deep Learning-based Image Captioning with TensorFlow

This repository contains the code for an image captioning system that uses deep learning techniques in TensorFlow. The project leverages a pre-trained VGG16 model for feature extraction and an LSTM-based network for generating captions.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Generating Captions](#generating-captions)
- [Evaluation](#evaluation)
- [Results](#results)
## Introduction
This project aims to generate descriptive captions for images using a deep learning approach. The system extracts features from images using a VGG16 model and generates captions using an LSTM network.

## Features
- **Feature Extraction**: Uses VGG16 to extract features from images.
- **Caption Generation**: Utilizes an LSTM network to generate captions.
- **Evaluation**: Implements BLEU score evaluation to measure caption quality.
- **Visualization**: Displays images with actual and predicted captions.

## Dataset
The project uses the Flickr8k dataset for training and evaluation. The dataset contains 8,000 images, each with five different captions.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/ShruthikRavula/Image-Captioning.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Image-Captioning
    ```


## Usage
1. **Extract Features**:
    Run the script to extract features from images using VGG16:
    ```bash
    python extract_features.py
    ```

2. **Preprocess Captions**:
    Preprocess the captions and create tokenizers:
    ```bash
    python preprocess_captions.py
    ```

3. **Train the Model**:
    Train the LSTM model on the preprocessed data:
    ```bash
    python train_model.py
    ```

4. **Generate Captions**:
    Use the trained model to generate captions for new images:
    ```bash
    python generate_captions.py --image_path /path/to/image.jpg
    ```

## Training the Model
The training script loads the preprocessed data and trains the LSTM model. Training parameters such as batch size and number of epochs can be configured in the script.

## Generating Captions
The caption generation script loads a pre-trained model and generates captions for a given image. It also displays the image with the actual and predicted captions.

## Evaluation
The project includes a script to calculate BLEU scores for evaluating the quality of the generated captions against the ground truth.

## Results
Sample results showing images with their actual and predicted captions are availabe along with code.
