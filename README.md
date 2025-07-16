# Vision Transformer (ViT) - PyTorch Implementation

This repository contains a PyTorch implementation of the **Vision Transformer (ViT)** model, which is a novel architecture for image classification tasks.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

## Introduction
The Vision Transformer (ViT) is a deep learning architecture that uses transformers (originally designed for NLP tasks) to handle image data. It divides the image into smaller patches and processes these patches in parallel through transformer layers. This project aims to provide a simple, yet functional, PyTorch implementation of ViT.


## Installation

To run this project, you need to have **Python** and **PyTorch** installed. You can install the necessary dependencies as follows:

1. Clone the repository:
   ```
   git clone https://github.com/maturyusuf/vision_transformer.git
   cd vision_transformer
   ```
   
2. Create and activate a virtual environment:
   python -m venv .venv
  # On Windows
  .\.venv\Scripts\activate
  # On Mac/Linux
  source .venv/bin/activate

3. Install required dependencies:
   pip install -r requirements.txt
4. Optionally, if you want to use the project with GPU support, make sure you have CUDA installed and configured correctly.

## Usage

Once the repository is cloned and dependencies are installed, you can train the Vision Transformer model on the MNIST dataset, or modify it to suit other datasets.

## Training

Training code is inside main.py file.Therefore, To train the Vision Transformer on the MNIST dataset, run the following command:
  ```
  python main.py
  ```






