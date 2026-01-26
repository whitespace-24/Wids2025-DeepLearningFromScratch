# WIDS 2025: Deep Learning From Scratch

A comprehensive learning repository for the Women in Data Science (WIDS) 2025 program, covering foundational deep learning concepts and practical implementations built from scratch.

---

## 📁 Repository Structure

### **Week 1: Foundations of Linear Algebra & NumPy**

This week establishes the mathematical and coding foundations required for deep learning, along with an introduction to Kaggle-style machine learning workflows.

#### Key Topics Covered:
- **Linear Algebra**: Importance in ML, vectors as points and directions, linear transformations, and geometric meaning of matrices
- **NumPy & PyTorch Basics**: Tensor creation, shapes, indexing, slicing, broadcasting, and basic operations
- **Distance Computations**: Efficient vectorized distance matrix calculations and optimization techniques
- **Kaggle Introduction**: Understanding competitions, datasets (train.csv, test.csv), submission workflows, and exploration of the Titanic dataset

#### Files Included:
- Jupyter notebooks with practical examples and implementations
- Distance computation scripts for efficient numerical operations
- Kaggle competition preparation materials

**Note**: Some datasets are not included due to GitHub size limitations.

---

### **Week 2: Neural Networks & Backpropagation**

This week focuses on understanding the theoretical foundations and practical building blocks of neural networks.

#### Key Topics Covered:
- **Perceptron**: Linear binary classifier, role of weights/bias, activation functions, and limitations
- **XOR Problem**: Why single-layer perceptrons fail, motivation for multi-layer networks, geometric intuition
- **Multi-Layer Neural Networks (MLPs)**: Hidden layers, non-linear activations, learning complex boundaries
- **Backpropagation**: Core learning algorithm, chain rule for gradients, weight/bias updates, mathematical foundations

**Note**: Week 2 focused on theoretical learning with interactive materials. No formal coding assignments were completed, but conceptual understanding is emphasized.

---

### **Week 3: Transitioning to Image Data with CNNs**

In this week, the repository explores Convolutional Neural Networks (CNNs), specifically optimized for computer vision tasks like digit recognition.

Context: Similar to the previous K-Nearest Neighbors (KNN) implementation, this project utilizes the MNIST dataset (a collection of 70,000 handwritten digits).
Pre-processing: Images are treated as 28×28 pixel grids. Unlike basic DNNs that flatten these into vectors, the CNN approach preserves the spatial 2D structure of the digits.

Key CNN Components Implemented: 
To understand how the code handles spatial features, we focus on three core concepts: Convolutional Layers (Conv2D): Instead of fully-connected weights, these layers use Filters (Kernels) that slide across the image. This allows the model to learn local patterns (edges, loops, and curves) using Parameter Sharing, which significantly reduces the number of trained variables.

Pooling Layers (MaxPool): These layers perform downsampling (typically taking the maximum value in a 2×2 window) to reduce the computational load and make the feature detection invariant to small shifts or distortions in the handwriting.

This directory includes SimpleCNN_MNIST.ipynb, a practical implementation of a Convolutional Neural Network using the PyTorch framework on MNSIT dataset. 

---

### **Project (Week 4 & 5)**

This project implements a Convolutional Neural Network (CNN) for MNIST digit classification, where the core convolution and pooling operations are implemented from scratch, instead of using PyTorch’s built-in nn.Conv2d and nn.MaxPool2d.

📂 File Descriptions

*conv2d_from_scratch.py*

This file implements a custom 2D convolution layer.

Defines the class Conv2DFromScratch, which behaves similarly to nn.Conv2d

Manually performs convolution using nested loops over:

batch dimension

output channels (filters)

spatial height and width

Supports:  multiple input/output channels

configurable kernel size, stride, and padding

Stores convolution kernels (weight) and bias (bias) as trainable parameters

Uses explicit region-wise multiplication and summation to compute feature maps

*maxpool2d_from_scratch.py*

This file implements a custom 2D max-pooling layer.

Defines the class MaxPool2DFromScratch

Applies spatial down-sampling by:

sliding a window over the input feature map

selecting the maximum value in each window

Supports configurable kernel size and stride

Implemented using explicit loops to make the pooling operation transparent

*CNN_MNIST_From_Scratch.ipynb*

This notebook ties everything together and performs training and evaluation on the MNIST dataset.

Loads the MNIST dataset using torchvision

Defines a CNN architecture using:

Conv2DFromScratch

MaxPool2DFromScratch

ReLU activation

Fully connected layers

Trains the model using:

Cross-entropy loss

Adam optimizer

Evaluates classification performance on test data

---

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- NumPy
- PyTorch
- Jupyter Notebook

### Installation
```bash
# Clone the repository
git clone https://github.com/whitespace-24/Wids2025-DeepLearningFromScratch.git

# Navigate to repository
cd Wids2025-DeepLearningFromScratch

# Install required packages into virtualenv

```

### Running Notebooks
```bash
jupyter notebook
# Open and run notebooks from the respective week folders
```

---

## 🔗 Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Kaggle Competitions](https://www.kaggle.com/competitions)

---

## 👤 Author

Created as part of the WIDS 2025 program by whitespace-24


---

**Last Updated**: January 2026
