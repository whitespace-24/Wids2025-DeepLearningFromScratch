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
🗓️ Week 3: Transitioning to Image Data with CNNs

In this week, the repository explores Convolutional Neural Networks (CNNs), which are specifically optimized for computer vision tasks such as handwritten digit recognition.

📌 Context

Similar to the previous K-Nearest Neighbors (KNN) implementation, this project uses the MNIST dataset, consisting of 70,000 handwritten digit images (28×28 pixels).

🧠 Pre-processing

Images are treated as 2D grids (28×28 pixels) rather than flattened vectors.

Unlike basic Deep Neural Networks (DNNs), CNNs preserve spatial structure, enabling the model to learn meaningful local patterns such as edges, curves, and shapes.

🔑 Key CNN Components Implemented

To understand how spatial features are captured, the following core CNN concepts are emphasized:

🔹 Convolutional Layers (Conv2D)

Use filters (kernels) that slide across the image instead of fully connected weights

Learn local patterns such as edges, loops, and curves

Employ parameter sharing, significantly reducing the number of trainable parameters

Preserve spatial relationships within the image

🔹 Pooling Layers (MaxPool)

Perform down-sampling by selecting the maximum value in local windows (e.g., 2×2)

Reduce computational complexity

Introduce translation invariance, making feature detection robust to small shifts or distortions

📘 Implementation (Week 3)

SimpleCNN_MNIST.ipynb
A practical implementation of a CNN using the PyTorch framework on the MNIST dataset, demonstrating standard convolution and pooling layers.

🚀 Project (Weeks 4 & 5): CNN from Scratch

This project extends the CNN implementation by manually implementing convolution and pooling operations from scratch, without using PyTorch’s built-in nn.Conv2d and nn.MaxPool2d layers.

The objective is to gain a deeper understanding of the internal mechanics of CNNs while still leveraging PyTorch for automatic differentiation and optimization.

📂 File Descriptions
conv2d_from_scratch.py

Implements a custom 2D convolution layer.

Defines the class Conv2DFromScratch, functionally similar to nn.Conv2d

Manually performs convolution using explicit nested loops over:

Batch dimension

Output channels (filters)

Spatial height and width

Supports:

Multiple input and output channels

Configurable kernel size, stride, and padding

Stores convolution kernels (weight) and biases (bias) as trainable parameters

Computes feature maps via region-wise multiplication and summation

maxpool2d_from_scratch.py

Implements a custom 2D max-pooling layer.

Defines the class MaxPool2DFromScratch

Applies spatial down-sampling by:

Sliding a window across the feature map

Selecting the maximum value within each window

Supports configurable kernel size and stride

Implemented using explicit loops for full transparency of the pooling operation

CNN_MNIST_From_Scratch.ipynb

Integrates all components and performs training and evaluation on the MNIST dataset.

Loads the MNIST dataset using torchvision

Defines a CNN architecture composed of:

Conv2DFromScratch

MaxPool2DFromScratch

ReLU activation functions

Fully connected layers

Trains the network using:

Cross-entropy loss

Adam optimizer

Evaluates classification accuracy on the test dataset

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
