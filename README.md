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

📂 Week 3: CNN Fundamentals with PyTorch

In this phase, we explore the power of CNNs for computer vision, focusing on preserving the spatial relationship of pixels.

    Dataset: Utilizes the MNIST dataset, a collection of 70,000 handwritten digits.

    Spatial Awareness: Unlike standard DNNs that flatten images, our CNN treats images as 28×28 pixel grids to preserve 2D structure.

    Core Concepts Implemented:

        Convolutional Layers (Conv2D): Uses filters (kernels) to learn local patterns like edges and curves through Parameter Sharing.

        Pooling Layers (MaxPool): Performs downsampling (typically 2×2) to reduce computational load and create invariance to small distortions.

🚀 Featured File: SimpleCNN_MNIST.ipynb

A practical implementation of a CNN using the PyTorch framework to establish a performance baseline on the MNIST dataset.
🛠️ Weeks 4 & 5: CNN Architecture From Scratch

The core of this project involves deconstructing the convolution and pooling operations. We implement these layers manually to understand the underlying mathematics, bypassing PyTorch’s built-in nn.Conv2d and nn.MaxPool2d.
📂 File Descriptions
1. conv2d_from_scratch.py

This file defines the Conv2DFromScratch class, mimicking the behavior of standard layers through manual computation.

    Manual Convolution: Implements the operation using nested loops over the batch, output channels, and spatial dimensions.

    Features: Supports multiple input/output channels, configurable kernel size, stride, and padding.

    Trainable Parameters: Manually manages convolution kernels (weights) and biases.

    Logic: Uses explicit region-wise multiplication and summation to generate feature maps.

2. maxpool2d_from_scratch.py

Defines the MaxPool2DFromScratch class for spatial downsampling.

    Operation: Slides a window over the input feature map and selects the maximum value.

    Transparency: Implemented with explicit loops to make the pooling operation fully transparent.

    Configurability: Supports custom kernel sizes and strides.

3. CNN_MNIST_From_Scratch.ipynb

The master notebook that integrates our custom-built layers for full-scale training and evaluation.

    Architecture: Combines Conv2DFromScratch, MaxPool2DFromScratch, ReLU activations, and fully connected layers.

    Training: Optimized using Adam and Cross-entropy loss.

    Workflow: Loads MNIST via torchvision, trains the custom model, and evaluates classification performance on test data.
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
