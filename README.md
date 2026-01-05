# WIDS 2025: Deep Learning From Scratch

A comprehensive course covering the mathematical foundations and implementation of deep learning models from first principles.

## Course Overview

This course builds a complete understanding of deep learning by starting with mathematical fundamentals and progressively implementing neural networks without relying on high-level abstractions. Students will gain both theoretical knowledge and practical coding skills through hands-on projects and Kaggle-style competitions.

---

## Week 1: Mathematical Foundations & ML Workflows

### Learning Objectives

Establish the mathematical and coding foundations required for the project, along with an introduction to Kaggle-style machine learning workflows.

### 🔹 Linear Algebra Intuition

- Importance of linear algebra in machine learning
- Vectors as points and directions
- Linear transformations and geometric meaning of matrices

**Resources:**
- [3Blue1Brown Linear Algebra Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (focus on the first lesson)

### 🔹 NumPy & PyTorch Basics

- Introduction to PyTorch tensors and their relation to NumPy arrays
- Tensor creation, shapes, indexing, and slicing
- Basic operations and broadcasting

### 🔹 Distance Computations

- Efficient computation of distance matrices using NumPy
- Understanding vectorization vs Python loops
- Useful preparation for Kaggle-style problems

### 🔹 Kaggle Introduction

- Overview of Kaggle competitions and workflows
- Understanding datasets (train.csv, test.csv) and submissions
- Explored the Titanic dataset as a first contest

---

## Week 2: Perceptrons & Neural Networks

### Learning Objectives

Understand how neural networks work from a mathematical perspective, starting from simple linear models and progressing to multi-layer architectures.

### 🔹 Perceptron

- Perceptron as a linear binary classifier
- Role of weights, bias, and activation functions
- Learning a linear decision boundary
- Limitations of linear models

### 🔹 XOR Problem

- Why a single-layer perceptron fails on XOR
- Motivation for multi-layer networks
- Geometric intuition behind non-linear separability

### 🔹 Neural Networks (MLPs)

- Extending perceptrons to multi-layer neural networks
- Role of hidden layers and non-linear activations
- Learning complex decision boundaries

### 🔹 Backpropagation

- Backpropagation as the core learning algorithm
- Gradient computation using the chain rule
- Weight and bias updates to minimize loss
- Focus on mathematical understanding

### 🔹 Automatic Differentiation (Optional)

- How modern frameworks compute gradients
- Connection between backpropagation and computational graphs

### 🔹 Interactive Learning

Used [TensorFlow Playground](https://playground.tensorflow.org) to visualize:
- Network depth and width effects
- Activation functions and their properties
- Decision boundaries in different configurations
- How network architecture affects learning behavior

---

## Prerequisites

- Python 3.8+
- NumPy
- PyTorch
- TensorFlow (for visualization with Playground)

## Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/WIDS2025-DeepLearningFromScratch.git
cd WIDS2025-DeepLearningFromScratch

# Install dependencies
pip install numpy pytorch tensorflow
```

---

## Project Structure

```
WIDS2025-DeepLearningFromScratch/
├── Week1/
│   ├── LinearAlgebra/
│   ├── NumPyPyTorch/
│   ├── DistanceComputations/
│   └── KaggleIntro/
├── Week2/
│   ├── Perceptron/
│   ├── NeuralNetworks/
│   └── Backpropagation/
└── README.md
```

---

## Key Concepts

| Concept | Week | Description |
|---------|------|-------------|
| Linear Algebra | 1 | Vectors, matrices, and transformations |
| Vectorization | 1 | Efficient NumPy/PyTorch operations |
| Distance Metrics | 1 | Euclidean and other distance computations |
| Perceptron | 2 | Single-layer linear classifier |
| Multi-layer Networks | 2 | MLPs with hidden layers |
| Backpropagation | 2 | Gradient-based learning algorithm |
| Activation Functions | 2 | Non-linear transformations in networks |

---

## Resources

- **3Blue1Brown**: Linear Algebra Essentials
- **TensorFlow Playground**: Interactive neural network visualization
- **Kaggle**: Datasets and competition experience
- **PyTorch Documentation**: https://pytorch.org/docs/stable/index.html
- **NumPy Documentation**: https://numpy.org/doc/

---

## Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- 3Blue1Brown for exceptional mathematical intuition videos
- Kaggle for providing datasets and competition platform
- TensorFlow team for the interactive Playground tool
