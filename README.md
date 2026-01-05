# Wids2025-DeepLearningFromScratch
Week 1: Foundations & Tooling Setup

Week 1 focused on building the mathematical intuition and tooling familiarity required for the project, along with an introduction to Kaggle-style machine learning workflows.

🔹 Linear Algebra Refresher

Emphasized the importance of linear algebra intuition for understanding machine learning models.

Covered core ideas such as:

Vectors as points and directions

Linear transformations

Geometric interpretation of matrix operations

Recommended the 3Blue1Brown linear algebra series, with focus on the first lesson.

🔹 NumPy & PyTorch Basics

Introduced PyTorch tensors as the fundamental data structure for deep learning.

Learned how tensors relate to NumPy arrays and how they differ.

Focused on:

Tensor creation and shapes

Basic operations and broadcasting

Indexing and slicing

🔹 Distance Computations with NumPy

Studied efficient computation of distance matrices using NumPy.

Learned how vectorization avoids slow Python loops.

Built intuition for how mathematical expressions translate into fast numerical code.

This directly prepares participants for the upcoming Kaggle contest tasks.

🔹 Kaggle Introduction

Introduced Kaggle competitions as a practical ML workflow.

Explored:

Dataset structure (train.csv, test.csv)

Submission format

Public vs private leaderboard

Used the Titanic dataset to understand how ML contests work end-to-end.

Week 2: Perceptrons & Neural Networks (Mathematical Foundations)

In Week 2, we focused on building a mathematical understanding of perceptrons and neural networks, starting from simple linear classifiers and gradually motivating deeper architectures and learning algorithms.

🔹 Perceptron

Introduced the perceptron as a basic linear binary classifier.

Understood how inputs, weights, bias, and activation function work together.

Interpreted the perceptron as learning a linear decision boundary in feature space.

Discussed the limitations of perceptrons, especially their inability to solve non-linearly separable problems.

🔹 XOR Problem

Studied the XOR problem as a classic example where a single-layer perceptron fails.

Used XOR to motivate the need for multiple layers and non-linear transformations.

Gained geometric intuition behind why XOR cannot be separated by a single line.

🔹 Neural Networks (Multi-Layer Perceptrons)

Extended perceptrons to multi-layer neural networks.

Understood the role of:

Hidden layers

Non-linear activation functions

Saw how stacking layers enables learning complex, non-linear decision boundaries.

🔹 Backpropagation

Introduced backpropagation as the core learning algorithm for neural networks.

Studied how gradients are computed using the chain rule.

Understood how weights and biases are updated to minimize loss.

Emphasis was placed on the mathematical derivation, not just implementation.

🔹 Automatic Differentiation (Optional)

Explored how modern ML libraries compute gradients efficiently using computational graphs and automatic differentiation.

Connected the theory of backpropagation to how frameworks like TensorFlow and PyTorch work internally.

🔹 Interactive Exploration

Used TensorFlow Playground to visually experiment with:

Network depth

Activation functions

Decision boundaries

Observed how architectural changes affect learning behavior.
