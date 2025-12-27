# Supervised Learning Neural Network Library

A custom-built supervised learning neural network library created from scratch in **C#**, designed for flexibility, performance, and educational value. This library not only implements essential neural network components but also handles key data preprocessing tasks such as one-hot encoding, dataset splitting, and missing feature checks. 

With functionality aimed at both performance and understanding the inner workings of machine learning, the library achieves **93.1% accuracy on the MNIST test set**. While functional, further optimization is underway to improve training time performance.

IMPORTANT: As mentioned above, this project was more of a learning experience and as a way for me to transfer Python neural networks to C# required by Unity; My library allows for the creation of custom neural network architectures and instead of using a Unity Plugin to allow Python ML models to run in Unity, I just put the weights and biases of the Python model into a custom C# model (seen in my Digit Recognizer project). For this reason, little UI exists for the library and for true training of models, it would be beneficial to use NumPy with tensors for quick matrix operations, and Matplotlib and scikit-learn to observe the metrics of the model as it learns, which this library does not use since I enjoyed having a "full" supervised neural network library created entirely (not including standard library DSA) from scratch.

---

## Features

### Neural Network Core:
- **Backpropagation**: Efficient implementation for updating weights and biases.
- **Activation Functions**: Supports common functions like ReLU, Sigmoid, and Softmax.
- **Optimization**: Implements gradient descent with scope for additional algorithms.
- **Normalization**: Facilitates stable and efficient training.

### Data Handling:
- **One-Hot Encoding**: Converts categorical labels into numerical arrays.
- **Dataset Splitting**: Automates training and testing data partitioning.
- **Missing Feature Handling**: Detects and addresses gaps in input data.

### Additional Functionality:
- **Weight and Bias Persistence**: Save and load model parameters using CSV for reusability.
- **Matrix Operations**: Efficient internal handling of matrix multiplication and related operations.

---

## Performance

- Trained and tested on the **MNIST dataset**, achieving:
  - **Accuracy**: 93.1% on the test set
  - **Optimization**: Currently undergoing performance tuning to improve training efficiency

---

## Purpose and Learning Outcomes

This project was created to:
1. **Understand Neural Network Mechanics**: Gain hands-on experience with backpropagation, optimization algorithms, and matrix manipulation.
2. **Develop Data Processing Skills**: Build tools akin to Python's Pandas for handling and preprocessing datasets efficiently in C#.
3. **Enhance AI Development for Unity**: Provide a robust foundation for integrating machine learning into Unity-based games and simulations.
4. **Build Modular AI Components**: Enable saving and loading model parameters, improving workflows for AI experimentation.

---

## Applications

The library is ideal for:
- Learning and experimenting with neural network fundamentals.
- Developing AI systems for Unity games and simulations.
- Handling data preprocessing and neural network training in standalone C# applications.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yckamra/MLLibrary.git
