# Digit Recognition Using ANN on Keras MNIST Dataset
## Objective:
Recognize handwritten digits using an Artificial Neural Network (ANN) on the MNIST dataset.

## Dataset:
- MNIST: 70,000 grayscale images (28x28 pixels) of digits (0-9).
- Model:
-- Input Layer: Flatten 28x28 images to 1D.
-- Hidden Layer: Dense layer with 128 neurons, ReLU activation.
-- Output Layer: 10 classes, softmax activation.
- Training:
-- Optimizer: Adam
-- Loss Function: Sparse categorical cross-entropy
-- Batch Size: 64
-- Epochs: 10
- Results:
-- Training Accuracy: 99%
-- Validation Accuracy: 97.6%
-- Test Accuracy: 97.45%
- Conclusion:
The model successfully recognizes handwritten digits with high accuracy.
