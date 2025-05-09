# ✍️ Handwritten Digit Recognition with CNN

This project implements a Convolutional Neural Network (CNN) for recognizing handwritten digits. It is trained on the widely-used MNIST dataset and includes an interactive Graphical User Interface (GUI) where users can draw a digit and get real-time predictions from the trained model.

## ✨ Features

* **CNN Model:** A robust CNN architecture designed for image classification tasks.
* **MNIST Dataset:** Utilizes the standard MNIST dataset for training and evaluation.
* **Data Preprocessing:** Includes steps for normalization, reshaping, and one-hot encoding.
* **Model Training:** Trains the CNN model with the Adam optimizer and categorical cross-entropy loss.
* **Comprehensive Evaluation:** Evaluates the model's performance using accuracy, classification reports, and confusion matrices.
* **Interactive GUI:** A Tkinter-based application allowing users to draw a digit on a canvas and get predictions from the trained model.

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* Python 3.6+
* Jupyter Notebook (or JupyterLab)
* Required Python libraries (listed in Requirements)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/indranil143/Digit_Recognition.git
    cd Digit_Recognition
    ```

2.  Install the required libraries. It's recommended to use a virtual environment:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Project

1.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook "Handwritten Digit Recognition with CNN.ipynb"
    ```
2.  Run all the cells in the notebook sequentially.
3.  The notebook will perform the following steps:
    * Load and preprocess the MNIST data.
    * Define, compile, and train the CNN model.
    * Evaluate the trained model.
    * Load the saved model (or attempt to train if no saved model is found).
    * Launch the interactive GUI application.

## 📊 Model Performance

The trained CNN model achieved excellent performance on the MNIST test dataset:

* **Test Accuracy:** 99.38%
* **Test Loss:** 0.0361

A detailed classification report shows high precision, recall, and f1-scores for each digit class (mostly 0.99 and 1.00).

## 💻 Usage (GUI Application)

Once the GUI application launches after running the notebook:

1.  Use your mouse to draw a digit on the black canvas.
2.  Click the "Predict" button to get the model's prediction and the associated probability.
3.  Click the "Clear" button to erase the canvas and draw a new digit.

## 📁 Project Structure

* `Handwritten Digit Recognition with CNN.ipynb`: The main Jupyter Notebook containing all the code for data loading, preprocessing, model building, training, evaluation, and the GUI implementation.
* `MNIST_Digit_Recognition_Model.ipynb`: This is the old project.

## ✅ Requirements

The project requires the following Python libraries:

* `numpy`
* `matplotlib`
* `tensorflow`
* `keras`
* `seaborn`
* `scikit-learn` (for `classification_report`, `confusion_matrix`)
* `Pillow` (PIL)
* `tkinter` (usually comes with Python)

# Also Checkout my previous project on ths topic
## Digit Recognition Using ANN (Previous Version)

- Objective: Recognize digits using an Artificial Neural Network (ANN) on the MNIST dataset.
- Dataset: MNIST, 70,000 grayscale images (28x28 pixels) of digits (0-9).
- Model:
  - Input Layer: Flatten 28x28 images to 1D.
  - Hidden Layer: Dense layer with 128 neurons, ReLU activation.
  - Output Layer: 10 classes, softmax activation.
- Training:
  - Optimizer: Adam
  - Loss Function: Sparse categorical cross-entropy
  - Batch Size: 64
  - Epochs: 10
- Results:
  - Training Accuracy: 99%
  - Validation Accuracy: 97.6%
  - Test Accuracy: 97.45%
- Conclusion: The model successfully recognizes digits with high accuracy.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please go ahead!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

* The MNIST dataset for providing the training data.
* The TensorFlow and Keras teams for providing the deep learning framework.
* The developers of the other open-source libraries used in this project.

---
© 2025 Indranil143
