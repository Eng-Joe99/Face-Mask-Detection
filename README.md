# Face-Mask-Detection


## Overview

In response to the global COVID-19 pandemic and the widespread enforcement of mask mandates, this project demonstrates a deep learning approach for detecting faces with and without masks. Utilizing a custom Convolutional Neural Network (CNN) model trained on 7,553 RGB images, the project achieves a training accuracy of 94% and a validation accuracy of 92%.

## Dataset
[Dataset Link](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset/data)
<br>The dataset consists of 7,553 images divided into two folders:
- **With Mask:** 3,725 images
- **Without Mask:** 3,828 images

Each image is in RGB format and has been preprocessed to ensure compatibility with the models.

## Features

The project is implemented in Python and executed on Google Colab. It includes the following features:

### **Core Methods**
1. **`load_data(self, path1, path2)`**
   - Loads the dataset from the specified paths.
   - Preprocesses images through resizing, color conversion (BGR to RGB), normalization, and reshaping.
   - Generates corresponding labels for the dataset.

2. **`visualize_results(self)`**
   - Visualizes sample labeled data.
   - Plots learning curves (e.g., accuracy and loss) for neural network models if training history is available.

3. **`plot_confusion_matrix(self, y_true, y_pred, title)`**
   - Generates a confusion matrix to evaluate model performance, displaying true positive, false positive, true negative, and false negative predictions.

4. **`run_complete_analysis(self)`**
   - Executes the entire pipeline, including data visualization, training, and evaluation of all models.

### **Implemented Models**
1. **K-Nearest Neighbors (KNN)**
   - Uses `KNeighborsClassifier` from `sklearn.neighbors` to classify images based on proximity in feature space.
   - Configurable `n_neighbors` parameter.

2. **Logistic Regression**
   - Implements logistic regression using `LogisticRegression` from `sklearn.linear_model`.

3. **Support Vector Machine (SVM)**
   - Trains an SVM classifier with `SVC` from `sklearn.svm`.

4. **Custom CNN**
   - A deep learning model built using Keras' Sequential API.
   - Architecture:
     - Two convolutional layers with ReLU activation and max pooling.
     - Fully connected layers with dropout regularization.
     - Sigmoid activation in the output layer for binary classification.
   - Achieves high accuracy with metrics calculated for precision, recall, and F1-score.

## Implementation Steps

1. **Import Libraries**
   - Essential libraries like `os`, `cv2`, `numpy`, `sklearn`, and `tensorflow.keras` are utilized.

2. **Load and Preprocess Data**
   - Images are read, resized, normalized, and labeled.

3. **Visualize Results**
   - Displays dataset samples and model training history.

4. **Train Models**
   - KNN, Logistic Regression, SVM, and CNN models are trained and evaluated.

5. **Evaluate Models**
   - Performance metrics such as accuracy, precision, recall, and F1-score are computed.
   - Confusion matrices are plotted for deeper insights.

## Results

- **Training Accuracy (CNN):** 94%
- **Validation Accuracy (CNN):** 96%
- Confusion matrices demonstrate strong performance across models.

## How to Run

1. Clone this repository.
   ```bash
   git clone https://github.com/your-repo/face-mask-detection.git
   ```
2. Upload the dataset (`with_mask` and `without_mask` folders) to your environment.
3. Open the project in Google Colab or any Python IDE.
4. Run the `main()` method to execute the pipeline.

## Team Members

- **Youssef Ahmed Khalil**
- **Eslam Saad Gomaa**

