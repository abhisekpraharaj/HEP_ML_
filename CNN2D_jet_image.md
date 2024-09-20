CNN for Jet Images
This project implements a Convolutional Neural Network (CNN) to analyze jet images in high-energy physics datasets. The goal is to classify jets based on the image representations of their internal structure, using Keras and TensorFlow for the model's design and training. The dataset consists of jet images stored in HDF5 format.

Table of Contents
Part 1: Dataset Preparation and Exploration
Part 2: Model Architecture
Part 3: Training the Model
Part 4: Evaluation and Analysis
Part 5: Conclusion
Part 1: Dataset Preparation and Exploration
Dataset:

The jet image data is stored in .h5 files in the JetDataset directory. Each file contains images of jets as 2D arrays, where each pixel represents the energy deposited in a certain region of space.

Exploration:

Before proceeding to model training, the jet images were explored for basic properties such as their size, pixel intensity distribution, and the total number of jets per class. This helped understand the structure of the data.

Data Loading:

The dataset was loaded and split into training and validation sets using the train_test_split function from sklearn. This split ensures the model's performance is validated during training to avoid overfitting.

Part 2: Model Architecture
The CNN architecture designed for jet image classification includes:

Input Layer:

The input to the model consists of jet images represented as 2D arrays with dimensions corresponding to the number of pixels (e.g., 100x100).

Convolutional Layers:

Several 2D convolutional layers with filters of increasing depth were used. These layers help detect local patterns in the jet images, such as high-energy deposits.

Pooling Layers:

Max pooling layers were applied after the convolutional layers to reduce the spatial dimensions of the feature maps and prevent overfitting.

Dense Layers:

After flattening the pooled feature maps, fully connected layers were added, with the final layer being a softmax layer for multi-class classification of the jet images.

Activation Functions:

Rectified Linear Units (ReLU) were used in convolutional and dense layers, while the softmax function was used for the output layer to classify the jet images into multiple categories.

Part 3: Training the Model
Data Augmentation:

To improve generalization, data augmentation techniques such as random rotations and flips were applied to the jet images.

Loss Function:

Categorical cross-entropy was used as the loss function since this is a multi-class classification problem.

Optimizer:

The Adam optimizer was used with a learning rate of 0.001 to update the model weights during training.

Batch Size and Epochs:

The model was trained with a batch size of 64 over 50 epochs. Early stopping was employed to halt the training process if the validation accuracy stopped improving.

Part 4: Evaluation and Analysis
Validation:

After training, the model was evaluated using the validation dataset. Various metrics, such as accuracy, precision, and recall, were computed to assess the model's performance.

Confusion Matrix:

A confusion matrix was generated to visually inspect the performance of the model across different jet classes, helping identify any misclassifications.

Training Visualization:

Training and validation accuracy and loss were plotted to track the model's progress. This also helped check whether the model was overfitting or underfitting.

Part 5: Conclusion
The CNN model for jet image classification achieved a validation accuracy of over 80%, demonstrating that the network can successfully learn from jet images. Potential future improvements could include deeper architectures or the application of transfer learning techniques.
