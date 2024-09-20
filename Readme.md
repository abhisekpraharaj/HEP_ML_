# CNN for Jet Constituents
This project implements a **Convolutional Neural Network (CNN)** designed to analyze jet constituents in high-energy physics datasets. The goal is to perform classification using jet particle features and explore how well CNNs can distinguish between different jet structures by leveraging constituent-level information. The project is implemented using **TensorFlow** and utilizes particle feature datasets stored in **HDF5** format.
## Table of Contents
1. [Part 1: Dataset Preparation and Exploration](#part-1-dataset-preparation-and-exploration)
1. [Part 2: Model Architecture](#part-2-model-architecture)
1. [Part 3: Training the Model](#part-3-training-the-model)
1. [Part 4: Evaluation and Analysis](#part-4-evaluation-and-analysis)
1. [Part 5: Conclusion](#part-5-conclusion)
## Part 1: Dataset Preparation and Exploration
- **Dataset:**

  The jet constituent data is stored in `.h5` files within the JetDataset directory. Each dataset contains information for 50,000 jets, with up to 100 particles per jet. For each particle, 16 features are provided, including momentum components (*px*, *py*, *pz*), energy, and relative angular information.
- **Exploration:**

  We explored basic dataset properties, including particle multiplicity per jet, momentum distribution, and energy patterns. This helped us understand the overall dataset characteristics before feeding it into the CNN.
## Part 2: Model Architecture
The CNN architecture used for jet constituent classification includes:

- **Input Layer:**

  The input shape is (100, 16), corresponding to 100 particles and their 16 features.
- **Convolutional Layers:**

  Several 1D convolutional layers are applied to the constituent-level feature matrix, capturing patterns across the particle feature space.
- **Pooling Layers:**

  Max pooling is used to down-sample the data and reduce computational complexity while retaining the most important information.
- **Dense Layers:**

  After flattening, the convolutional layers are followed by fully connected (dense) layers, leading to a softmax output for classification.
- **Activation Functions:**

  ReLU activations are used in convolutional and dense layers. The final layer uses a softmax function for multi-class classification.
## Part 3: Training the Model
- **Data Augmentation:**

  To enhance model generalization, various data augmentation techniques were applied, including random rotations and translations of jet constituents.
- **Loss Function:**

  The categorical cross-entropy loss function was employed to measure the model's classification performance.
- **Optimizer:**

  Adam optimizer was used with a learning rate of 0.001 for faster convergence.
- **Batch Size and Epochs:**

  The model was trained using a batch size of 64 for 50 epochs, with early stopping to prevent overfitting.
## Part 4: Evaluation and Analysis
- **Validation:**

  The model's performance was evaluated using a validation set. Metrics such as accuracy, precision, recall, and F1-score were computed.
- **Confusion Matrix:**

  A confusion matrix was generated to visualize the model's classification performance across different jet classes.
- **Visualization:**

  Plots of training vs. validation loss and accuracy were generated to ensure that the model is not overfitting and that training proceeds smoothly.
## Part 5: Conclusion
The CNN-based approach for jet constituent classification shows promising results, achieving a validation accuracy of over 85%. Future improvements may include experimenting with deeper network architectures and applying advanced techniques like **attention mechanisms** to better capture the hierarchical structure of jets.
## References
- **Jet Dataset:**

  The dataset used for training is available in the public HDF5 format.
- **TensorFlow Documentation:**

  Refer to the official [TensorFlow documentation](https://www.tensorflow.org/) for more details on the framework and tools used.
# CNN for Jet Images
This project implements a **Convolutional Neural Network (CNN)** to analyze jet images in high-energy physics datasets. The goal is to classify jets based on the image representations of their internal structure, using **Keras** and **TensorFlow** for the model's design and training. The dataset consists of jet images stored in **HDF5** format.
## Table of Contents
1. [Part 1: Dataset Preparation and Exploration](#part-1-dataset-preparation-and-exploration)
1. [Part 2: Model Architecture](#part-2-model-architecture)
1. [Part 3: Training the Model](#part-3-training-the-model)
1. [Part 4: Evaluation and Analysis](#part-4-evaluation-and-analysis)
1. [Part 5: Conclusion](#part-5-conclusion)
## Part 1: Dataset Preparation and Exploration
- **Dataset:**

  The jet image data is stored in `.h5` files in the JetDataset directory. Each file contains images of jets as 2D arrays, where each pixel represents the energy deposited in a certain region of space.
- **Exploration:**

  Before proceeding to model training, the jet images were explored for basic properties such as their size, pixel intensity distribution, and the total number of jets per class. This helped understand the structure of the data.
- **Data Loading:**

  The dataset was loaded and split into training and validation sets using the `train_test_split` function from `sklearn`. This split ensures the model's performance is validated during training to avoid overfitting.
## Part 2: Model Architecture
The CNN architecture designed for jet image classification includes:

- **Input Layer:**

  The input to the model consists of jet images represented as 2D arrays with dimensions corresponding to the number of pixels (e.g., 100x100).
- **Convolutional Layers:**

  Several 2D convolutional layers with filters of increasing depth were used. These layers help detect local patterns in the jet images, such as high-energy deposits.
- **Pooling Layers:**

  Max pooling layers were applied after the convolutional layers to reduce the spatial dimensions of the feature maps and prevent overfitting.
- **Dense Layers:**

  After flattening the pooled feature maps, fully connected layers were added, with the final layer being a softmax layer for multi-class classification of the jet images.
- **Activation Functions:**

  Rectified Linear Units (ReLU) were used in convolutional and dense layers, while the softmax function was used for the output layer to classify the jet images into multiple categories.
## Part 3: Training the Model
- **Data Augmentation:**

  To improve generalization, data augmentation techniques such as random rotations and flips were applied to the jet images.
- **Loss Function:**

  Categorical cross-entropy was used as the loss function since this is a multi-class classification problem.
- **Optimizer:**

  The Adam optimizer was used with a learning rate of 0.001 to update the model weights during training.
- **Batch Size and Epochs:**

  The model was trained with a batch size of 64 over 50 epochs. Early stopping was employed to halt the training process if the validation accuracy stopped improving.
## Part 4: Evaluation and Analysis
- **Validation:**

  After training, the model was evaluated using the validation dataset. Various metrics, such as accuracy, precision, and recall, were computed to assess the model's performance.
- **Confusion Matrix:**

  A confusion matrix was generated to visually inspect the performance of the model across different jet classes, helping identify any misclassifications.
- **Training Visualization:**

  Training and validation accuracy and loss were plotted to track the model's progress. This also helped check whether the model was overfitting or underfitting.
## Part 5: Conclusion
The CNN model for jet image classification achieved a validation accuracy of over 80%, demonstrating that the network can successfully learn from jet images. Potential future improvements could include deeper architectures or the application of transfer learning techniques.

Sure! Here's a `README.md` file for your project:
~~~ markdown
# Jet Classification Using Deep Learning

This project demonstrates the process of using deep learning techniques to classify jet images from high-energy physics experiments.

## Dataset

The dataset used for this analysis is available in `.h5` format and consists of multiple files. Each file contains jet data with various features and corresponding labels. The script processes a selection of these files to extract relevant features and targets for training.

## Data Preparation

### Loading Libraries

The code begins by importing required libraries such as `h5py` for handling HDF5 files, `numpy` for numerical operations, and `matplotlib` for plotting.

### Cloning the Dataset

The code attempts to clone the dataset from a GitHub repository. If the repository already exists, it skips this step.

### Defining Targets and Features

The script initializes empty numpy arrays for target and features. It then specifies the files to be processed.

### Appending Data

For each specified file, it opens the HDF5 file, extracts the features and targets, and appends them to the respective arrays. The features include 16 attributes related to the jets, while the targets contain labels for each jet.

### Data Shape

After processing, the shapes of the features and targets arrays are printed to confirm the dimensions (50,000 jets, each with 16 features).

## Data Splitting

The dataset is split into training and validation sets using `train_test_split` from `scikit-learn`. The training set contains 67% of the data, while 33% is reserved for validation. This split ensures the model is evaluated on unseen data during training.

## Model Building

### Model Architecture

A Dense Neural Network is constructed using Keras. The model consists of:

- An input layer with shape corresponding to the feature size.
- Several dense layers with ReLU activation functions.
- Dropout layers to prevent overfitting.
- An output layer using softmax activation to output probabilities for the 5 jet classes (gluon, quark, W, Z, top).

### Compilation

The model is compiled with categorical cross-entropy loss and the Adam optimizer.

## Training the Model

The model is trained on the training data for a maximum of 50 epochs. Early stopping, learning rate reduction, and termination on NaN are used as callbacks to enhance training stability and efficiency.

The training progress is printed for each epoch, showing the training and validation loss.

## Evaluation

After training, the model's performance is evaluated using Receiver Operating Characteristic (ROC) curves for each class. The curves are plotted to visualize the trade-off between sensitivity and specificity, and the area under the curve (AUC) is calculated.

## Visualizing Training History

The training and validation loss are plotted on a logarithmic scale to visualize the learning process. This helps in understanding how well the model is converging.

## Conclusion

This project demonstrates the process of using deep learning techniques to classify jet images from high-energy physics experiments. The model can be further refined and evaluated with more advanced techniques and larger datasets.
~~~

Feel free to customize it further based on your specific needs! If you need any more details or adjustments, let me know.
# Document Overview: Graph Neural Network Analysis
## Introduction
This document provides a detailed overview of the Graph Neural Network (GNN) analysis conducted by Abhisek Praharaj. The analysis focuses on leveraging GNNs to explore and model graph-structured data, which is increasingly relevant in various domains such as social networks, biological networks, and recommendation systems.
## Purpose of the Analysis
The primary goal of the GNN analysis is to utilize the capabilities of graph neural networks to effectively capture relationships and interactions within graph data. By doing so, the analysis aims to improve predictive performance on tasks that involve complex relational structures.
## Key Components of the GNN Analysis
### 1\. Graph Representation
- **Data Transformation**: The dataset is transformed into a graph format, where:
  - **Nodes** represent entities (e.g., users, items).
  - **Edges** represent relationships or interactions between these entities (e.g., friendships, connections).
# Graph Neural Network Architecture Overview
## Introduction
This document provides a streamlined overview of the architecture used in the Graph Neural Network (GNN) analysis conducted by Abhisek Praharaj. The architecture is designed to effectively process graph-structured data, capturing relationships and interactions among entities.
## GNN Architecture Components
### 1\. **Input Layer**
- **Graph Representation**: The input to the GNN consists of a graph where:
  - **Nodes** represent entities (e.g., users, items).
  - **Edges** represent relationships between these entities (e.g., connections, interactions).
### 2\. **Message Passing Layers**
- **Information Propagation**: The core of the GNN architecture involves multiple layers that facilitate message passing between nodes. Each node aggregates information from its neighbors to update its own representation. This process typically involves:
  - **Neighbor Aggregation**: Each node collects features from its neighboring nodes.
  - **Update Function**: A function (often a neural network) updates the node's feature vector based on the aggregated information.
### 3\. **Activation Functions**
- **Non-linearity**: After each message passing layer, activation functions (such as ReLU or Sigmoid) are applied to introduce non-linearity into the model, allowing it to learn complex patterns in the data.
### 4\. **Readout Layer**
- **Global Representation**: After several rounds of message passing, a readout layer is employed to generate a global representation of the graph or individual node representations for downstream tasks. This can involve:
  - Summing or averaging node features.
  - Using a more complex pooling mechanism.
### 5\. **Output Layer**
- **Prediction**: The final output layer produces predictions based on the learned representations. This could be for tasks such as node classification, link prediction, or graph classification.
### 3\. Training Process
- **Training Setup**: The model is trained using labeled data. Key aspects include:
  - Selection of loss functions to optimize model performance.
  - Use of optimization algorithms to adjust model parameters during training.
### 4\. Evaluation Metrics
- **Performance Assessment**: The trained model is evaluated on a test dataset to measure its effectiveness. Common metrics may include:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
### 5\. Results Interpretation
- **Outcome Analysis**: The results from the GNN analysis are interpreted to understand:
  - How well the model captured relationships within the graph.
  - Insights gained regarding the effectiveness of GNNs for the specific application at hand.
## Conclusion
This document summarizes the essential components and methodologies employed in the Graph Neural Network analysis by Abhisek Praharaj. By understanding these processes, one can appreciate how GNNs can be utilized to analyze complex graph-structured data and improve predictive modeling in various applications.
# RNN for Jet Tagging Overview
## Introduction
This document provides an overview of the architecture and methodology used in the "RNN\_FOR\_JET\_TAGGING" analysis. The primary goal of this analysis is to utilize Recurrent Neural Networks (RNNs) for tagging jets in particle physics, which is crucial for identifying and classifying particles resulting from high-energy collisions.
## Purpose of the Analysis
The RNN model is designed to process sequences of data, making it suitable for tasks where temporal or sequential relationships are significant. In the context of jet tagging, the model aims to analyze sequences of features derived from jets to accurately classify them.
## Key Components of the RNN Architecture
### 1\. **Input Layer**
- **Data Representation**: The input consists of sequences representing jet features. Each sequence may include various attributes such as momentum, energy, and spatial coordinates.
### 2\. **Recurrent Layers**
- **Sequence Processing**: The core of the RNN architecture includes one or more recurrent layers that process the input sequences. Key characteristics include:
  - **Hidden States**: The RNN maintains hidden states that capture information from previous time steps, allowing it to learn temporal dependencies.
  - **Gated Mechanisms**: Variants like Long Short-Term Memory (LSTM) or Gated Recurrent Units (GRU) may be employed to better manage long-range dependencies and mitigate issues like vanishing gradients.
### 3\. **Activation Functions**
- **Non-linearity**: After processing through recurrent layers, activation functions such as Tanh or ReLU are applied to introduce non-linearity into the model, enhancing its ability to learn complex patterns.
### 4\. **Output Layer**
- **Classification**: The final output layer generates predictions regarding jet tags. This could involve:
  - Softmax activation for multi-class classification tasks.
  - Binary output for distinguishing between two types of jets.
## Training Process
- The RNN model is trained using labeled data relevant to jet tagging. A loss function appropriate for classification tasks (e.g., categorical cross-entropy) is used to evaluate performance, while optimization algorithms (like Adam) adjust model parameters during training.
## Conclusion
The "RNN\_FOR\_JET\_TAGGING" analysis leverages the strengths of Recurrent Neural Networks to effectively classify jets based on sequential data. By capturing temporal relationships within jet features, the RNN model aims to improve tagging accuracy in particle physics applications.
