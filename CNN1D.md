# CNN for Jet Constituents

This project implements a **Convolutional Neural Network (CNN)** designed to analyze jet constituents in high-energy physics datasets. The goal is to perform classification using jet particle features and explore how well CNNs can distinguish between different jet structures by leveraging constituent-level information. The project is implemented using **TensorFlow** and utilizes particle feature datasets stored in **HDF5** format.

## Table of Contents

1. [Part 1: Dataset Preparation and Exploration](#part-1-dataset-preparation-and-exploration)
2. [Part 2: Model Architecture](#part-2-model-architecture)
3. [Part 3: Training the Model](#part-3-training-the-model)
4. [Part 4: Evaluation and Analysis](#part-4-evaluation-and-analysis)
5. [Part 5: Conclusion](#part-5-conclusion)

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
