# Plant Disease Classification

## Project Overview

This project aims to classify diseases in plant leaves using deep learning techniques. By leveraging computer vision, the goal is to accurately identify various diseases from images of plant leaves, which can be crucial for early detection and prevention in agriculture. The models are trained on the PlantVillage dataset.

## Dataset

The project utilizes the **PlantVillage Dataset**, a publicly available dataset containing images of healthy and diseased plant leaves.
* **Source:** Kaggle
* **Link to a version of the dataset:** [PlantVillage Dataset]((https://www.kaggle.com/datasets/emmarex/plantdisease/data))
* **Details:** The dataset used in this project contains approximately 20,639 images across 15 different classes of plants and diseases. Some class imbalance was noted during exploration.

## Methodology

The project follows a systematic approach to image classification:

1.  **Data Loading and Exploration:** Images were loaded, and an initial exploration was performed to understand class distributions and image characteristics.
2.  **Data Preprocessing:**
    * Images were resized to a uniform dimension (256x256 pixels).
    * Pixel values were normalized (rescaled to a 0-1 range).
    * Data augmentation techniques were applied using `ImageDataGenerator` to increase the diversity of the training set and improve model generalization.
    * The dataset was split into training (80%) and validation (20%) sets.
3.  **Model Development:**
    * **Baseline CNN:** A custom Convolutional Neural Network (CNN) was built from scratch to establish a performance baseline.
    * **Transfer Learning:** Several pre-trained models were leveraged for transfer learning, including:
        * ResNet50
        * VGG16
        * InceptionV3
        These models had their final layers replaced with custom dense layers suitable for the specific classification task.
4.  **Model Training:**
    * Models were compiled using the Adam optimizer and categorical cross-entropy loss function.
    * Key metrics tracked included Accuracy, Precision, Recall, and F1-Score.
    * Callbacks such as `ModelCheckpoint` (to save the best model) and `EarlyStopping` (to prevent overfitting) were considered/used.
5.  **Evaluation:**
    * Models were evaluated on the validation set.
    * Performance was assessed using classification reports (providing precision, recall, F1-score per class) and confusion matrices.
    * Training and validation accuracy/loss curves were plotted to monitor learning progress.

## Results

The performance of different models was compared. Based on the analysis:

* **Custom CNN:** Achieved an accuracy of approximately 91.95% and a weighted F1-score of approximately 91.99%.
* **VGG16:** Achieved an accuracy of approximately 93.11% and a weighted F1-score of approximately 93.12%.
* **InceptionV3:** Achieved an accuracy of approximately 92.07% and a weighted F1-score of approximately 92.10%.
* **ResNet50:** Emerged as the best-performing model with an **accuracy of approximately 96.80%** and a **weighted F1-score of approximately 96.85%**.

*(Note: Insert specific final metrics from your notebook if they differ slightly or if you have more precise numbers for all models).*

## Technologies Used

* **Programming Language:** Python
* **Core Libraries:**
    * TensorFlow & Keras (for deep learning model development and training)
    * Scikit-learn (for metrics like classification report, confusion matrix, and class weights)
    * Pandas (for data handling, e.g., results comparison table)
    * NumPy (for numerical operations)
    * Matplotlib & Seaborn (for data visualization)
* **Environment:** Jupyter Notebook (Run on Kaggle with GPU acceleration)

## File Structure

* `plant-disease-classification.ipynb`: The main Jupyter Notebook containing all the code for data processing, model building, training, and evaluation.

## Setup and Usage

1.  **Clone the repository (if applicable).**
2.  **Ensure you have a Python environment with the necessary libraries installed.** You can typically install them using pip:
    ```bash
    pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
    ```
3.  **Dataset:** Download the PlantVillage dataset from Kaggle. You might need to adjust the file paths in the notebook (`/kaggle/input/plantdisease/PlantVillage`) if you run it locally, to point to where you've stored the dataset.
4.  **Run the Jupyter Notebook:** Open and run the `plant-disease-classification.ipynb` notebook in a Jupyter environment. If using GPU, ensure your TensorFlow installation is GPU-compatible and drivers are correctly set up.
