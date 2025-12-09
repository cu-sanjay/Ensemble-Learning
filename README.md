# Pulmonary Disease Detection System using Ensemble Learning

## Introduction

This project presents an ensemble learning based pulmonary disease detection system using chest X-ray images. It integrates a custom Convolutional Neural Network with pre-trained models and a meta-learner to improve classification performance across multiple lung disease categories. It is designed for research, experimentation, and deployment through both Google Colab and a Streamlit web interface.

This work aligns with the study *Development of Ensemble Learning Algorithms for Pulmonary Disease Detection System* by S. Choudhary, A. P. Singh and A. Tiwari, published in the 2025 International Conference on Data Science, Agents and Artificial Intelligence (ICDSAAI), Chennai, India.
DOI: 10.1109/ICDSAAI65575.2025.11011653.

# Project Title

**Development of Ensemble Learning Algorithms for Pulmonary Disease Detection System**

# Project Objectives

1. Review the current methods used for detecting pulmonary diseases, with emphasis on Chronic Obstructive Pulmonary Disease.
2. Develop an ensemble detection system that integrates machine learning and deep learning models.
3. Improve accuracy by combining multiple feature extraction and classification methods.
4. Provide a practical deployment platform using a Streamlit web application.

# Team Members

* Ajay
* Ashwani
* Pragati
* Sanjay

## Supervisor

**Ajay Pal Singh (E13293)**

# Dataset

* **Source**: Lungs Disease Dataset – Kaggle
* **Classes**:

  * Bacterial Pneumonia
  * Corona Virus Disease
  * Normal
  * Tuberculosis
  * Viral Pneumonia
* **Format**: JPEG, JPG, PNG
* **Placement for Web App**: Place the dataset inside the `data` folder.

> The dataset does not include COPD labels. The proposed system can be extended for COPD-specific detection when appropriate data becomes available.

# System Architecture

1. Load and preprocess images with augmentation.
2. Train a custom CNN for baseline learning.
3. Use ResNet50 and VGG16 for deep feature extraction.
4. Combine extracted features and train an XGBoost meta-learner.
5. Evaluate all models using standard metrics.
6. Deploy the final ensemble model using a Streamlit interface.

# Project Structure

The project includes both a **Google Colab notebook** for training and a **Streamlit web interface** for deployment.

### Colab Notebook Tasks

* Dataset download using Kaggle API
* Preprocessing with augmentation
* Training of custom CNN
* Feature extraction using ResNet50 and VGG16
* Ensemble training using XGBoost
* Model evaluation
* Visualization of training history
* Saving metrics, plots, and trained models to Google Drive

### Streamlit Web Application Tasks

* Load trained models
* Accept user-uploaded chest X-ray image
* Preprocess and classify image
* Display final disease prediction
* Provide probability scores
* Show intermediate features when required

# Prerequisites

### For Google Colab

* Google Colab account
* Kaggle account and API key
* Google Drive for saving output

### For Local Streamlit Application

* Python 3
* Required dependencies installed
* Dataset placed in `data/`

# Dependencies

The system uses the following libraries:

* tensorflow
* opencv-python
* xgboost
* scikit-learn
* seaborn
* matplotlib
* numpy
* pandas
* kaggle
* streamlit

# How to Run

## 1. Run the Streamlit Web Application Locally

```bash
python -m streamlit run web_app/web.py
```

Ensure the dataset is placed in the `data/` directory.

## 2. Run the Full Training Pipeline in Google Colab

### Steps

1. Open Google Colab.
2. Copy contents of the training notebook into a Colab file.
3. Enable GPU under Runtime settings.
4. Upload your Kaggle API key (`kaggle.json`).
5. Run all cells to perform:

   * Dataset download
   * Preprocessing
   * CNN training
   * ResNet50 and VGG16 feature extraction
   * XGBoost meta-learner training
   * Evaluation and visualization
   * Saving all outputs to Google Drive

# Outputs

## Models

* `cnn_model.h5`
* `xgb_model.json`
* Extracted feature files
* Saved Streamlit-ready model files

## Plots

* `cnn_training_plot.png`
* `confusion_matrix.png`

## Metrics

* `metrics.txt` stored in Google Drive
* Accuracy, precision, recall, and F1-score for validation and test sets

## Console Output

* Training logs
* Evaluation metrics
* Confusion matrix

# Output Screenshots (Web Application)

![Screenshot 1](images/Screenshot%202025-03-13%20163816.png)
![Screenshot 2](images/Screenshot%202025-03-13%20163845.png)
![Screenshot 3](images/Screenshot%202025-03-13%20163901.png)
![Screenshot 4](images/Screenshot%202025-03-13%20164046.png)
![Screenshot 5](images/Screenshot%202025-03-13%20164121.png)
![Screenshot 6](images/Screenshot%202025-03-13%20164420.png)

# Expected Results

* Custom CNN provides a moderate accuracy baseline.
* Ensemble model combining CNN, ResNet50, and VGG16 features achieves higher accuracy, generally above 90 percent depending on preprocessing.
* Confusion matrix highlights classification strength across disease classes.
* Training history visualizations show performance trends.

# Notes for Research

* COPD-specific detection requires datasets with COPD labels.
* The ensemble system can be extended using more advanced feature selection methods.
* Literature review can be expanded using X-ray based deep learning studies.
* Incorporating clinical data may improve COPD performance.

# Future Work

* Add DenseNet, InceptionV3, or EfficientNet models.
* Explore 3D CNNs for CT scan datasets.
* Integrate optimization algorithms for feature selection.
* Deploy on cloud for real-time clinical use.
* Improve explainability with Grad-CAM or similar tools.

# Troubleshooting

### Kaggle API Failure

Regenerate `kaggle.json` if download fails.

### Memory Issues in Colab

Reduce batch size or image resolution.

### Slow Training

Confirm GPU is enabled.

### Streamlit Errors

Check model paths and dataset folder structure.

# Research Publication Reference

S. Choudhary, A. P. Singh and A. Tiwari, “Development of Ensemble Learning Algorithms for Pulmonary Disease Detection System,” ICDSAAI 2025, Chennai, India, pp. 1–6.
DOI: 10.1109/ICDSAAI65575.2025.11011653.

# License

This project is for academic use. Commercial use requires appropriate licensing.