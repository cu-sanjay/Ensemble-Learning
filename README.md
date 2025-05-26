# Ensemble Learning for Pulmonary Disease Detection System

## Project Overview
This project focuses on the development of an ensemble learning system for detecting pulmonary diseases, with a specific emphasis on Chronic Obstructive Pulmonary Disease (COPD). The system leverages deep learning models (Custom CNN, ResNet50, VGG16) combined with an XGBoost meta-learner to achieve high-accuracy classification of chest X-ray images. The project uses the **Lung Disease Dataset (4 types)** from Kaggle, containing approximately 10.1k images across five classes: Bacterial Pneumonia, Corona Virus Disease, Normal, Tuberculosis, and Viral Pneumonia.

### Research Objectives
1. **Review State-of-the-Art Methods**: Analyze current deep learning and ensemble learning approaches for pulmonary disease detection, particularly for COPD.
2. **Propose an Ensemble Learning System**: Develop a detection system for COPD using an ensemble of machine learning models, including deep learning architectures.
3. **Integrate Advanced Techniques**: Combine multiple deep learning models with sophisticated feature extraction to improve detection accuracy.

### Dataset
- **Source**: [Lung Disease Dataset (4 types)](https://www.kaggle.com/datasets/omkarmanohardalvi/lungs-disease-dataset-4-types) on Kaggle.
- **Structure**: ~10.1k chest X-ray images (.jpeg, .jpg, .png) split into:
  - **Train**: Images for training the models.
  - **Validation**: Images for hyperparameter tuning and validation.
  - **Test**: Images for final evaluation.
- **Classes**: 
  - Bacterial Pneumonia
  - Corona Virus Disease
  - Normal
  - Tuberculosis
  - Viral Pneumonia
> **Note**: The dataset does not explicitly label COPD. The system is designed to detect general pulmonary diseases, with adaptability for COPD-specific detection if additional data or labels are provided.

## Project Structure
The project is implemented as a **Google Colab notebook** (**Pulmonary_Disease_Detection_Ensemble.ipynb**) that includes:
- Data loading and preprocessing with augmentation.
- Training a custom Convolutional Neural Network (CNN).
- Feature extraction using pre-trained ResNet50 and VGG16 models.
- Ensemble learning with XGBoost as the meta-learner.
- Evaluation with accuracy, precision, recall, F1-score, and confusion matrix.
- Visualization of training history and results.
- Saving models, plots, and metrics to Google Drive.

## Prerequisites
To run the project, you need:
- **Google Colab Account**: Access to **[Google Colab](https://colab.research.google.com/)** with GPU support.
- **Kaggle Account**: For downloading the dataset via the Kaggle API.
- **Google Drive**: For saving models, plots, and metrics.
- **Kaggle API Key**:
  - Generate a **kaggle.json** file from Kaggle (Account > API > Create New API Token).
  - Upload this file when prompted in Colab.

### Dependencies
The project uses the following Python libraries, which are installed automatically in the Colab notebook:
- **tensorflow** (for deep learning models)
- **opencv-python** (for image processing)
- **xgboost** (for ensemble meta-learner)
- **scikit-learn** (for evaluation metrics)
- **seaborn**, **matplotlib** (for visualization)
- **kaggle** (for dataset download)
- **numpy**, **pandas** (for data handling)

## Setup and Installation
1. **Open Google Colab**:
   - Create a new notebook in **[Google Colab](https://colab.research.google.com/)**.
2. **Copy the Code**:
   - Copy the contents of **Pulmonary_Disease_Detection_Ensemble.ipynb** into the Colab notebook. The notebook is provided separately or can be obtained from the project repository.
3. **Enable GPU**:
   - Go to **Runtime > Change runtime type > Hardware accelerator > GPU** to enable GPU acceleration for faster training.
4. **Set Up Kaggle API**:
   - When prompted by the notebook, upload your **kaggle.json** file (downloaded from Kaggle).
   - The notebook will configure the Kaggle API and download the dataset.
5. **Mount Google Drive**:
   - Follow the prompt to mount your Google Drive, where outputs (models, plots, metrics) will be saved.

## Running the Project
1. **Execute the Notebook**:
   - Run all cells in the Colab notebook sequentially (`Ctrl+F9` or `Runtime > Run all`).
   - The notebook will:
     - Install dependencies.
     - Download and unzip the Lung Disease Dataset.
     - Preprocess images with data augmentation and normalization.
     - Train a custom CNN for 10 epochs.
     - Extract features using ResNet50 and VGG16.
     - Train an XGBoost meta-learner on combined features.
     - Evaluate the ensemble model on validation and test sets.
     - Generate and save plots (training history, confusion matrix).
     - Save models and metrics to Google Drive.
2. **Monitor Outputs**:
   - Check the Colab output for training progress, evaluation metrics, and visualizations.
   - Outputs are saved to **📁 /content/drive/MyDrive/** in your Google Drive.

## Outputs
The project generates the following outputs, saved to Google Drive:
- **Models**:
  - `cnn_model.h5`: Trained custom CNN model.
  - `xgb_model.json`: Trained XGBoost meta-learner.
- **Plots**:
  - `cnn_training_plot.png`: Training and validation accuracy/loss for the CNN.
  - `confusion_matrix.png`: Confusion matrix for the ensemble model on the test set.
- **Metrics**:
  - `metrics.txt`: Validation and test metrics (accuracy, precision, recall, F1-score).
- **Console Output**:
  - Training logs, evaluation metrics, and confusion matrix visualization displayed in Colab.

## Expected Results
- **Custom CNN**: Achieves moderate accuracy after 10 epochs, serving as a baseline.
- **Ensemble Model**: Combines features from CNN, ResNet50, and VGG16, typically achieving test accuracy >90% (depending on dataset balance and augmentation).
- **Metrics**: Includes accuracy, precision, recall, and F1-score for both validation and test sets.
- **Visualizations**: Training history plots and a confusion matrix to assess model performance across classes.

## Notes for Research
- **COPD Detection**: The dataset does not explicitly label COPD. The system detects general pulmonary diseases but is designed to be adaptable for COPD with additional data (e.g., spirometry or COPD-labeled X-rays). For COPD-specific enhancements, consider:
  - Labeling a subset of Pneumonia/Tuberculosis images as COPD proxies.
  - Integrating clinical data or feature selection (e.g., Mayfly optimization).
- **Performance**: The ensemble approach leverages transfer learning and data augmentation to handle dataset challenges (e.g., imbalanced classes). Results depend on data quality and preprocessing.
- **Documentation**: Use the notebook’s markdown sections (literature review, conclusion) and saved outputs (metrics, plots) for your thesis. Expand the literature review with additional references as needed.
- **Future Work**:
  - Experiment with additional models (e.g., DenseNet, InceptionV3).
  - Implement 3D CNNs for CT scans if available.
  - Explore advanced feature selection techniques.

## Troubleshooting
- **Kaggle API Issues**: Ensure your **kaggle.json** file is valid and has not expired. Re-generate it from Kaggle if needed.
- **Memory Errors**: If Colab runs out of memory, reduce the batch size (**BATCH_SIZE**) or image size (**IMG_SIZE**) in the notebook.
- **Dataset Download Fails**: Verify your Kaggle account has access to the dataset and that the API key is correctly configured.
- **Slow Training**: Ensure GPU is enabled in Colab. If training is still slow, reduce the number of epochs or models in the ensemble.

## Acknowledgments
- **Dataset**: Provided by Omkar Manohar Dalvi on Kaggle.
- **Libraries**: TensorFlow, XGBoost, Scikit-learn, OpenCV, Seaborn, Matplotlib.
- **Platform**: Google Colab for providing free GPU resources.

## License
This project should not be used for commercial applications without proper licensing of the dataset and dependencies.
