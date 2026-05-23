# Toxic_Comments_Classification
# Toxic Comment Classification Project

## Overview

This project implements and compares multiple machine learning models for toxic comment classification. The goal is to automatically detect toxic comments across six different categories: toxic, severe toxic, obscene, threat, insult, and identity hate. This is a multi-label classification problem where each comment can belong to one or more toxicity categories.

## Dataset

The project uses the **Wikipedia Toxic Comments** dataset, which contains:
- **Training data**: 159,571 comments with toxicity labels
- **Test data**: 153,164 comments for prediction
- **Labels**: 6 toxicity categories (toxic, severe_toxic, obscene, threat, insult, identity_hate)

The dataset is highly imbalanced, with only ~10.17% of comments labeled as toxic.

## Models Implemented

The following machine learning models were implemented and compared:

| Model | Learning Type | Description |
|-------|--------------|-------------|
| **SVM (Support Vector Machine)** | Supervised | LinearSVC with probability calibration for text classification |
| **Neural Network (MLPClassifier)** | Supervised | Multi-layer perceptron for non-linear text classification |
| **Logistic Regression** | Supervised | Binary classification with probability output |
| **SGDClassifier** | Supervised | Fast linear classifier for large-scale data |
| **Naive Bayes (MultinomialNB)** | Supervised | Probabilistic classifier based on word frequencies |

## Project Structure
toxic-comment-classification/
│
├── Main.ipynb # Main Jupyter notebook with full implementation
├── train.csv # Training dataset
├── test.csv # Test dataset
├── test_labels.csv # Test labels (for evaluation)
├── submission_*.csv # Prediction submissions for each best model
└── README.md # Project documentation


## Methodology

### 1. Data Preprocessing
- Convert text to lowercase
- Remove URLs and special characters
- Strip punctuation and extra whitespaces

### 2. Feature Extraction
- **TF-IDF Vectorization** with:
  - Maximum features: 15,000
  - Unigrams only (ngram_range=(1,1))
  - Min document frequency: 2
  - Max document frequency: 0.95
  - Sublinear TF scaling

### 3. Model Training
- Train/validation split: 80/20
- Stratified splitting based on 'toxic' label
- Each model trained separately for each of the 6 toxicity labels

### 4. Evaluation Metrics
- Accuracy
- F1 Score (macro average)
- ROC AUC Score
- Training time comparison

## Results

### Model Performance Summary

| Model | Avg Accuracy | Avg F1 Score | Avg ROC AUC | Training Time (s) |
|-------|--------------|--------------|-------------|-------------------|
| **Logistic Regression** | 0.9001 | 0.4119 | **0.9681** | 165.61 |
| Neural Network | 0.9817 | 0.5434 | 0.9645 | 459.85 |
| SGD Classifier | 0.9428 | 0.4155 | 0.9632 | 2.43 |
| SVM | 0.9798 | 0.4304 | 0.9581 | 102.92 |
| Naive Bayes | 0.9771 | 0.3284 | 0.9290 | 0.24 |

### Key Findings

1. **Best Overall Model**: Logistic Regression achieved the highest ROC AUC score (0.9681), making it the best performer for distinguishing between toxic and non-toxic comments.

2. **Fastest Training**: SGDClassifier demonstrated excellent scalability with only 2.43 seconds total training time while maintaining competitive performance.

3. **Performance Challenge**: All models struggled with minority classes (threat, severe_toxic, identity_hate), as evidenced by lower F1 scores for these categories due to class imbalance.

## Hyperparameter Tuning

GridSearchCV was performed on Logistic Regression with:
- **Parameters tuned**: C (0.1, 1, 10), solver (saga, liblinear), max_iter (500, 1000)
- **Best parameters**: {'C': 1, 'max_iter': 500, 'solver': 'saga'}
- **Cross-validation ROC AUC**: 0.9700

## Requirements
python >= 3.11
numpy
pandas
scikit-learn
matplotlib
seaborn


Install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn


**## Usage**

Prepare the data: Ensure train.csv, test.csv, and test_labels.csv are in the specified paths
Run the notebook: Execute Main.ipynb cell by cell to:

Load and explore the data
Preprocess text comments
Extract TF-IDF features
Train all models
Compare results
Generate predictions on test set
Generate submissions: The notebook automatically saves submission files with predictions for the best model


**Key Insights**

Text Length: Non-toxic comments tend to be longer (average 404 characters) compared to toxic comments (average 303 characters).
Label Correlations: Strong correlations exist between obscene, insult, and toxic labels, indicating these categories often co-occur.
Class Imbalance: The dataset is heavily imbalanced, with only:

10.17% comments labeled as toxic
0.3% comments labeled as threat
This explains the lower F1 scores for rare categories
