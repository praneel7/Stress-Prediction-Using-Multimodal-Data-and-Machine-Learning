# Stress Prediction Using Multimodal Data and Machine Learning
*CSE 419: Artificial Intelligence for Health — Final Project*

## Overview

This project explores the use of mobile sensor data and survey responses from the GLOBEM dataset to predict stress levels in students. The final objective is to classify end-of-semester stress into Low, Medium, or High using classical ML models and deep learning architectures. The final classification model, based on Random Forest, achieved an AUROC of 0.810 and AUPRC of 0.690 on an external validation set.

## Project Structure


## Dataset

- **Source**: GLOBEM Dataset (2018–2021)
- **Participants**: 497 college students
- **Modalities Used**:
  - Pre-study surveys (mental health, well-being)
  - EMA surveys (weekly stress and affect)
  - Mobile sensor data (steps, screen time, sleep, location, etc.)

**Note**: 2018 was excluded due to missing features.  
Training: 2019 + 2020 | Testing: 2021

## Data Processing

- **EMA Aggregation**: 10 statistical features per variable (mean, std, skewness, etc.)
- **Sensor Data**: Used SSA (Singular Spectrum Analysis) to extract temporal trends
- **Missing Data**:
  - Dropped students with >20% missing sensor data
  - Backfill imputation applied for remaining missing values

## Feature Engineering

- SSA decomposition on 22 semantic sensor features → 220+ spline/linear features
- Hierarchical Recursive Feature Elimination (RFE) for feature selection
  - Final set: 27 features for classification

## Models Used

### Classical Machine Learning

| Task         | Models Used              |
|--------------|--------------------------|
| Regression   | Linear, Lasso, Ridge, SVR, RF, XGB |
| Classification | Logistic Regression, SVC, RF, XGB |

- Evaluation: RMSE, R² (for regression), AUROC, AUPRC (for classification)

### Deep Learning (Exploratory)
- CNN
- LSTM  
Both models underperformed (R² ~ 0), likely due to data limitations.

## Results Summary

| Setting          | Best Model      | AUROC | AUPRC |
|------------------|------------------|--------|--------|
| External Test (2021) | Random Forest | 0.810  | 0.690  |

- Survey data (EMA + Pre) was most predictive
- Sensor data contributed marginally
- SHAP showed top features were primarily survey-based

## How to Reproduce

1. Clone the repo
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
