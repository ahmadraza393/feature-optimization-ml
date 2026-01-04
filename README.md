# Feature Engineering and Model Optimization Study

This project investigates the impact of feature scaling and regularization techniques on machine learning models, focusing on optimization stability and generalization performance.

## Overview
Includes feature scaling, regularization, training multiple supervised learning models, and evaluating performance metrics.

## Dataset
Use any publicly available classification dataset, e.g., Heart Disease dataset.  
Expected location:

data/raw/heart_disease.csv

Recommended dataset link (Kaggle):  
https://www.kaggle.com/datasets/toupasana/heart-disease-data-set

## Models
- Logistic Regression (with L1/L2 regularization)
- Random Forest
- Support Vector Machine

## Technologies
- Python
- Scikit-learn
- Pandas
- NumPy

## How to Run

1. Install dependencies:

pip install -r requirements.txt

2. Place the dataset in:

data/raw/heart_disease.csv

3. Run the project:

python main.py

## Folder Structure
data/
    raw/
    processed/
models/
notebooks/
results/
src/
    data_preprocessing.py
    feature_engineering.py
    model_training.py
    evaluation.py
main.py
requirements.txt
README.md
