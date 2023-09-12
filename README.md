# House Price Prediction Project

This repository contains the code and documentation for a machine learning project aimed at predicting house prices. In this project, we explore various machine learning models, data preprocessing techniques, and data analysis to build an accurate house price prediction model.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling](#modeling)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction

The goal of this project is to predict house prices based on various features such as the type of zoning, lot area, sale condition, and more. We employ a variety of machine learning models, including linear regression, polynomial linear regression, decision trees, random forests, and neural networks, to find the best predictive model.

## Getting Started

To run this project locally, follow these steps:

1. Clone this repository to your local machine:

   ```
   git clone https://github.com/your-username/house-price-prediction.git
   ```
2. Install the required Python libraries and dependencies as needed for your specific dataset.
3. Prepare your dataset in the data directory. Ensure that you have separate training and test data files, preferably in CSV format.
4. Modify the configuration files and parameters as necessary for your specific dataset.
5. Run the Jupyter notebooks or Python scripts to execute the code for data preprocessing, analysis, and model training.
## Data Preprocessing

In the data preprocessing phase, we handle missing values, encode categorical features using one-hot encoding, and normalize the data using the MinMaxScaler from scikit-learn. We also handle outliers and perform feature engineering when necessary.

## Exploratory Data Analysis

In this section, we explore the dataset visually by creating scatter plots, box plots, and correlation matrices to gain insights into the relationships between different features and the target variable (SalePrice).

## Modeling

We experiment with multiple machine learning models, including linear regression, polynomial linear regression, decision trees, random forests, and neural networks. We perform hyperparameter tuning and cross-validation to optimize the model's performance.

## Results

Here are some of the results achieved in our project:

- **Linear Regression**:
  - Accuracy on test data: 67,212.67
  - Accuracy on train data: 65,012.93

- **Polynomial Linear Regression (degree=7)**:
  - Accuracy on test data: 46,770.40
  - Accuracy on train data: 42,349.68

- **Decision Tree Regressor (max_depth=30, min_samples_split=3, max_features=14, random_state=0)**:
  - Accuracy on test data: 12,866.77
  - Accuracy on train data: 11,702.44

- **Random Forest Regressor (n_estimators=170, max_depth=30, random_state=0)**:
  - Accuracy on test data: 22,641.53
  - Accuracy on train data: 65,012.93 (overfitting)

- **Neural Network (using TensorFlow/Keras)**:
  - Accuracy on test data: 67,212.67
  - Accuracy on train data: 65,012.93

## Conclusion

In conclusion, the Decision Tree Regressor with specific hyperparameters proved to be the best-performing model for house price prediction in this project, achieving a low mean squared error on the test dataset. This model is suitable for predicting house prices based on the provided features.

## References

- [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Neural Networks in Scikit-Learn](https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- [Decision Trees in Scikit-Learn](https://scikit-learn.org/stable/modules/tree.html)

