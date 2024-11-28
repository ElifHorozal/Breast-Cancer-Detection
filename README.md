# Breast-Cancer-Detection

## Project Overview
This project explores different methods for breast cancer detection using the Breast Cancer Wisconsin (Diagnostic) Dataset. The goal is to determine the most accurate method for classifying breast cancer cases as benign or malignant. The project is divided into two parts:

1. Custom Implementation of Naïve Bayes Algorithm
2. Comparison of Gaussian Naïve Bayes and Logistic Regression using scikit-learn


## Dataset
The dataset used for this project can be found here: Breast Cancer Dataset

## Methods and Implementation
### Part 1: Custom Naïve Bayes Implementation
In this section, a Naïve Bayes algorithm was implemented from scratch using Python, with minimal libraries such as NumPy and Pandas.

#### Key Functions:
1. Mean Calculation
2. Covariance Matrix Calculation
3. Bayesian Classification

#### Process:
1. Split the dataset into training (456 samples) and test (114 samples) sets.
2.  Divided the training data into benign and malignant classes.
3.   Converted data into matrices for easier computation.
4.    Applied Bayes Decision Rule for classification.

### Part 2: scikit-learn Implementation
This section uses scikit-learn to implement and compare two methods: Gaussian Naïve Bayes and Logistic Regression.

#### Process:
1. Data Preparation
2. Gaussian Naïve Bayes
3. Logistic Regression

#### Results:
Method	Accuracy
Custom Naïve Bayes	81.58%
Gaussian Naïve Bayes	92.98%
Logistic Regression	96.49%


### Results and Conclusion
The results indicate that Logistic Regression performed the best on the Breast Cancer Wisconsin Dataset with an accuracy of 96.49%, followed by Gaussian Naïve Bayes (92.98%) and the custom implementation (81.58%).
