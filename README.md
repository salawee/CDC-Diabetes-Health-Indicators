# ⚕️ Predictive Analytics for Early Detection of Diabetes 🩺

## 🔍 Overview

This repository presents a machine learning-based approach for early detection of diabetes using the Behavioral Risk Factor Surveillance System (BRFSS) data. The project aims to develop predictive models that help identify individuals at risk of diabetes or prediabetes based on survey responses. By utilizing advanced machine learning techniques such as logistic regression, random forests, and support vector machines (SVM), this analysis seeks to provide accurate predictions, aiding public health initiatives and early intervention strategies.

# 📑 Table of Contents
- [Introduction](#introduction)
- [Data Sources](#data-sources)
- [Methodology](#methodology)
  - [Model Building](#model-building)
  - [Feature Selection](#feature-selection)
  - [Model Evaluation](#model-evaluation)
- [Results](#results)
   - [Key Findings](#key-findings)
- [Discussion](#discussion)
- [Contributions](#contributions)
- [Citation](#citation)

## 📘 Introduction
Diabetes has become a major public health issue, affecting millions and imposing significant economic burdens. Early detection of diabetes through predictive modeling allows for timely intervention, potentially reducing the impact of the disease. This project leverages BRFSS data, a large-scale health-related survey, to build machine learning models that predict the onset of diabetes based on risk factors such as age, BMI, physical activity, and more. By applying predictive analytics, this project aims to improve public health surveillance and guide strategies for early diagnosis and prevention.

## 🛠️ Data Sources
The dataset used in this project comes from the Behavioral Risk Factor Surveillance System (BRFSS), collected by the CDC. This survey includes responses from over 400,000 Americans, covering health behaviors, chronic conditions, and preventative service usage. Three primary datasets were used, containing data on diabetes, prediabetes, and health indicators. The dataset includes both binary classification (diabetes vs. non-diabetes) and multi-class labels (no diabetes, prediabetes, and diabetes).

## 📊 Methodology

## 🧠 Model Building
This project employed various machine learning models to predict diabetes status:

Logistic Regression: A simple yet effective model for binary classification.
Random Forests: A robust model that handles both linear and non-linear relationships.
Support Vector Machines (SVM): A powerful method for handling high-dimensional data and finding optimal decision boundaries.

## 🔍 Feature Selection
To improve model efficiency and prediction accuracy, feature selection techniques were applied to identify the most significant predictors of diabetes. The study assessed variables such as BMI, high blood pressure, and physical activity to understand their contribution to the model. Using feature importance scores, we developed a reduced version of the BRFSS survey to maintain accuracy with fewer inputs.

## 🧪 Model Evaluation
The models were evaluated using cross-validation, ROC-AUC scores, sensitivity, and specificity. These metrics allowed for a comprehensive understanding of the models’ ability to accurately predict diabetes across different groups, with particular attention to minimizing false negatives in early diabetes detection.

## 🎯 Results
The results of this project revealed varying performances across different machine learning models. Logistic Regression and Random Forest models exhibited strong predictive power, with Random Forest showing the highest overall accuracy. The ROC-AUC scores further demonstrated the ability of these models to distinguish between diabetic and non-diabetic individuals. After feature elimination, the models maintained high accuracy with a reduced number of input variables.

## 📊 Key Findings
Random Forest provided the highest overall accuracy for predicting diabetes, with an AUC of 0.8044.
Logistic Regression performed well, with an accuracy of 86.33%, showing its reliability in simpler models.
Feature selection reduced the number of inputs while maintaining accuracy, simplifying the prediction process and making it more scalable for real-world applications.

## 💬 Discussion
This project demonstrates the potential of machine learning to enhance early diabetes detection using survey-based data. By building predictive models with high accuracy, we can support public health initiatives focused on early intervention. The results indicate that Random Forests are particularly effective at identifying high-risk individuals, while logistic regression models offer a simpler, yet effective, solution. Future work could focus on integrating additional data sources, such as genetic information or continuous glucose monitoring data, to further enhance predictive power.

## 🤝 Contributions
We encourage contributions from data scientists, healthcare professionals, and public health researchers interested in improving diabetes prediction models. Contributions could include refining existing models, experimenting with new algorithms, or adding external data sources to enhance prediction accuracy.

##📚 Citation
Please cite this work using the following format: [Your Name]. Predictive Analytics for Early Detection of Diabetes: Using Machine Learning to Forecast Diabetes Risk. 2024.
