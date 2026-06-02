🏠 House Price Prediction using Machine Learning
kaggle competition link - https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

📌 Project Overview

This project focuses on building an end-to-end machine learning pipeline to predict house prices using structured real-estate data. The goal was not just to train a model, but to follow a proper data science workflow including EDA, feature engineering, regularization, and model evaluation.

The final solution was evaluated on the Kaggle House Prices competition, achieving a strong score.

🎯 Problem Statement
Predict the SalePrice of houses based on various numerical and categorical features such as location, size, condition, and amenities.
This is a regression problem with a continuous target variable.

📂 Dataset
Source: Kaggle – House Prices: Advanced Regression Techniques
Files used:
train.csv – Training data with target variable
test.csv – Test data without target variable

🧠 Approach & Methodology
1️⃣ Exploratory Data Analysis (EDA)
Analyzed feature types (numerical vs categorical)
Identified missing values and their distribution
Inspected target variable (SalePrice) and observed right-skewness
Studied the impact of missing values on the target variable

2️⃣ Missing Value Handling
Numerical features filled using median
Categorical features filled using mode
Decision was data-driven based on EDA results

3️⃣ Feature Engineering
Converted categorical variables using one-hot encoding
Ensured consistency between training and test datasets
Removed data inconsistencies introduced by encoding

4️⃣ Feature Scaling
Applied standardization using StandardScaler
Required for regularized linear models such as Lasso, Ridge, and ElasticNet

5️⃣ Feature Selection
Used Lasso-based regularization to reduce dimensionality
Removed weak and noisy features to prevent overfitting

6️⃣ Target Transformation
Applied log transformation (log1p) on SalePrice
Reduced skewness and improved model generalization

7️⃣ Modeling & Evaluation
Built clean and reproducible scikit-learn pipelines

Compared:
Ridge Regression
ElasticNet Regression

Evaluation Metric:
Root Mean Squared Error (RMSE) on log-transformed target

🏆 Results
Final Kaggle Score: 0.18394 (Log-RMSE)
Best Model: ElasticNet Regression
Demonstrated strong generalization with reduced overfitting

🛠️ Tech Stack
Python
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
Jupyter Notebook

📈 Key Learnings
Importance of EDA before modeling
Handling skewed targets using log transformation
Role of regularization in reducing overfitting
Advantages of pipelines for preventing data leakage
Proper Kaggle submission workflow

🚀 Future Improvements
Hyperparameter tuning using cross-validation
Experiment with tree-based models (Random Forest, XGBoost, LightGBM)
Feature interaction engineering
Deployment as a web application

📎 Kaggle Submission
Competition: House Prices – Advanced Regression Techniques
Public Score: 0.18394

👤 Author

Darsh Jilka
