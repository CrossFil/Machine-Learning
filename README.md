Supervised Learning Algorithms – Part 1

This repository demonstrates the implementation of a supervised learning workflow using the California Housing dataset. The following steps outline the process:
Steps Completed

Import Required Libraries
Imported necessary libraries such as pandas, numpy, scipy, and sklearn for data handling, analysis, and model building.
Load Dataset
The California Housing dataset was loaded for regression analysis.
Exploratory Data Analysis (EDA) was performed previously and is not repeated here.
Data Preprocessing
Outlier Removal:
Identified and removed outliers from the AveRooms, AveBedrms, AveOccup, and Population columns using zscore() from scipy.
Applied apply() to calculate Z-scores and removed rows where any value exceeded the threshold using any().
Feature Selection:
Removed one highly correlated feature based on the correlation matrix.
Split Data
Divided the dataset into training and testing subsets using train_test_split() from sklearn.
Feature Normalization
Standardized feature values using StandardScaler from sklearn.
Model Building
Constructed a regression model using the LinearRegression class from sklearn.
Model Evaluation
Calculated the following performance metrics:
R-squared (R²): Coefficient of determination.
MAE: Mean Absolute Error.
MAPE: Mean Absolute Percentage Error.
Comparison and Conclusion
Compared the performance metrics of the current model with those from a previous implementation and documented the observations.
