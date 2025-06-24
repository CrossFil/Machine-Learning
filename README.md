# Supervised Learning Algorithms

## Task Instructions (Linear_Regression.Evaluating_Regression_Quality.py)
1. Import the required packages.

2. Load the California Housing dataset, as shown in the section  
   *“Applying Linear Regression. EDA of the California Housing dataset”*  
   from the topic *“Linear Regression. Evaluating Regression Quality”*.

3. Perform data preprocessing:

   - **3.1. Outlier removal**  
     Remove outliers in the columns: `AveRooms`, `AveBedrms`, `AveOccup`, `Population`.  
     Use `scipy.stats.zscore()` and `pandas.DataFrame.apply()` to compute z-scores.  
     Drop rows where at least one of these columns contains an outlier (e.g., z > 3), using `any()`.

   - **3.2. Remove one highly correlated feature**  
     Use the correlation matrix and drop one feature from a highly correlated pair.

4. Split the dataset into training and testing sets using `train_test_split()` from `sklearn`.

5. Normalize features using `StandardScaler` from `sklearn`.

6. Build a model using `LinearRegression` from `sklearn`.

7. Evaluate model performance using:
   - R² (coefficient of determination)
   - MAE (Mean Absolute Error)
   - MAPE (Mean Absolute Percentage Error)

8. Compare the new metrics with those from the section  
   *“Evaluating Model Accuracy”* in the topic *“Linear Regression. Evaluating Regression Quality”*.  
   Draw conclusions.
