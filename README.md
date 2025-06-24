# Supervised Learning Algorithms

## Task 1 Instructions - Linear Regression. Evaluating Regression Quality
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

 

## Task 2 Instructions - Logistic Regression — Evaluating Classification Quality

1. Import the required packages.

2. Load the *Rain in Australia* dataset, as shown in the section  
   *“Applying Logistic Regression. EDA of the Rain in Australia Dataset”*  
   from the topic *“Logistic Regression. Evaluating Classification Quality”*.

3. Prepare the dataset:

   - 3.1 Remove features with a large number of missing values.  
   - 3.2 Create separate subsets for numeric and categorical features.  
   - 3.3 Convert the `Date` column to `datetime`, and extract new columns: `Year` and `Month`.  
   - 3.4 Move `Year` to the numeric features subset, keep `Month` in the categorical subset.  
   - 3.5 Split the dataset:
     - Assign all rows from the **latest year** to the test set.
     - Use the remaining data for training.
     - Use boolean indexing in `pandas` to implement the split.

4. Impute missing values using `SimpleImputer` from `sklearn`.

5. Normalize numeric features using `StandardScaler` from `sklearn`.

6. Encode categorical features using `OneHotEncoder` from `sklearn`.

7. Combine the numeric and encoded categorical features and train a model using `LogisticRegression` from `sklearn`.

8. Evaluate the model using `classification_report()` from `sklearn.metrics`.  
   Compare the metrics to those from the earlier section  
   *“Applying Logistic Regression. Model Training and Evaluation”*,  
   and summarize your findings.

