# Supervised Learning Algorithms

## Task 1 Description - Linear Regression. Evaluating Regression Quality
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

 

## Task 2 Description - Logistic Regression. Evaluating Classification Quality

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


## Task 3 Description - kNN Regressor & SVM. Model Evaluation

1. **Import the required packages.**

2. **Load the datasets:**
   - `mod_04_hw_train_data.csv` — training data.
   - `mod_04_hw_valid_data.csv` — validation data.
   - The target variable is `Salary`.

3. **Perform exploratory data analysis (EDA):**
   - Assess the completeness, data types, and distributions of all features.
   - Analyze the usefulness and relevance of the available features.

4. **Preprocess the data:**
   - Handle missing values.
   - Convert `Date_Of_Birth` to age.
   - Drop irrelevant columns (`Name`, `Phone_Number`, `Date_Of_Birth`).
   - Apply scaling to numeric features using `StandardScaler`.
   - Encode categorical features using `TargetEncoder`.

5. **Train a model using `KNeighborsRegressor` from `sklearn`:**
   - Tune hyperparameters via `GridSearchCV`.

6. **Repeat data preparation steps for the validation set.**

7. **Make predictions on the validation set:**
   - Evaluate model accuracy using `mean_absolute_percentage_error`.

8. **Build and evaluate an additional model using `SVR` (Support Vector Regression).**
   - Tune hyperparameters (`C`, `kernel`, `degree`) via `GridSearchCV`.
   - Compare the performance of SVM to kNN.

## Task 4 Description - Mutual Information vs. Feature Importance (Autos Dataset)

### Task Overview

In this task, we analyze feature relevance for predicting car prices (`price`) using two different approaches:

1. **Mutual Information** — quantifies the nonlinear dependency between each feature and the target variable.
2. **Model-Based Feature Importance** — calculated using a `RandomForestRegressor`.

### Steps Performed

1. **Dataset Loading**  
   The `autos` dataset was loaded from a `.pkl` file containing multiple dataframes. The target variable is `price`.

2. **Discrete Feature Identification**  
   Features were treated as discrete if:
   - they had `object` dtype, or
   - they had fewer than 20 unique values.

3. **Encoding for Mutual Information**  
   All categorical/discrete features were label-encoded before calculating mutual information using `mutual_info_regression`.

4. **Model Training**  
   A `RandomForestRegressor` model was trained using data transformed with `TargetEncoder` for categorical features.

5. **Ranking Feature Relevance**  
   Both mutual information scores and feature importances were normalized using `.rank(pct=True)` to compare them on the same scale.

6. **Visualization**  
   A grouped bar plot was created using `seaborn.catplot()` to visually compare the ranks of each feature from both methods.

### Conclusion

- **Agreement**: Features ranked highly by mutual information were often also ranked highly by Random Forest, confirming their predictive relevance.
- **Discrepancies**: Differences may arise due to interactions between features, which mutual information does not account for.
- **Usage**: Features with low ranks in both methods can potentially be dropped. The most relevant features can be used for further modeling or dimensionality reduction.

# Unsupervised Learning Algorithms

## Task 5 Description - Clustering of Concrete Formulations

This task belongs to the **Unsupervised Learning Algorithms** module. We apply k‑means clustering to concrete recipe data to uncover patterns in material composition.

### Steps

1. **Load the dataset.**  
   Retrieve the `concrete` DataFrame from a remote pickle file.

2. **Create `Components` feature.**  
   Count non-zero ingredient values per row.

3. **Normalize features.**  
   Scale all columns (including `Components`) with `StandardScaler`.

4. **Determine optimal clusters.**  
   Use `KElbowVisualizer` (Yellowbrick) on k = 2…10 with distortion metric.

5. **Apply k-means clustering.**  
   Fit `KMeans(n_clusters=optimal)` and assign each sample a cluster label.

6. **Compute cluster summary.**  
   For each cluster, calculate:
   - Median of each original feature.
   - Average `Components` count.
   - Number of recipes.

7. **Analyze results.**  
   Interpret how material combinations group together and impact concrete strength.

### Output

A summary table with per-cluster medians, average component counts, and sample counts, providing insights into formulation patterns and key influencing factors.

# Final Project: Kaggle Competition “ML Fundamentals and Applications 2025-01-09”

**Kaggle competition:** https://www.kaggle.com/competitions/ml-fundamentals-and-applications-2025-01-09/leaderboard  
**Username:** Cross Fil  

## Objective

Develop a high‑performance binary classification model that can handle a large number of features.

## Data Description

- **Training set:** `final_proj_data.csv` (10 000 rows, 230 features)
- **Validation set:** `final_proj_test.csv`
- **Sample submission:** `final_proj_sample_submission.csv`
- **Features:**  
  - 190 numerical  
  - 40 categorical  
- **Target:** binary variable `y`

## Workflow

1. **Exploratory Data Analysis (EDA)**  
   - Inspect data shapes, types, distributions  
   - Identify missing values  

2. **Missing Value Strategy**  
   - Decide which features to impute, drop, or engineer  

3. **Categorical Encoding**  
   - Count unique categories per feature  
   - Choose optimal encoding (e.g. target, one‑hot, ordinal)  

4. **Class Balance**  
   - Evaluate `y` distribution  
   - Apply balancing (SMOTE, undersampling) if needed  

5. **Feature Scaling**  
   - Normalize/standardize numerical features as required by chosen models  

6. **Dimensionality Reduction**  
   - Experiment with PCA, feature selection, embedding  

7. **Model Development**  
   - Try different algorithms and ensembles  
   - Hyperparameter tuning with `GridSearchCV` or `BayesSearchCV`  

8. **Pipeline & Cross‑Validation**  
   - Build an `sklearn.Pipeline` (optionally with `imblearn.Pipeline` for balancing)  
   - Perform stratified k‑fold CV, measure balanced accuracy  

9. **Final Model Training**  
   - Retrain best model on combined training + validation data before test predictions  

10. **Submission File**  
    - Create `submission.csv` with format:  
      ```
      index,y
      0,0
      1,0
      2,1
      ...
      ```  
    - Metric on leaderboard: **Balanced Accuracy**

11. **Iterate & Improve**  
    - Analyze leaderboard feedback  
    - Continue feature engineering, model tuning, ensembling

