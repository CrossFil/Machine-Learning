# Machine-Learning
Supervised Learning Algorithms (Part 1) – Completed Steps
Imported Necessary Packages
Imported the required libraries, including pandas, numpy, scipy, and sklearn, to handle data processing, analysis, and modeling.
Loaded California Housing Dataset
The dataset was loaded as described in the section "Linear Regression: Quality Evaluation of Regression" (California Housing dataset). Exploratory Data Analysis (EDA) was performed earlier, so it was not repeated here.
Data Preprocessing
Outlier Removal:
Used the zscore() function from scipy to identify and remove outliers in the AveRooms, AveBedrms, AveOccup, and Population columns.
Applied apply() from pandas to calculate Z-scores.
Removed rows where any value in the specified columns exceeded the outlier threshold using any().
Feature Selection:
Removed one of the highly correlated features from the dataset based on the correlation matrix. This was identified in the earlier analysis.
Split Data into Training and Test Sets
Utilized the train_test_split() method from sklearn to divide the data into training and testing subsets.
Feature Normalization
Applied normalization to the dataset using StandardScaler from sklearn to standardize feature values.
Built a Linear Regression Model
Constructed a regression model using the LinearRegression class from sklearn.
Evaluated Model Performance
Calculated key performance metrics:
R-squared (R²) – Coefficient of determination.
MAE – Mean Absolute Error.
MAPE – Mean Absolute Percentage Error.
Comparison with Earlier Results
Compared the metrics obtained from the current model with those from the previous implementation ("Evaluating Model Accuracy" section). Key observations and conclusions were documented.
