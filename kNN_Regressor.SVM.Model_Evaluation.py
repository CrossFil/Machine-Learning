import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_percentage_error

# 1. Load datasets
train_data = pd.read_csv('https://raw.githubusercontent.com/goitacademy/MACHINE-LEARNING-NEO/main/datasets/mod_04_hw_train_data.csv')
valid_data = pd.read_csv('https://raw.githubusercontent.com/goitacademy/MACHINE-LEARNING-NEO/main/datasets/mod_04_hw_valid_data.csv')

# 2. Initial data analysis
print(train_data.info())
print(train_data.describe())
print(train_data.isnull().sum())

# 3. Missing values handling
train_data['Experience'] = train_data['Experience'].fillna(train_data['Experience'].median())
valid_data['Experience'] = valid_data['Experience'].fillna(train_data['Experience'].median())

def calculate_age(birthdate):
    current_year = pd.Timestamp.now().year
    return current_year - pd.to_datetime(birthdate, errors='coerce', dayfirst=True).dt.year

train_data['Age'] = calculate_age(train_data['Date_Of_Birth'])
valid_data['Age'] = calculate_age(valid_data['Date_Of_Birth'])

train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
valid_data['Age'] = valid_data['Age'].fillna(valid_data['Age'].median())

# Drop irrelevant columns
train_data = train_data.drop(columns=['Name', 'Phone_Number', 'Date_Of_Birth'])
valid_data = valid_data.drop(columns=['Name', 'Phone_Number', 'Date_Of_Birth'])

# 4. Feature preparation
numerical_features = ['Experience', 'Age']
categorical_features = ['Qualification', 'University', 'Role', 'Cert']
target = 'Salary'

X_train = train_data.drop(columns=target)
y_train = train_data[target]
X_valid = valid_data.drop(columns=target)
y_valid = valid_data[target]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', TargetEncoder(), categorical_features)
    ]
)

# 5. KNN regression
knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', KNeighborsRegressor())
])

param_grid_knn = {
    'regressor__n_neighbors': range(3, 21, 2),
    'regressor__weights': ['uniform', 'distance'],
    'regressor__p': [1, 2]
}

knn_search = GridSearchCV(knn_pipeline, param_grid_knn, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
knn_search.fit(X_train, y_train)

# 6. SVM regression
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR())
])

param_grid_svm = {
    'regressor__C': [0.1, 1, 10],
    'regressor__kernel': ['linear', 'rbf', 'poly'],
    'regressor__degree': [2, 3]
}

svm_search = GridSearchCV(svm_pipeline, param_grid_svm, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
svm_search.fit(X_train, y_train)

# 7. Predictions and evaluation
knn_best = knn_search.best_estimator_
svm_best = svm_search.best_estimator_

y_pred_knn = knn_best.predict(X_valid)
y_pred_svm = svm_best.predict(X_valid)

knn_mape = mean_absolute_percentage_error(y_valid, y_pred_knn)
svm_mape = mean_absolute_percentage_error(y_valid, y_pred_svm)

print(f"Validation MAPE (KNN): {knn_mape:.2%}")
print(f"Best parameters (KNN): {knn_search.best_params_}")
print(f"Validation MAPE (SVM): {svm_mape:.2%}")
print(f"Best parameters (SVM): {svm_search.best_params_}")

# Validation MAPE (KNN): 11.08%
# Best parameters (KNN): {'regressor__n_neighbors': 3, 'regressor__p': 1, 'regressor__weights': 'distance'}
# Validation MAPE (SVM): 13.75%
# Best parameters (SVM): {'regressor__C': 0.1, 'regressor__degree': 2, 'regressor__kernel': 'linear'}

# Final conclusion:
# KNN outperformed SVM in this task, achieving a lower validation MAPE (~11.08% vs ~13.75%).
# Best KNN params: {'n_neighbors': 3, 'p': 1, 'weights': 'distance'}
# Best SVM params: {'C': 0.1, 'degree': 2, 'kernel': 'linear'}
# Despite solid preprocessing and tuning, the accuracy may still be insufficient for some business cases.
# Potential improvements include feature engineering, using additional data, or trying alternative models.
