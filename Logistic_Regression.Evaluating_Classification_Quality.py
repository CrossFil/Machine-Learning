import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv('/Users/admin/Applications/weatherAUS.csv')

# Initial inspection
print(data.head())
print(data.info())

# Drop columns with more than 30% missing values
threshold = 0.3
data = data.loc[:, data.isnull().mean() < threshold]
print(f"After dropping columns: {data.shape}")

# Drop rows with missing target
data = data.dropna(subset=['RainTomorrow'])

# Convert 'Date' to datetime and extract Year and Month
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month.astype('category')
data = data.drop(columns=['Date'])

# Split dataset by Year: last year for test set
max_year = data['Year'].max()
train_data = data[data['Year'] < max_year].copy()
test_data = data[data['Year'] == max_year].copy()
print(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")

# Separate numerical and categorical features
numerical_features = train_data.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = train_data.select_dtypes(include=['object', 'category']).columns.tolist()

# Remove target from categorical and move 'Year' to numerical
categorical_features.remove('RainTomorrow')
if 'Year' in categorical_features:
    categorical_features.remove('Year')
if 'Year' not in numerical_features:
    numerical_features.append('Year')

print(f"Numerical features: {numerical_features}")
print(f"Categorical features: {categorical_features}")

# Impute missing values for numerical features
imputer = SimpleImputer(strategy='mean')
train_data[numerical_features] = imputer.fit_transform(train_data[numerical_features])
test_data[numerical_features] = imputer.transform(test_data[numerical_features])

# Scale numerical features
scaler = StandardScaler()
train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])
test_data[numerical_features] = scaler.transform(test_data[numerical_features])

# Impute missing values for categorical features
cat_imputer = SimpleImputer(strategy='most_frequent')
train_data[categorical_features] = cat_imputer.fit_transform(train_data[categorical_features])
test_data[categorical_features] = cat_imputer.transform(test_data[categorical_features])

# Encode categorical features, drop one binary column if applicable
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary').set_output(transform='pandas')
train_categorical = encoder.fit_transform(train_data[categorical_features])
test_categorical = encoder.transform(test_data[categorical_features])

# Check encoded feature names (optional)
print(train_categorical.columns)

# Combine numeric and categorical features
X_train = np.hstack([train_data[numerical_features], train_categorical])
X_test = np.hstack([test_data[numerical_features], test_categorical])

# Encode target variable
y_train = train_data['RainTomorrow'].map({'No': 0, 'Yes': 1}).values
y_test = test_data['RainTomorrow'].map({'No': 0, 'Yes': 1}).values

# Train logistic regression model
model = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)

# Model evaluation
pred = model.predict(X_test)
print(classification_report(y_test, pred))

#               precision    recall  f1-score   support
#
#            0       0.92      0.82      0.87      6703
#            1       0.51      0.72      0.60      1763
#
#     accuracy                           0.80      8466
#    macro avg       0.71      0.77      0.73      8466
# weighted avg       0.83      0.80      0.81      8466

# Conclusion:
# The model achieves approximately 80% accuracy. It performs well in predicting the majority class (no rain),
# and maintains reasonable recall (0.72) for the minority class (rain).
# The use of proper data splitting (by year), imputation, scaling, and binary drop in encoding helps avoid data leakage
# and ensures a more reliable evaluation setting. However, precision for rain remains modest (0.51),
# which may require further optimization or class balancing techniques.
