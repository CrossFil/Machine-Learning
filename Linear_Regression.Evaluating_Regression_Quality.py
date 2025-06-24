from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import zscore
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Load the dataset
california_housing = fetch_california_housing(as_frame=True)
data = california_housing.frame

# Compute Z-scores
columns_to_check = ['AveRooms', 'AveBedrms', 'AveOccup', 'Population']
z_scores = data[columns_to_check].apply(zscore)

# Identify rows with outliers
outliers = (z_scores.abs() > 3).any(axis=1)

# Remove outliers
data_cleaned = data[~outliers]

# Compute the correlation matrix
correlation_matrix = data_cleaned.corr()

# Select highly correlated column pairs (correlation > 0.9)
high_corr_pairs = correlation_matrix[correlation_matrix.abs() > 0.9].stack().reset_index()
high_corr_pairs = high_corr_pairs[high_corr_pairs['level_0'] != high_corr_pairs['level_1']]

# Drop 'AveBedrms' instead of 'AveRooms'
data_cleaned = data_cleaned.drop(columns=['AveBedrms'])

# Split into features (X) and target variable (y)
X = data_cleaned.drop(columns=['MedHouseVal'])
y = data_cleaned['MedHouseVal']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = model.predict(X_test_scaled)

# R-squared score
r_sq_upd = model.score(X_train_scaled, y_train)

# Mean Absolute Error
mae_upd = mean_absolute_error(y_test, y_pred)

# Mean Absolute Percentage Error
mape_upd = mean_absolute_percentage_error(y_test, y_pred)

# Print results
print(f'R2: {r_sq_upd:.2f} | MAE: {mae_upd:.2f} | MAPE: {mape_upd:.2f}')

# Generate polynomial features
poly = PolynomialFeatures(degree=3, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Train a model with polynomial features
model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)
y_pred_poly = model_poly.predict(X_test_poly)

# Evaluate the model
r_sq_poly = model_poly.score(X_train_poly, y_train)
mae_poly = mean_absolute_error(y_test, y_pred_poly)
mape_poly = mean_absolute_percentage_error(y_test, y_pred_poly)

print(f'R2 (poly): {r_sq_poly:.2f} | MAE: {mae_poly:.2f} | MAPE: {mape_poly:.2f}')

# Grid search for polynomial degree and ridge regularization
param_grid = {
    'polynomialfeatures__degree': [2, 3, 4],
    'ridge__alpha': [0.1, 1, 10]
}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('polynomialfeatures', PolynomialFeatures(include_bias=False)),
    ('ridge', Ridge())
])

grid_search = GridSearchCV(pipeline, param_grid, scoring='neg_mean_absolute_error', cv=5)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Evaluate best model
r_sq_best = best_model.score(X_train, y_train)
mae_best = mean_absolute_error(y_test, y_pred_best)
mape_best = mean_absolute_percentage_error(y_test, y_pred_best)

print(f'R2 (Best Model): {r_sq_best:.2f} | MAE: {mae_best:.2f} | MAPE: {mape_best:.2f}')

# R2: 0.64 | MAE: 0.51 | MAPE: 0.30
# R2 (poly): 0.74 | MAE: 0.43 | MAPE: 0.25
# R2 (Best Model): 0.78 | MAE: 0.41 | MAPE: 0.24

# Conclusion:
# The baseline linear model shows reasonable performance, while using polynomial features with Ridge regularization improves the results.
# Grid search helped to select the optimal degree and regularization strength, reducing error metrics further.
