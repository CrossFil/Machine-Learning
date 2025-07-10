import os
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings('ignore')

# 1. Load data
train_url = 'https://raw.githubusercontent.com/goitacademy/MACHINE-LEARNING-NEO/main/competition/final_proj_data.csv'
test_url  = 'https://raw.githubusercontent.com/goitacademy/MACHINE-LEARNING-NEO/main/competition/final_proj_test.csv'
sub_url   = 'https://raw.githubusercontent.com/goitacademy/MACHINE-LEARNING-NEO/main/competition/final_proj_sample_submission.csv'

train = pd.read_csv(train_url)
test  = pd.read_csv(test_url)
submission = pd.read_csv(sub_url)

# 2. Drop columns with >30% missing values
missing_pct = train.isnull().mean() * 100
cols_to_drop = missing_pct[missing_pct > 30].index
train = train.drop(columns=cols_to_drop)
test  = test.drop(columns=[c for c in cols_to_drop if c in test.columns])

# 3. Identify feature types
target = 'y'
num_feats = train.select_dtypes(include=['int64','float64']).drop(target, axis=1).columns
cat_feats = train.select_dtypes(include=['object']).columns

# 4. Build preprocessing pipelines
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler',  StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_feats),
    ('cat', cat_pipeline, cat_feats)
])

# 5. Compute class weights
classes = np.unique(train[target])
weights = compute_class_weight('balanced', classes=classes, y=train[target])
class_weights = dict(zip(classes, weights))

# 6. Initial pipeline with SMOTE and CatBoost
initial_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote',        SMOTE(random_state=42)),
    ('clf',          CatBoostClassifier(
                        random_state=42,
                        verbose=0,
                        class_weights=class_weights
                    ))
])

# Evaluate before hyperparameter tuning
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(
    initial_pipeline,
    train.drop(columns=target),
    train[target],
    scoring='balanced_accuracy',
    cv=cv
)
print("Balanced accuracy before tuning:", scores, f"Mean: {scores.mean():.4f}")

# 7. Hyperparameter tuning with RandomizedSearchCV
param_dist = {
    'clf__iterations':    [100, 200],
    'clf__depth':         [4, 6],
    'clf__learning_rate': [0.01, 0.1],
}

search_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote',        SMOTE(random_state=42)),
    ('clf',          CatBoostClassifier(
                        random_state=42,
                        verbose=0,
                        class_weights=class_weights
                    ))
])

search = RandomizedSearchCV(
    search_pipeline,
    param_distributions=param_dist,
    n_iter=5,
    scoring='balanced_accuracy',
    cv=cv,
    n_jobs=-1,
    random_state=42
)
search.fit(train.drop(columns=target), train[target])

best_model = search.best_estimator_
scores_tuned = cross_val_score(
    best_model,
    train.drop(columns=target),
    train[target],
    scoring='balanced_accuracy',
    cv=cv
)
print("Balanced accuracy after tuning:", scores_tuned, f"Mean: {scores_tuned.mean():.4f}")

# 8. Train on full training data and predict on test set
best_model.fit(train.drop(columns=target), train[target])
preds = best_model.predict(test)

# 9. Save submission
submission['y'] = preds
output_file = 'final_submission.csv'
submission.to_csv(output_file, index=False)
print(f"Submission saved to {output_file}")
