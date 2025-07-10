import pandas as pd
import numpy as np
import requests
import io
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the Autos dataset
url = "https://raw.githubusercontent.com/goitacademy/MACHINE-LEARNING-NEO/refs/heads/main/datasets/mod_05_topic_10_various_data.pkl"
response = requests.get(url)
data = pd.read_pickle(io.BytesIO(response.content))

# Extract the 'autos' DataFrame if present
if isinstance(data, dict) and 'autos' in data:
    data = data['autos']
else:
    raise KeyError("Dataset 'autos' not found.")

# 2. Identify discrete features
discrete_features = [
    col for col in data.columns
    if data[col].dtype == 'object' or data[col].nunique() < 20
]

# Encode discrete features for mutual information calculation
label_encoders = {}
for col in discrete_features:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# 3. Calculate mutual information
if 'price' in data.columns:
    X = data.drop(columns=['price'])
    y = data['price']
    mutual_info = mutual_info_regression(X, y, random_state=42)
    mutual_info_series = pd.Series(mutual_info, index=X.columns)

    # 4. Prepare data using TargetEncoder for the model
    encoder = TargetEncoder(cols=discrete_features)
    X_encoded = encoder.fit_transform(X, y)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_encoded, y)
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)

    # 5. Normalize rankings using percentile rank
    ranked_mutual_info = mutual_info_series.rank(pct=True)
    ranked_importances = feature_importances.rank(pct=True)

    comparison_df = pd.DataFrame({
        'Feature': X.columns,
        'Mutual Info Rank': ranked_mutual_info,
        'Feature Importance Rank': ranked_importances
    })

    # 6. Visualization: grouped bar plot
    melted_df = comparison_df.melt(
        id_vars='Feature',
        var_name='Metric',
        value_name='Rank'
    )

    sns.catplot(
        data=melted_df,
        kind='bar',
        x='Feature',
        y='Rank',
        hue='Metric',
        height=6,
        aspect=2
    )
    plt.title("Comparison of Feature Importance and Mutual Information")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

else:
    print("Target variable 'price' not found.")
