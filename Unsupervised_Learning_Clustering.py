import requests
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# 1. Load the Concrete dataset
url = "https://raw.githubusercontent.com/goitacademy/MACHINE-LEARNING-NEO/refs/heads/main/datasets/mod_05_topic_10_various_data.pkl"
response = requests.get(url)

if response.status_code == 200:
    data = pd.read_pickle(io.BytesIO(response.content))
    print("Data loaded successfully!")
else:
    raise Exception(f"Error loading data: {response.status_code}")

# If the data is a dict, extract the 'concrete' DataFrame
if isinstance(data, dict):
    print("Dataset keys:", data.keys())
    df = data['concrete'] if 'concrete' in data else None
    if df is None:
        raise Exception("Key 'concrete' not found in the data.")
else:
    df = data

# 2. Create the 'Components' feature
# Components = count of non-zero values per row
component_columns = df.columns
if 'ConcreteCompressiveStrength' in df.columns:
    component_columns = df.drop('ConcreteCompressiveStrength', axis=1).columns

df['Components'] = df[component_columns].gt(0).sum(axis=1)

# 3. Normalize the data (excluding 'Components')
scaler = StandardScaler()
normalized_features = scaler.fit_transform(df.drop(columns=['Components']))

# Normalize 'Components' and append
scaled_components = scaler.fit_transform(df[['Components']])
normalized_data = pd.concat([
    pd.DataFrame(normalized_features, columns=df.drop(columns=['Components']).columns),
    pd.DataFrame(scaled_components, columns=['Components'])
], axis=1).values

# 4. Determine the optimal number of clusters with KElbowVisualizer
model = KMeans(random_state=42)
visualizer = KElbowVisualizer(model, k=(2, 10), metric='distortion', timings=False)
visualizer.fit(normalized_data)
visualizer.show()

# Optimal number of clusters
optimal_clusters = visualizer.elbow_value_
print(f"Optimal number of clusters: {optimal_clusters}")

# 5. Apply k-means clustering
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
labels = kmeans.fit_predict(normalized_data)
df['Cluster'] = labels

# 6. Compute descriptive statistics per cluster
cluster_summary = df.groupby('Cluster').median()

# Add average component count per cluster
cluster_summary['Components_Count'] = df.groupby('Cluster')['Components'].mean()

# 7. Add recipe counts per cluster
cluster_summary['Recipe_Count'] = df['Cluster'].value_counts().sort_index()

print("\nCluster summary statistics:")
print(cluster_summary)

# Analysis:
# - Optimal number of clusters: changed from 6 to 5, indicating that when 'Components' is equally weighted,
#   the data form more coherent groups with 5 clusters.
# - Cluster distribution: Cluster 2 now contains 112 recipes (up from a previous maximum of 80),
#   reflecting a different interpretation of feature relationships.
# - Components_Count values: became more balanced, showing improved consideration of ingredient counts in clustering.
#
# Scaling the 'Components' feature made its influence comparable to other variables.
# This is critical for algorithms like K-Means, where feature weights depend on their scale.
