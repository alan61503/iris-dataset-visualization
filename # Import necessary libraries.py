# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # 3 clusters because we have 3 species in the Iris dataset
kmeans.fit(X)

# Get cluster labels (predicted cluster for each sample)
labels = kmeans.labels_

# Get cluster centers
centroids = kmeans.cluster_centers_

# Visualize the clusters (using only the first two features: Sepal Length and Sepal Width)
plt.figure(figsize=(8, 6))
plt.scatter(X[labels == 0, 0], X[labels == 0, 1], color='red', label='Cluster 1')
plt.scatter(X[labels == 1, 0], X[labels == 1, 1], color='blue', label='Cluster 2')
plt.scatter(X[labels == 2, 0], X[labels == 2, 1], color='green', label='Cluster 3')

# Plot the centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='black', s=200, label='Centroids')

plt.title('K-means Clustering on Iris Dataset (2D view of Sepal Length and Sepal Width)')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()

# Print the cluster centers (centroids)
print("Cluster centers (centroids):")
print(centroids)

# Optional: If you want to add labels to the original dataset for comparison
iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['Cluster'] = labels
print(iris_df.head())