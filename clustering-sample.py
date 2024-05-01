import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import norm

# Generate synthetic data for three clusters
np.random.seed(42)
num_data_points = 100000
centers = [[2, 2], [6, 6], [10, 10]]
X, _ = make_blobs(n_samples=[num_data_points//2, num_data_points//4, num_data_points//4], centers=centers, cluster_std=1.0)

# Visualize the original data
plt.figure(figsize=(48, 8))
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], s=10, alpha=0.7)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid(True)

# Fuzzy C-means clustering
def fcm(X, n_clusters, m=2, max_iter=100, tol=1e-4):
    # Initialize cluster centers randomly
    centroids = np.random.rand(n_clusters, X.shape[1])
    
    for _ in range(max_iter):
        # Compute distances and membership degrees
        distances = np.linalg.norm(X[:, None] - centroids, axis=-1)
        membership = 1 / distances ** (2 / (m - 1))
        membership = membership / np.sum(membership, axis=-1, keepdims=True)
        
        # Update cluster centers
        new_centroids = np.dot(membership.T, X) / np.sum(membership, axis=0, keepdims=True).T
        
        # Check convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        
        centroids = new_centroids
    
    return centroids, membership

# Perform FCM clustering
k = int(input("How many clusters? "))  # Number of clusters
fcm_centroids, fcm_membership = fcm(X, n_clusters=k)
fcm_labels = np.argmax(fcm_membership, axis=1)

# Perform K-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X)
kmeans_labels = kmeans.labels_
kmeans_centroids = kmeans.cluster_centers_

# Plot FCM clustering
plt.subplot(1, 3, 2)
for i in range(k):
    plt.scatter(X[fcm_labels == i, 0], X[fcm_labels == i, 1], s=10, alpha=0.7, label=f'Cluster {i+1}')
plt.scatter(fcm_centroids[:, 0], fcm_centroids[:, 1], marker='x', s=200, color='black', label='Centroids')
plt.title('FCM Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

# Plot K-means clustering
plt.subplot(1, 3, 3)
for i in range(k):
    plt.scatter(X[kmeans_labels == i, 0], X[kmeans_labels == i, 1], s=10, alpha=0.7, label=f'Cluster {i+1}')
plt.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], marker='x', s=200, color='black', label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)
plt.show()
