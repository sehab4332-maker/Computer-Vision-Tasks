import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate hypothetical data points
X, _ = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=0.75,
    random_state=42
)

# Apply K-Means clustering
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)
kmeans.fit(X)

labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Display the clustering result
plt.figure(figsize=(8, 6))
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=labels,
    cmap="viridis",
    s=35,
    label="Data points"
)
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    c="red",
    marker="X",
    s=220,
    edgecolors="black",
    label="Centroids"
)

plt.title("K-Means Clustering on Hypothetical Points")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("Number of clusters:", kmeans.n_clusters)
print("Cluster centers:")
print(centroids)
print("Inertia:", kmeans.inertia_)
