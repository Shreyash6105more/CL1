import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)

X = data.iloc[:, 0:4]

print(data.isnull().sum())

wcss = []

for k in range(1, 11):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    wcss.append(model.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.axvline(x=3, color='red', linestyle='--', label='K = 3')
plt.legend()
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

km = KMeans(n_clusters=3, init='k-means++',
            random_state=42, n_init=10)

clusters = km.fit_predict(X)

data["Cluster"] = clusters

print(data.head())

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters)
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            color='red', marker='X', s=150)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means Clustering")
plt.show()

# ==========================================
# CODE EXPLANATION
# ==========================================
# - Imports pandas, matplotlib, and scikit-learn's `load_iris` dataset and `KMeans` algorithm.
# - Loads the Iris dataset into a DataFrame and selects all 4 numerical features (`X = data.iloc[:, 0:4]`).
# - Checks for missing values using `isnull().sum()`.
# - Elbow Method: Loops `k` from 1 to 10, fits K-Means for each `k`, and stores `model.inertia_` (WCSS) in a list.
# - Plots WCSS against `k` with a vertical dashed line at `K = 3` where the curve bends ("elbow"), indicating the optimal cluster count.
# - Trains the final K-Means model with `n_clusters=3` and `init='k-means++'`, assigning each sample a cluster label (0, 1, or 2) via `fit_predict()`.
# - Appends the predicted `Cluster` labels to `data` and previews the first 5 rows.
# - Plots a 2D scatter plot of Sepal Length vs. Sepal Width colored by cluster, marking cluster centroids with large red `'X'` markers.

# ==========================================
# VIVA QUESTIONS & ANSWERS
# ==========================================
# Q1: What is K-Means Clustering, and is it supervised or unsupervised?
# A1: It is an unsupervised machine learning algorithm that partitions unlabeled data into `K` distinct clusters by grouping points closest to their respective cluster centroids.
#
# Q2: What is WCSS (`model.inertia_`)?
# A2: Within-Cluster Sum of Squares—it measures the sum of squared distances between each data point and its assigned cluster centroid. Lower WCSS means tighter, more compact clusters.
#
# Q3: How does the Elbow Method determine the optimal value of `K`?
# A3: As `K` increases, WCSS drops rapidly at first and then flattens out. The "elbow" point (here `K = 3`) marks where adding more clusters yields diminishing returns in variance reduction.
#
# Q4: What does `init='k-means++'` do?
# A4: It initializes centroids smartly by placing them far apart from one another, leading to faster convergence and avoiding poor local optima compared to random initialization.
#
# Q5: What does `n_init=10` mean?
# A5: K-Means runs 10 separate times with different initial centroid seeds and selects the best clustering result with the lowest WCSS.
#
# Q6: What are `km.cluster_centers_`?
# A6: The coordinates of the final centroids (mean feature values) for each of the `K` clusters.
#
# Q7: What is the difference between K-Means and KNN (K-Nearest Neighbors)?
# A7: K-Means is an unsupervised clustering algorithm that groups unlabeled data into `K` clusters, whereas KNN is a supervised classification/regression algorithm that predicts labels based on the `K` closest labeled neighbors.