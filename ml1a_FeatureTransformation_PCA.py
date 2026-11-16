import pandas as pd
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['Target'] = wine.target

X = df.drop('Target', axis=1)
y = df['Target']

X = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Target'] = y.values

print("Explained Variance:", pca.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
for label in pca_df['Target'].unique():
    data = pca_df[pca_df['Target'] == label]
    plt.scatter(data['PC1'], data['PC2'], label=label)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Wine Dataset")
plt.legend()
plt.show()

# ==========================================
# CODE EXPLANATION
# ==========================================
# - Imports pandas, matplotlib, and scikit-learn modules (load_wine dataset, StandardScaler, and PCA).
# - Loads the built-in Wine dataset (13 chemical features, 3 wine classes) and converts it into a Pandas DataFrame.
# - Separates input features (X) from the target class labels (y).
# - StandardScaler().fit_transform(X) standardizes all features to have mean = 0 and variance = 1.
# - PCA(n_components=2) reduces the 13-dimensional feature space down to 2 uncorrelated Principal Components (PC1 and PC2).
# - Stores the transformed coordinates and target labels in a new DataFrame `pca_df`.
# - Prints `pca.explained_variance_ratio_`, showing how much information (variance) PC1 and PC2 retain.
# - Loops through each unique wine class (0, 1, 2) and plots a 2D scatter plot to visualize class separation.

# ==========================================
# VIVA QUESTIONS & ANSWERS
# ==========================================
# Q1: What is PCA (Principal Component Analysis)?
# A1: An unsupervised dimensionality reduction technique that transforms correlated features into a smaller set of uncorrelated variables called Principal Components while preserving maximum variance.
#
# Q2: Why is `StandardScaler` applied before PCA?
# A2: PCA is sensitive to feature scales; without standardization, features with larger numerical ranges (like Proline) would dominate the principal components.
#
# Q3: What does `n_components=2` do?
# A3: It reduces the original 13 features down to 2 dimensions (PC1 and PC2) so the dataset can be easily visualized on a 2D scatter plot.
#
# Q4: What is `explained_variance_ratio_`?
# A4: It indicates the percentage of the dataset's total variance (information) captured by each individual principal component (PC1 captures the most, followed by PC2).
#
# Q5: Is PCA supervised or unsupervised?
# A5: Unsupervised—it only uses input features `X` to find directions of maximum variance and ignores the target labels `y` during transformation.
#
# Q6: What are Eigenvalues and Eigenvectors in the context of PCA?
# A6: Eigenvectors define the directions (axes) of the new principal components, while Eigenvalues represent the magnitude of variance along each direction.
#
# Q7: What is the difference between Feature Selection and Dimensionality Reduction (PCA)?
# A7: Feature selection picks a subset of original features as-is, whereas PCA creates brand-new synthetic features (linear combinations of original features).