import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from lib.data_source import load_vector_data

X, y = load_vector_data()

# Convert lists to NumPy arrays
X = np.array(X)
# For labels, we need numerical representation for `cmap` in scatter plot.
# We'll create a mapping from unique names to integers.
unique_labels = list(set(y))
label_to_int = {label: i for i, label in enumerate(unique_labels)}
y_numeric = np.array([label_to_int[label] for label in y])

print(f"Loaded {len(y)} documents from the database.")
print(f"Embeddings shape: {X.shape}") # e.g., (300, 128) if your vectors are 128-dim


# --- Check if data was loaded successfully before proceeding ---
if X.size == 0:
    print("Exiting: No data available for PCA.")
    exit()

# --- 2. Perform PCA ---
n_components = 2 # We want to reduce to 2 dimensions for easy visualization

# Initialize PCA with the number of components you want to keep
pca = PCA(n_components=n_components)

# Fit PCA on the data and transform the data
X_pca = pca.fit_transform(X)

print(f"Reduced data shape (after PCA): {X_pca.shape}")

# --- 3. Visualize the PCA results with labels beside points ---
plt.figure(figsize=(12, 10)) # Adjust figure size to accommodate labels

# Scatter plot the transformed data
# Using a colormap to differentiate points even with direct labels
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_numeric, cmap='tab20', s=70, alpha=0.7, edgecolor='k')

# Annotate each point with its label (name)
for i, (x_coord, y_coord) in enumerate(X_pca):
    plt.annotate(
        y[i], # The label (name) for the point
        (x_coord, y_coord), # The (x,y) coordinates of the point
        textcoords="offset points", # How to position the text
        xytext=(5, 5), # Offset from the point (5 points right, 5 points up)
        ha='left', # Horizontal alignment of the text
        va='bottom', # Vertical alignment of the text
        fontsize=8, # Font size for the label
        color='dimgray', # Color of the label text
        alpha=0.9
    )

plt.title(f'PCA of Vector Database Embeddings to {n_components} Dimensions with Point Labels')
plt.xlabel(f'Principal Component 1 (explains {pca.explained_variance_ratio_[0]*100:.2f}% variance)')
plt.ylabel(f'Principal Component 2 (explains {pca.explained_variance_ratio_[1]*100:.2f}% variance)')

plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# # --- 4. Explained Variance Ratio (important for understanding PCA) ---
print("\nExplained variance ratio by each principal component:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"  Principal Component {i+1}: {ratio*100:.2f}%")

print(f"\nTotal explained variance by {n_components} components: "
      f"{pca.explained_variance_ratio_.sum()*100:.2f}%")

# # --- 5. Accessing the Components (Loadings) ---
print("\nPrincipal Components (Loadings):")
print(pca.components_)
print(f"Shape of principal components: {pca.components_.shape}")