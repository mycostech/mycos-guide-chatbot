import code
import numpy as np
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import linalg as LA

from lib.data_source import load_vector_data

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

def perform_umap_plotting(data_matrix, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42, title="UMAP Projection"):
    """
    Performs UMAP dimensionality reduction and plots the result.

    Args:
        data_matrix (np.ndarray): The input data matrix of shape (sample_size, num_features).
        n_components (int): The dimension of the space to embed into. Default is 2.
        n_neighbors (int): The size of local neighborhood (in terms of number of neighboring points)
                           for UMAP. Default is 15.
        min_dist (float): The effective minimum distance between embedded points. Default is 0.1.
        random_state (int): A seed for the random number generator for reproducibility. Default is 42.
        title (str): The title for the plot.
    """

    print(f"Input data shape: {data_matrix.shape}")

    # Initialize UMAP
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )

    print("Performing UMAP dimensionality reduction...")
    # Fit and transform the data
    embedding = reducer.fit_transform(data_matrix)
    print(f"UMAP embedding shape: {embedding.shape}")

    # Plotting the UMAP projection
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], s=10, alpha=0.7)
    plt.title(title, fontsize=16)
    plt.xlabel(f'UMAP Component 1', fontsize=12)
    plt.ylabel(f'UMAP Component 2', fontsize=12)

    n_samples = embedding.shape[0]

    for i, (x_coord, y_coord) in enumerate(embedding):
        label = question if i + 1 == n_samples else y[i]
        color = 'red' if i + 1 == n_samples else 'dimgray'

        plt.annotate(
            label, # The label (name) for the point
            (x_coord, y_coord), # The (x,y) coordinates of the point
            textcoords="offset points", # How to position the text
            xytext=(5, 5), # Offset from the point (5 points right, 5 points up)
            ha='left', # Horizontal alignment of the text
            va='bottom', # Vertical alignment of the text
            fontsize=8, # Font size for the label
            color=color, # Color of the label text
            alpha=0.9
        )


    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

    return embedding

if __name__ == "__main__":
    X, y = load_vector_data()
    question = "I'm new graduate student looking for job"
    question_embedding = model.encode(question) 
    norm_question_embedding = (question_embedding / LA.norm(question_embedding)).tolist()

    # code.interact(local=locals())

    X.append(norm_question_embedding)

    samples = np.array(X)
    
    # 2. Call the function to perform UMAP and plot the results
    umap_embedding_2d = perform_umap_plotting(
        data_matrix=samples,
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
        title="UMAP Projection of Dummy Vector DB Data"
    )

    # You can also experiment with different parameters
    # For example, to see the effect of n_neighbors
    # umap_embedding_2d_nn5 = perform_umap_plotting(
    #     data_matrix=samples,
    #     n_components=2,
    #     n_neighbors=5,
    #     min_dist=0.1,
    #     random_state=42,
    #     title="UMAP Projection (n_neighbors=5)"
    # )

    # For example, to see the effect of min_dist
    # umap_embedding_2d_md05 = perform_umap_plotting(
    #     data_matrix=samples,
    #     n_components=2,
    #     n_neighbors=15,
    #     min_dist=0.5,
    #     random_state=42,
    #     title="UMAP Projection (min_dist=0.5)"
    # )

    # If you wanted to embed into 3 dimensions (for 3D plotting, though not covered here)
    # umap_embedding_3d = perform_umap_plotting(
    #     samples,
    #     n_components=3,
    #     n_neighbors=15,
    #     min_dist=0.1,
    #     random_state=42,
    #     title="UMAP 3D Projection"
    # )
    # print(f"UMAP 3D embedding shape: {umap_embedding_3d.shape}")