from hdbscan import HDBSCAN
from umap import UMAP
import pandas as pd

def get_cluster_and_outlier_dataframes(embeddings):
    umap_model = UMAP(n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    reduced_embeddings = umap_model.fit_transform(embeddings)

    hdbscan_model = HDBSCAN(min_cluster_size=50, metric='euclidean', cluster_selection_method='eom').fit(reduced_embeddings)
    clusters = hdbscan_model.labels_

    reduced_embeddings = UMAP(n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings)

    df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
    df["cluster"] = [str(c) for c in clusters]

    clusters_df = df.loc[df.cluster != "-1", :]
    outliers_df = df.loc[df.cluster == "-1", :]

    return clusters_df, outliers_df