import pandas as pd

try:
    from cuml.manifold import UMAP as cumlUMAP
    from cuml.cluster import HDBSCAN as cumlHDBSCAN
    import cupy as cp
    HAS_CUML = True
except Exception as e:
    HAS_CUML = False


def get_cluster_and_outlier_dataframes(embeddings):
    n = embeddings.shape[0] if hasattr(embeddings, 'shape') else len(embeddings)
    min_cluster_size = max(10, int(n * 0.005))

    use_gpu = False
    if HAS_CUML:
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
            use_gpu = True
        except Exception as e:
            use_gpu = False

    if use_gpu:

        try:

            umap_5 = cumlUMAP(n_components=5, min_dist=0.0, metric='cosine')
            reduced_5d = umap_5.fit_transform(embeddings)

            hdb = cumlHDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')

            try:
                clusters = hdb.fit_predict(reduced_5d)
        
                if hasattr(clusters, 'get'):
                    clusters = clusters.get()
                clusters = clusters.astype(int) if hasattr(clusters, 'astype') else clusters
            except Exception:
                hdb.fit(reduced_5d)
                clusters = hdb.labels_
                if hasattr(clusters, 'get'):
                    clusters = clusters.get()

            umap_2 = cumlUMAP(n_components=2, min_dist=0.0, metric='cosine')
            reduced_2d = umap_2.fit_transform(embeddings)
            if hasattr(reduced_2d, 'get'):
                reduced_2d = reduced_2d.get()

            df = pd.DataFrame(reduced_2d, columns=["x", "y"])
            df["cluster"] = [str(int(c)) for c in clusters]

            clusters_df = df.loc[df.cluster != "-1", :]
            outliers_df = df.loc[df.cluster == "-1", :]

            return clusters_df, outliers_df
        except Exception as e:
            use_gpu = False

    from hdbscan import HDBSCAN
    from umap import UMAP

    umap_5 = UMAP(n_components=5, min_dist=0.0, metric='cosine', n_jobs=-1, low_memory=False, random_state=None)
    reduced_5d = umap_5.fit_transform(embeddings)

    hdb = HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean', cluster_selection_method='eom', core_dist_n_jobs=-1)
    hdb.fit(reduced_5d)
    clusters = hdb.labels_

    umap_2 = UMAP(n_components=2, min_dist=0.0, metric='cosine', n_jobs=-1, low_memory=False, random_state=None)
    reduced_2d = umap_2.fit_transform(embeddings)

    df = pd.DataFrame(reduced_2d, columns=["x", "y"])
    df["cluster"] = [str(c) for c in clusters]

    clusters_df = df.loc[df.cluster != "-1", :]
    outliers_df = df.loc[df.cluster == "-1", :]

    return clusters_df, outliers_df
