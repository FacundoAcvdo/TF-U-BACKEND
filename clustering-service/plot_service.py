import io
import matplotlib
import matplotlib.pyplot as plt

from plot_theme import generate_palette, apply_theme, ZINC_500

matplotlib.use('Agg')
apply_theme()

def plot_clusters(clusters_df, outliers_df):
    unique_clusters = sorted(clusters_df["cluster"].unique(), key=lambda x: int(x))
    n_clusters = len(unique_clusters)
    palette = generate_palette(n_clusters) if n_clusters > 0 else []
    color_map = {cid: palette[i] for i, cid in enumerate(unique_clusters)}
    point_colors = clusters_df["cluster"].map(color_map) if n_clusters > 0 else []

    plt.figure(figsize=(10, 8))
    if not outliers_df.empty:
        plt.scatter(outliers_df.x, outliers_df.y, alpha=0.05, s=2, c=ZINC_500)
    if not clusters_df.empty:
        plt.scatter(
            clusters_df.x, clusters_df.y, c=point_colors,
            alpha=0.6, s=2,
        )
    plt.axis('off')

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    plt.close()
    buffer.seek(0)
    return buffer.read()