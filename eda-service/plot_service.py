import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
from io import BytesIO
from plot_theme import PRIMARY, generate_palette, apply_theme, style_axes

#############

matplotlib.use('Agg')
apply_theme()

def generate_sentence_distribution_plot(dataset: pd.DataFrame):
    imagenes = []
    FIGSIZE = (6.0, 4.5)
    DPI = 200

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    sns.boxplot(x=dataset["sentences"], ax=ax, color=PRIMARY)
    ax.set_title("")
    ax.set_xlabel("sentences")
    style_axes(ax)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    imagenes.append((buf.getvalue(), "boxplot"))

    g = sns.catplot(data=dataset, x="sentences", color=PRIMARY, height=4.5, aspect=1.333)
    g.set_titles("")
    g.set_axis_labels("sentences", "")

    try:
        g.fig.set_size_inches(*FIGSIZE)
        g.fig.set_dpi(DPI)
    except Exception:
        pass
    for ax2 in g.axes.flat:
        ax2.set_title("")
        style_axes(ax2)
    g.fig.tight_layout()
    buf = BytesIO()
    g.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.15)
    plt.close("all")
    imagenes.append((buf.getvalue(), "catplot"))

    return imagenes

def generate_class_distribution_plot(dataset: pd.DataFrame):
    numeric_cols = [c for c in dataset.columns if c != "clase"]
    n_series = len(numeric_cols)
    palette = generate_palette(n_series)

    FIGSIZE = (10, 6)
    DPI = 150

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    dataset.plot.bar(x="clase", logy=True, ax=ax, color=palette, edgecolor="white", linewidth=0.6)
    ax.set_xlabel("")
    plt.setp(ax.get_xticklabels(), rotation=0)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", frameon=False)
    style_axes(ax)
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)

    return [(buf.getvalue(), "class_distribution")]

def generate_distribution_plot(dataset: pd.DataFrame):
    clases = np.unique(dataset['clase'])
    palette = generate_palette(len(clases))

    FIGSIZE = (20, 10)
    DPI = 150

    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=FIGSIZE, dpi=DPI)

    for i, clase in enumerate(clases):
        grupo = dataset[dataset['clase'] == clase]
        color = palette[i % len(palette)]
        sns.histplot(grupo['words'], label=clase, ax=axes[0], color=color, kde=True)
        sns.histplot(grupo['sentences'], label=clase, ax=axes[1], color=color, kde=True)

    axes[0].set_title("")
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', labelsize=12)
    axes[0].tick_params(axis='y', labelsize=12)
    axes[0].legend(frameon=False)
    style_axes(axes[0])

    axes[1].set_title("")
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', labelsize=12)
    axes[1].tick_params(axis='y', labelsize=12)
    axes[1].legend(frameon=False)
    axes[1].set_xscale('log')
    style_axes(axes[1])

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)

    return [(buf.getvalue(), "distribution")]