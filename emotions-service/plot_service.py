import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
from plot_theme import PRIMARY, generate_palette, apply_theme, style_axes, ZINC_300, ZINC_500

apply_theme()

# ---------------------------------------------------------------------------
# 1) emotion_corpus_mean - Bar horizontal corpus (mean + std)
# ---------------------------------------------------------------------------
def generate_emotion_corpus_mean_plot(corpus_stats: dict) -> list[tuple[bytes, str]]:
    sorted_emotions = sorted(
        corpus_stats.keys(),
        key=lambda e: corpus_stats[e].get("mean", 0.0),
        reverse=True
    )
    means = [corpus_stats[e].get("mean", 0.0) for e in sorted_emotions]
    stds = [corpus_stats[e].get("std", 0.0) for e in sorted_emotions]

    n = len(sorted_emotions)
    palette = generate_palette(n)

    FIGSIZE = (10, 6)
    DPI = 200
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    y_pos = np.arange(n)
    ax.barh(
        y_pos,
        means,
        xerr=stds,
        color=palette,
        edgecolor="white",
        linewidth=0.8,
        capsize=3,
        error_kw={"ecolor": ZINC_300, "elinewidth": 1},
        height=0.6,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_emotions, fontsize=10)
    ax.set_xlabel("score promedio (empath, normalize=True)", fontsize=10)
    ax.set_title("Perfil emocional promedio - Corpus (mean ± std)", fontsize=11, pad=12)
    ax.invert_yaxis()

    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.002, y_pos[i], f"{m:.3f} ±{s:.3f}", va="center", ha="left", fontsize=8, color="#52525b")

    max_val = max([m + s for m, s in zip(means, stds)] or [0.01])
    ax.set_xlim(0, max_val * 1.35 if max_val > 0 else 0.05)

    style_axes(ax)
    ax.grid(True, axis="x", alpha=0.35, linewidth=0.6)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return [(buf.getvalue(), "emotion_corpus_mean")]


# Alias backwards compatibility
def generate_emotions_corpus_plot(corpus_stats: dict):
    return generate_emotion_corpus_mean_plot(corpus_stats)


# ---------------------------------------------------------------------------
# 2) emotion_corpus_distribution - Violin/Boxplot grid por emocion
# ---------------------------------------------------------------------------
def generate_emotion_corpus_distribution_plot(df: pd.DataFrame, emotions: list[str]) -> list[tuple[bytes, str]]:
    """
    Violin + box por emocion sobre todo el corpus.
    Muestra min/median/mean/max/std visualmente.
    df debe contener columnas emociones con scores 0-1.
    """
    if df.empty or not emotions:
        FIGSIZE = (8, 4)
        DPI = 200
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        ax.text(0.5, 0.5, "Sin datos para distribucion", ha="center", va="center", fontsize=10)
        ax.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        return [(buf.getvalue(), "emotion_corpus_distribution")]

    # Filtrar emociones presentes
    emotions_present = [e for e in emotions if e in df.columns]
    if not emotions_present:
        emotions_present = emotions

    # Melt a formato largo
    df_melt = df[emotions_present].melt(var_name="emotion", value_name="score")

    # Ordenar emociones por mean descendente para coherencia con bar
    order = df[emotions_present].mean().sort_values(ascending=False).index.tolist()

    # Tamano dinamico
    n = len(emotions_present)
    FIGSIZE = (max(12, n * 1.1), 6)
    DPI = 200
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    palette = generate_palette(n)
    # Mapear emotion -> color para violin
    palette_dict = {emo: palette[i % len(palette)] for i, emo in enumerate(order)}

    # Violin con box interior (muestra median, quartiles, min/max) + quartiles
    sns.violinplot(
        data=df_melt,
        x="emotion",
        y="score",
        hue="emotion",
        order=order,
        palette=palette_dict,
        inner="box",
        linewidth=0.8,
        cut=0,
        ax=ax,
        legend=False,
    )

    # Overlay mean como punto diamante
    means = df[emotions_present].mean().reindex(order)
    x_pos = np.arange(len(order))
    ax.scatter(x_pos, means.values, color="white", edgecolor="#18181b", s=45, zorder=5, marker="D", linewidth=0.8, label="mean")
    # Anotar mean arriba del punto
    for i, m in enumerate(means.values):
        ax.text(i, m + 0.015, f"{m:.3f}", ha="center", va="bottom", fontsize=7, color="#52525b", weight="500")

    ax.set_title("Distribucion de emociones - Corpus (violin/box + mean)", fontsize=11, pad=12)
    ax.set_xlabel("emocion", fontsize=10)
    ax.set_ylabel("score empath", fontsize=10)
    ax.set_ylim(0, max(0.6, df_melt["score"].max() * 1.25 + 0.02) if not df_melt.empty else 1)

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    style_axes(ax)
    ax.grid(True, axis="y", alpha=0.35, linewidth=0.6)
    # Leyenda solo mean
    ax.legend(frameon=False, loc="upper right", fontsize=8)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return [(buf.getvalue(), "emotion_corpus_distribution")]


# ---------------------------------------------------------------------------
# 3) emotion_corpus_correlation - Heatmap 10x10 Pearson
# ---------------------------------------------------------------------------
def generate_emotion_corpus_correlation_plot(df: pd.DataFrame, emotions: list[str]) -> list[tuple[bytes, str]]:
    emotions_present = [e for e in emotions if e in df.columns]
    if not emotions_present or df.empty:
        FIGSIZE = (8, 6)
        DPI = 200
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        ax.text(0.5, 0.5, "Sin datos para correlacion", ha="center", va="center", fontsize=10)
        ax.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        return [(buf.getvalue(), "emotion_corpus_correlation")]

    corr = df[emotions_present].corr(method="pearson")
    # Constantes (std 0) dan NaN -> rellenar 0, diagonal 1
    corr = corr.fillna(0.0)
    # Ordenar por clustering simple: ordenar por mean para coherencia
    order = df[emotions_present].mean().sort_values(ascending=False).index.tolist()
    corr = corr.reindex(index=order, columns=order)

    FIGSIZE = (8, 7)
    DPI = 200
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Solo azules pero con rango dinamico para que se diferencie
    # vmin/vmax fijos (-1/1) comprimian todo en celeste porque r real es -0.2..0.5
    cmap = sns.color_palette("Blues", as_cmap=True)

    sns.heatmap(
        corr,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Pearson r"},
        annot_kws={"fontsize": 7},
        ax=ax,
        square=True,
    )

    ax.set_title("Co-ocurrencia de emociones - Corpus (correlacion Pearson 10x10)", fontsize=11, pad=14)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(ZINC_300)
    ax.spines["bottom"].set_color(ZINC_300)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return [(buf.getvalue(), "emotion_corpus_correlation")]


# ---------------------------------------------------------------------------
# Helpers para heatmaps por clase (mean/std/median)
# ---------------------------------------------------------------------------
def _generate_emotion_class_heatmap_by_metric(
    class_stats: list[dict],
    emotions: list[str],
    metric: str,
    function_name: str,
    cbar_label: str,
    title: str,
    fmt: str = ".3f",
) -> list[tuple[bytes, str]]:
    if not class_stats or not emotions:
        FIGSIZE = (8, 4)
        DPI = 200
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
        ax.text(0.5, 0.5, f"Sin datos para {function_name}", ha="center", va="center", fontsize=10)
        ax.axis("off")
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        return [(buf.getvalue(), function_name)]

    rows = []
    index = []
    for entry in class_stats:
        index.append(str(entry.get("clase", "")))
        rows.append([entry.get(emo, {}).get(metric, 0.0) for emo in emotions])

    df = pd.DataFrame(rows, index=index, columns=emotions)

    # Ordenar columnas por metrica global descendente para resaltar patrones
    col_order = df.mean(axis=0).sort_values(ascending=False).index.tolist()
    df = df[col_order]

    n_classes = len(df)
    n_emotions = len(col_order)

    FIGSIZE = (max(10, n_emotions * 1.0), max(4, n_classes * 1.2 + 1.5))
    DPI = 200
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    cmap = sns.color_palette("Blues", as_cmap=True)

    sns.heatmap(
        df,
        cmap=cmap,
        annot=True,
        fmt=fmt,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": cbar_label},
        annot_kws={"fontsize": 8},
        ax=ax,
    )

    ax.set_title(title, fontsize=11, pad=14)
    ax.set_xlabel("emocion", fontsize=10)
    ax.set_ylabel("clase", fontsize=10)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", fontsize=9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(ZINC_300)
    ax.spines["bottom"].set_color(ZINC_300)

    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return [(buf.getvalue(), function_name)]


def generate_emotion_class_mean_plot(class_stats: list[dict], emotions: list[str]) -> list[tuple[bytes, str]]:
    return _generate_emotion_class_heatmap_by_metric(
        class_stats, emotions, "mean", "emotion_class_mean", "mean score", "Emociones promedio por clase (mean)"
    )


def generate_emotion_class_std_plot(class_stats: list[dict], emotions: list[str]) -> list[tuple[bytes, str]]:
    return _generate_emotion_class_heatmap_by_metric(
        class_stats, emotions, "std", "emotion_class_std", "std", "Dispersion de emociones por clase (std)"
    )


def generate_emotion_class_median_plot(class_stats: list[dict], emotions: list[str]) -> list[tuple[bytes, str]]:
    return _generate_emotion_class_heatmap_by_metric(
        class_stats, emotions, "median", "emotion_class_median", "median score", "Emocion tipica por clase (median)"
    )


# Alias backwards compatibility para clase unica
def generate_emotions_class_heatmap(class_stats: list[dict], emotions: list[str] = None):
    if emotions is None:
        if class_stats:
            emotions = [k for k in class_stats[0].keys() if k not in ("clase", "count")]
        else:
            emotions = []
    return generate_emotion_class_mean_plot(class_stats, emotions)
