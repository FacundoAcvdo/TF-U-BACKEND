"""
plot_theme.py — Theming centralizado para matplotlib estilo shadcn.

- Paleta fija binaria: PRIMARY (#1447e6) y SECONDARY (#FDBA74)
- Paleta dinámica: generate_palette(n) con hues equiespaciados HSL (360/n)
  saturación y luminosidad fijas moderadas, fallback a husl de seaborn si está
  disponible.
- apply_theme(): configura rcParams para estética shadcn (spines, grid, colores zinc).

No hardcodear colores en scripts de graficación: importar desde este módulo.
"""

from __future__ import annotations

import colorsys
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Constantes shadcn ──────────────────────────────────────────────
PRIMARY: str = "#1447e6"      # color primario app (azul)
SECONDARY: str = "#FF7C39"    # naranja claro para comparación binaria
BINARY_PALETTE: List[str] = [PRIMARY, SECONDARY]

# Grises zinc (Tailwind)
ZINC_200: str = "#e4e4e7"
ZINC_300: str = "#d4d4d8"
ZINC_400: str = "#a1a1aa"
ZINC_500: str = "#71717a"
ZINC_600: str = "#52525b"

# Parámetros por defecto para generate_palette
_DEFAULT_SATURATION: float = 0.62   # 62%  dentro de 55-70%
_DEFAULT_LIGHTNESS: float = 0.56    # 56%  dentro de 50-60%


def _hsl_to_hex(h_deg: float, s: float, l: float) -> str:
    """Convierte HSL (h en grados 0-360, s/l en 0-1) a hex.

    Usa colorsys.hls_to_rgb (orden H, L, S). HLS de colorsys == HSL con L y S
    intercambiados en el nombre pero misma semántica.
    """
    h = (h_deg % 360.0) / 360.0
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(round(r * 255)):02x}{int(round(g * 255)):02x}{int(round(b * 255)):02x}"


def generate_palette(n: int, saturation: float = _DEFAULT_SATURATION,
                     lightness: float = _DEFAULT_LIGHTNESS) -> List[str]:
    """Devuelve n colores distintos para datos categóricos.

    - n == 0 -> []
    - n == 1 -> [PRIMARY]
    - n == 2 -> [PRIMARY, SECONDARY]  (paleta fija binaria, Requisito 1)
    - n >= 3 -> n colores con hues equiespaciados (360/n grados), s/l fijos.

    Si seaborn está instalado, intenta usar sns.color_palette("husl", n) que es
    perceptualmente uniforme y evita tonos agresivos; si falla usa conversión
    HSL manual. Funciona sin seaborn.

    No usa colormaps continuos (viridis etc.) porque no dan buen contraste
    entre categorías vecinas.
    """
    if n <= 0:
        return []
    if n == 1:
        return [PRIMARY]
    if n == 2:
        return [PRIMARY, SECONDARY]

    # Intento con seaborn husl (perceptualmente uniforme) si está disponible
    try:
        import seaborn as sns  # type: ignore
        # husl puede fallar si n es grande pero soporta cualquier n
        palette = sns.color_palette("husl", n)  # lista de tuplas rgb 0-1
        # sns puede devolver colores muy cercanos a PRIMARY/SECONDARY pero
        # distintos; está bien para n>=3.
        return [mcolors.to_hex(c) for c in palette]
    except Exception:
        pass

    # Fallback puro: HSL equiespaciado
    s = max(0.0, min(1.0, saturation))
    l = max(0.0, min(1.0, lightness))
    colors: List[str] = []
    for i in range(n):
        h = (i * 360.0) / n
        colors.append(_hsl_to_hex(h, s, l))
    return colors


def apply_theme() -> None:
    """Configura rcParams globales para estética shadcn.

    - Quita spines superior y derecho
    - Grid sutil zinc-200, alpha bajo
    - Texto/ejes en zinc-500/600, no negro puro
    - Fuente sans-serif (Inter si está, fallback Helvetica/Arial)
    - Fondo blanco/transparente sin bordes pesados
    Llamar una vez al inicio del proceso (después de matplotlib.use('Agg')).
    """
    mpl.rcParams.update({
        # Figura / ejes
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": ZINC_300,
        "axes.labelcolor": ZINC_600,
        "axes.titlecolor": ZINC_600,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.titleweight": "600",
        # Spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        # Grilla
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": ZINC_200,
        "grid.alpha": 0.7,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        # Ticks
        "xtick.color": ZINC_500,
        "ytick.color": ZINC_500,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Texto
        "text.color": ZINC_600,
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
        # Leyenda
        "legend.frameon": False,
        "legend.edgecolor": ZINC_200,
        "legend.facecolor": "white",
        "legend.fontsize": 9,
        # Guardado
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.transparent": False,
        # Líneas
        "lines.linewidth": 1.8,
        "patch.linewidth": 0.8,
        "patch.edgecolor": ZINC_300,
    })


def style_axes(ax, despine: bool = True, grid: bool = True) -> None:
    """Aplica estilo shadcn a un Axes existente (complemento a rcParams).

    Útil porque algunos rcParams de spines no se aplican retroactivamente.
    """
    if despine:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(ZINC_300)
        ax.spines["bottom"].set_color(ZINC_300)
    if grid:
        ax.grid(True, axis="y", color=ZINC_200, alpha=0.7, linewidth=0.8, linestyle="-")
        ax.grid(True, axis="x", color=ZINC_200, alpha=0.35, linewidth=0.6, linestyle="-")
        ax.set_axisbelow(True)


__all__ = [
    "PRIMARY",
    "SECONDARY",
    "BINARY_PALETTE",
    "ZINC_200",
    "ZINC_300",
    "ZINC_400",
    "ZINC_500",
    "ZINC_600",
    "generate_palette",
    "apply_theme",
    "style_axes",
]
