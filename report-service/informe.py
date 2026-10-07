import io
from datetime import datetime, timedelta, timezone
from xml.sax.saxutils import escape as _xml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

FONDO = "#ffffff"
TEXTO = "#1f2937"
SUAVE = "#6b7280"
AZUL = "#2563eb"
CELESTE = "#0d9488"
NARANJA = "#d97706"
ROJO = "#e11d48"
VIOLETA = "#9333ea"
PALETA = [AZUL, CELESTE, NARANJA, ROJO, VIOLETA, "#475569", "#a16207", "#15803d"]
REJILLA = "#e5e7eb"
BORDE = "#d1d5db"


def _figura(ancho=7.5, alto=3.6):
    fig, ax = plt.subplots(figsize=(ancho, alto))
    fig.patch.set_facecolor(FONDO)
    ax.set_facecolor(FONDO)
    ax.tick_params(colors=SUAVE, labelsize=8)
    for etiqueta in ax.get_xticklabels() + ax.get_yticklabels():
        etiqueta.set_color(SUAVE)
    for spine in ax.spines.values():
        spine.set_color(BORDE)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="y", color=REJILLA, linewidth=0.6, linestyle="--", alpha=1.0)
    ax.set_axisbelow(True)
    return fig, ax


def _a_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=fig.get_facecolor(), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _acortar(texto: str, n: int = 18) -> str:
    texto = str(texto)
    return texto if len(texto) <= n else texto[: n - 1] + "…"


def graficar_barras(pares: list[tuple[str, float]], titulo: str,
                    horizontal: bool = False) -> bytes | None:
    pares = [(str(e), float(v)) for e, v in pares if v is not None][:25]
    if not pares:
        return None
    etiquetas = [_acortar(e) for e, _ in pares]
    valores = [v for _, v in pares]
    if horizontal:
        pares_rev = list(zip(etiquetas, valores))[::-1]
        etiquetas = [e for e, _ in pares_rev]
        valores = [v for _, v in pares_rev]
        fig, ax = _figura(alto=max(2.5, 0.45 * len(etiquetas) + 1))
        barras = ax.barh(etiquetas, valores, color=AZUL, edgecolor="none", height=0.6)
        ax.set_xlabel("Cantidad", color=SUAVE, fontsize=9)
        for b, v in zip(barras, valores):
            ax.text(v, b.get_y() + b.get_height() / 2, f" {v:g}",
                    va="center", ha="left", color=TEXTO, fontsize=8)
    else:
        fig, ax = _figura()
        barras = ax.bar(etiquetas, valores, color=AZUL, edgecolor="none", width=0.6)
        ax.set_ylabel("Cantidad", color=SUAVE, fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        for b, v in zip(barras, valores):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:g}",
                    va="bottom", ha="center", color=TEXTO, fontsize=8)
    ax.set_title(titulo, color=TEXTO, fontsize=11, pad=10)
    fig.tight_layout()
    return _a_png(fig)


def graficar_dispersion(puntos: list[tuple[float, float, str]]) -> bytes | None:
    limpios = []
    for p in puntos[:2000]:
        try:
            limpios.append((float(p[0]), float(p[1]), str(p[2])))
        except (TypeError, ValueError, IndexError):
            continue
    if not limpios:
        return None
    etiquetas = sorted({c for _, _, c in limpios})
    color_por = {c: PALETA[i % len(PALETA)] for i, c in enumerate(etiquetas)}
    fig, ax = _figura(alto=4.2)
    for c in etiquetas:
        xs = [x for x, _, cc in limpios if cc == c]
        ys = [y for _, y, cc in limpios if cc == c]
        ax.scatter(xs, ys, s=10, alpha=0.65, color=color_por[c],
                   label=_acortar(c, 15), edgecolors="none")
    ax.set_title("Mapa de embeddings (2D)", color=TEXTO, fontsize=11, pad=10)
    ax.set_xlabel("Dimensión 1", color=SUAVE, fontsize=9)
    ax.set_ylabel("Dimensión 2", color=SUAVE, fontsize=9)
    if len(etiquetas) <= 10:
        leg = ax.legend(frameon=True, fontsize=8, loc="best")
        leg.get_frame().set_facecolor("#ffffff")
        leg.get_frame().set_edgecolor("#e5e7eb")
        for texto in leg.get_texts():
            texto.set_color(SUAVE)
    fig.tight_layout()
    return _a_png(fig)


def _celda(texto, negrita=False, tam=9, clara=False):
    estilo = ParagraphStyle(
        "Celda", fontName="Helvetica-Bold" if negrita else "Helvetica",
        fontSize=tam, leading=tam + 3,
        textColor=colors.white if clara else colors.HexColor("#1a1c22"),
        wordWrap="CJK",
    )
    return Paragraph(_xml(str(texto)), estilo)


def _tabla_metricas(filas: list[tuple[str, str]]):
    datos = [[_celda("Métrica", negrita=True), _celda("Valor", negrita=True)]]
    for k, v in filas:
        datos.append([_celda(k), _celda(v)])
    tabla = Table(datos, colWidths=[10 * cm, 7 * cm])
    tabla.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#2a2e3a")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f5f9")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return tabla


def _tabla_datos(encabezados: list[str], filas: list[list], cuerpo=None) -> Table:
    datos = [[_celda(h, negrita=True) for h in encabezados]]
    for fila in filas:
        datos.append([_celda(c) for c in fila])
    n = max(1, len(encabezados))
    ancho = 17.0 / n
    tabla = Table(datos, colWidths=[ancho * cm] * n, repeatRows=1)
    tabla.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#2a2e3a")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f5f9")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tabla


def _estilos():
    base = getSampleStyleSheet()
    titulo = ParagraphStyle("Titulo", parent=base["Heading1"], fontSize=24,
                            textColor=colors.HexColor("#111827"), spaceAfter=6,
                            leading=28)
    subtitulo = ParagraphStyle("Sub", parent=base["Normal"], fontSize=11,
                               textColor=colors.HexColor("#4b5563"), spaceAfter=6,
                               leading=15)
    h2 = ParagraphStyle("H2", parent=base["Heading2"], fontSize=15,
                        textColor=colors.HexColor("#111827"), spaceBefore=14,
                        spaceAfter=8, keepWithNext=True)
    cuerpo = ParagraphStyle("Cuerpo", parent=base["Normal"], fontSize=10,
                            textColor=colors.HexColor("#1f2937"), spaceAfter=6,
                            leading=14)
    pie = ParagraphStyle("Pie", parent=base["Normal"], fontSize=8,
                         textColor=colors.HexColor("#9aa0b4"), alignment=2)
    aviso = ParagraphStyle("Aviso", parent=base["Normal"], fontName="Helvetica-Bold",
                           fontSize=8, textColor=colors.HexColor("#111827"),
                           spaceBefore=8, spaceAfter=8, leading=11)
    return titulo, subtitulo, h2, cuerpo, pie, aviso


AVISO_LEGAL = (
    "<b>AVISO IMPORTANTE:</b> el presente documento es un reporte técnico generado "
    "automáticamente a partir de las actividades y análisis realizados por el usuario "
    "en la plataforma de análisis de datos, con fines exclusivamente descriptivos y "
    "académicos. No constituye diagnóstico, evaluación clínica, concepto profesional "
    "ni recomendación de intervención. Los resultados provienen de modelos automáticos "
    "que pueden contener errores y deben ser interpretados por personal calificado."
)


def _ahora_argentina() -> datetime:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/Argentina/Buenos_Aires"))
    except Exception:
        return datetime.now(timezone(timedelta(hours=-3)))


def _pie_pagina(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#9aa0b4"))
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Página {doc.page}")
    canvas.drawString(2 * cm, 1.2 * cm, "Informe de análisis de datos")
    canvas.restoreState()


def _imagen_bytes(png: bytes | None, ancho=15 * cm):
    if not png:
        return None
    try:
        img = Image(io.BytesIO(png))
        img._restrictSize(ancho, 10 * cm)
        img.hAlign = "CENTER"
        return img
    except Exception:
        return None


def _imagen(png: bytes | None, ancho=15 * cm):
    return _imagen_bytes(png, ancho)


SVG_MAX_BYTES = 2_000_000


def incrustar_svg(svg: str, ancho=15 * cm):
    try:
        from svglib.svglib import svg2rlg
        if not isinstance(svg, str):
            return None
        texto = svg.strip()
        if len(texto) > SVG_MAX_BYTES or "<svg" not in texto[:1000]:
            return None
        dibujo = svg2rlg(io.StringIO(texto))
        if dibujo is None:
            return None
        factor = 1.0
        if dibujo.width and dibujo.width > 0:
            factor = min(1.0, float(ancho) / float(dibujo.width))
        if factor < 1.0:
            dibujo.width = dibujo.width * factor
            dibujo.height = dibujo.height * factor
            dibujo.scale(factor, factor)
        dibujo.hAlign = "CENTER"
        return dibujo
    except Exception:
        return None


def componer_pdf(dataset: str, secciones: list[dict], omitidas: list[str] | None = None) -> bytes:
    titulo, subtitulo, h2, cuerpo, _pie, aviso = _estilos()
    fecha = _ahora_argentina().strftime("%d/%m/%Y %H:%M ART")
    nombre_ds = _xml(dataset)
    historia = [
        Spacer(1, 1 * cm),
        Paragraph("Informe de análisis de datos", titulo),
        Spacer(1, 0.3 * cm),
        Paragraph(f"Dataset: <b>{nombre_ds}</b>", subtitulo),
        Paragraph(f"Generado: {fecha} &nbsp;·&nbsp; Secciones: {len(secciones)}", subtitulo),
    ]
    historia.append(Paragraph(AVISO_LEGAL, aviso))
    if omitidas:
        historia.append(Paragraph(
            f"<i>Se omitieron secciones sin datos: {', '.join(omitidas)}.</i>", cuerpo))
    for n, sec in enumerate(secciones, start=1):
        if n > 1:
            historia.append(Spacer(1, 0.6 * cm))
        historia.append(Paragraph(f"{n}. {sec['titulo']}", h2))
        if sec.get("descripcion"):
            historia.append(Paragraph(sec["descripcion"], cuerpo))
        if sec.get("metricas"):
            historia.append(_tabla_metricas(sec["metricas"]))
            historia.append(Spacer(1, 0.25 * cm))
        png_ok = False
        for png in sec.get("pngs_bytes", []):
            img = _imagen_bytes(png)
            if img is not None:
                historia.append(img)
                historia.append(Spacer(1, 0.25 * cm))
                png_ok = True
        svg_ok = False
        if not png_ok:
            for svg in sec.get("svgs", []):
                dibujo = incrustar_svg(svg)
                if dibujo is not None:
                    historia.append(dibujo)
                    historia.append(Spacer(1, 0.25 * cm))
                    svg_ok = True
        if not svg_ok and not png_ok:
            for png in sec.get("graficos", []):
                img = _imagen(png)
                if img is not None:
                    historia.append(img)
                    historia.append(Spacer(1, 0.25 * cm))
        for titulo_tabla, encabezados, filas in sec.get("tablas_datos", []):
            historia.append(Paragraph(_xml(str(titulo_tabla)), cuerpo))
            historia.append(_tabla_datos(encabezados, filas))
            historia.append(Spacer(1, 0.25 * cm))
        if sec.get("nota"):
            historia.append(Paragraph(f"<i>{sec['nota']}</i>", cuerpo))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2 * cm,
                            rightMargin=2 * cm, topMargin=2 * cm,
                            bottomMargin=2 * cm, title=f"Informe - {dataset}")
    doc.build(historia, onFirstPage=_pie_pagina, onLaterPages=_pie_pagina)
    buf.seek(0)
    return buf.getvalue()
