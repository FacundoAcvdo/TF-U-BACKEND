import os
import tempfile
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from informe import (
    componer_pdf,
    graficar_barras,
    graficar_dispersion,
)
from inventario import FUENTE_POR_SECCION, SECCIONES, TITULOS, get_supabase_client, inventario, leer_fuente, resolver_dataset

app = FastAPI()

origins = ["http://localhost:8080", "http://localhost:3000", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerarRequest(BaseModel):
    dataset: str
    secciones: list[str]
    graficos: dict[str, list[str]] | None = None
    pngs: dict[str, list[str]] | None = None


import base64
import binascii


PNG_MAX_BYTES = 8_000_000


def decodificar_png(dato: str) -> bytes | None:
    try:
        if not isinstance(dato, str):
            return None
        texto = dato.strip()
        if len(texto) > PNG_MAX_BYTES:
            return None
        if "," in texto and texto.split(",", 1)[0].endswith("base64"):
            texto = texto.split(",", 1)[1]
        crudo = base64.b64decode(texto, validate=True)
        if not crudo.startswith(b"\x89PNG"):
            return None
        return crudo
    except (binascii.Error, ValueError):
        return None


def spec_barras(titulo: str, pares: list[tuple[str, float]], horizontal: bool = False) -> dict | None:
    pares = [(str(e)[:30], float(v)) for e, v in pares if v is not None][:20]
    if not pares:
        return None
    return {"tipo": "barrasH" if horizontal else "barras", "titulo": titulo,
            "etiquetas": [e for e, _ in pares], "valores": [v for _, v in pares]}


def spec_lineas(titulo: str, series: list[tuple[str, list[tuple[str, float]]]]) -> dict | None:
    series = [(str(n)[:20], [(str(x)[:20], float(v)) for x, v in pts]) for n, pts in series]
    series = [(n, pts) for n, pts in series if pts][:8]
    if not series:
        return None
    return {"tipo": "lineas", "titulo": titulo,
            "series": [{"nombre": n, "puntos": [{"x": x, "y": v} for x, v in pts]} for n, pts in series]}


def spec_dispersion(titulo: str, puntos: list[tuple[float, float, str]], tope: int = 1500,
                    clusters: list | None = None, outliers=None, total=None) -> dict | None:
    limpios = []
    for p in puntos[:tope]:
        try:
            limpios.append({"x": float(p[0]), "y": float(p[1]), "c": str(p[2]) if len(p) > 2 else "0"})
        except (TypeError, ValueError, IndexError):
            continue
    if not limpios:
        return None
    spec = {"tipo": "dispersion", "titulo": titulo, "puntos": limpios}
    if clusters is not None:
        spec["clusters"] = clusters
    if outliers is not None:
        spec["outliers"] = outliers
    if total is not None:
        spec["total"] = total
    return spec


def specs_para_fuente(sid: str, fuente: dict) -> list[dict]:
    specs: list[dict] = []
    try:
        if sid == "estadisticas":
            s = spec_barras("Distribución por clase", _pares(fuente.get("distribution", []))[:15])
            if s:
                specs.append(s)
        elif sid == "nube":
            palabras = []
            for f in (fuente.get("frequencies", []) or [])[:130]:
                if isinstance(f, dict) and f.get("text"):
                    try:
                        palabras.append({"text": str(f["text"]),
                                         "count": int(f.get("count", f.get("value", 0)))})
                    except (TypeError, ValueError):
                        continue
            if palabras:
                maximo = max((p["count"] for p in palabras), default=1) or 1
                specs.append({"tipo": "nube", "titulo": "Nube de palabras",
                              "palabras": palabras, "max_count": maximo})
        elif sid == "emociones":
            s = spec_barras("Distribución de emociones", _pares(fuente.get("distribution", [])))
            if s:
                specs.append(s)
        elif sid == "evolucion":
            for fila in fuente.get("filas", []) or []:
                payload = fila.get("payload") or {}
                periodos = payload.get("periodos") or payload.get("bloques") or []
                acumulado: dict[str, float] = {}
                for per in periodos:
                    for w in (per.get("palabras_top") or [])[:10]:
                        if isinstance(w, dict) and w.get("palabra") is not None:
                            try:
                                acumulado[str(w["palabra"])] = acumulado.get(str(w["palabra"]), 0) + float(w.get("cantidad", 0))
                            except (TypeError, ValueError):
                                continue
                top = sorted(acumulado.items(), key=lambda kv: kv[1], reverse=True)[:5]
                series = []
                for palabra, _ in top:
                    puntos = []
                    for per in periodos:
                        valor = 0.0
                        for w in per.get("palabras_top") or []:
                            if isinstance(w, dict) and str(w.get("palabra")) == palabra:
                                try:
                                    valor = float(w.get("cantidad", 0))
                                except (TypeError, ValueError):
                                    valor = 0.0
                                break
                        puntos.append((str(per.get("etiqueta", "?")), valor))
                    series.append((palabra, puntos))
                s = spec_lineas(f"Evolución — {fila.get('remitente', '?')}", series)
                if s:
                    specs.append(s)
        elif sid == "embeddings":
            puntos = []
            clusters = []
            vistos: dict[str, int] = {}
            for p in fuente.get("points", []) or []:
                try:
                    if isinstance(p, (list, tuple)) and len(p) >= 2:
                        c = str(p[2]) if len(p) > 2 else "0"
                        puntos.append((float(p[0]), float(p[1]), c))
                    elif isinstance(p, dict):
                        c = str(p.get("cluster", p.get("label", "0")))
                        puntos.append((float(p.get("x")), float(p.get("y")), c))
                    else:
                        continue
                    vistos[c] = vistos.get(c, 0) + 1
                except (TypeError, ValueError, IndexError):
                    continue
            for c in sorted(vistos):
                clusters.append({"id": c, "count": vistos[c]})
            s = spec_dispersion("Mapa de embeddings (2D)", puntos,
                                clusters=clusters,
                                outliers=fuente.get("outliers", 0),
                                total=fuente.get("total", len(puntos)))
            if s:
                specs.append(s)
    except Exception:
        pass
    return specs


class DatosGraficosRequest(BaseModel):
    dataset: str
    secciones: list[str]


@app.post("/datos-graficos")
async def datos_graficos(
    data: DatosGraficosRequest,
    x_user_id: str = Header(..., alias="X-User-ID"),
    x_user_token: str = Header(..., alias="X-User-Token"),
):
    supabase = get_supabase_client(x_user_token)
    ds = resolver_dataset(supabase, data.dataset, x_user_id)
    dataset_id = ds.get("dataset_id")
    validos = set(FUENTE_POR_SECCION.keys())
    salida: dict[str, list[dict]] = {}
    for sid in (data.secciones or []):
        if sid not in validos:
            continue
        fuente = leer_fuente(supabase, FUENTE_POR_SECCION[sid], data.dataset, x_user_id, dataset_id)
        if fuente is None:
            continue
        specs = specs_para_fuente(sid, fuente)
        if specs:
            salida[sid] = specs
    return {"dataset": data.dataset, "graficos": salida}


def _pares(lista, nombres=("nombre", "clase", "label", "text", "texto", "palabra", "emocion", "categoria"),
           valores=("cantidad", "valor", "value", "count", "total", "frecuencia")) -> list[tuple[str, float]]:
    pares: list[tuple[str, float]] = []
    if not isinstance(lista, list):
        return pares
    for item in lista:
        if not isinstance(item, dict):
            continue
        etiqueta = next((str(item[k]) for k in nombres if item.get(k) not in (None, "")), None)
        valor = next((item[k] for k in valores if isinstance(item.get(k), (int, float))), None)
        if etiqueta is not None and valor is not None:
            pares.append((etiqueta, float(valor)))
    return pares


ETIQUETAS_ESTADISTICAS = [
    ("texto", "Mensajes"),
    ("words", "Palabras totales"),
    ("word_avg", "Promedio palabras/mensaje"),
    ("sentences", "Oraciones totales"),
    ("vocabulary", "Vocabulario"),
    ("sentence_avg", "Promedio por oración"),
    ("median", "Longitud mediana"),
    ("mean", "Longitud media"),
    ("std", "Desvío longitud"),
    ("count", "Textos analizados"),
]


def _redondear(v):
    try:
        f = float(v)
        return f"{f:.2f}" if abs(f) >= 0.01 or f == 0 else str(v)
    except (TypeError, ValueError):
        return str(v)


def _metricas_curadas(d: dict) -> list[tuple[str, str]]:
    filas: list[tuple[str, str]] = []
    if not isinstance(d, dict):
        return filas
    for clave, etiqueta in ETIQUETAS_ESTADISTICAS:
        if d.get(clave) is not None:
            filas.append((etiqueta, _redondear(d[clave])))
    return filas


def construir_estadisticas(d: dict) -> dict:
    graficos = []
    png = graficar_barras(_pares(d.get("distribution", []))[:15], "Distribución por clase")
    if png:
        graficos.append(png)
    png = graficar_barras(_pares(d.get("sentence_freq", []))[:12], "Frecuencia por longitud de oración")
    if png:
        graficos.append(png)
    metricas = _metricas_curadas(d.get("corpus_stats", {})) + _metricas_curadas(d.get("sentence_stats", {}))
    return {
        "titulo": "Estadísticas generales",
        "descripcion": "Resumen descriptivo del corpus.",
        "metricas": metricas[:10],
        "graficos": graficos,
    }


def construir_nube(d: dict) -> dict:
    pares = []
    for f in (d.get("frequencies", []) or [])[:15]:
        if isinstance(f, dict):
            try:
                pares.append((str(f.get("text", "?")), float(f.get("value", f.get("count", 0)))))
            except (TypeError, ValueError):
                continue
    graficos = []
    png = graficar_barras(pares, "Top 15 palabras", horizontal=True)
    if png:
        graficos.append(png)
    return {
        "titulo": "Nube de palabras",
        "descripcion": f"{d.get('unique_words', '?')} palabras únicas sobre {d.get('total_words', '?')} totales.",
        "metricas": [
            ("Palabras totales", str(d.get("total_words", "?"))),
            ("Palabras únicas", str(d.get("unique_words", "?"))),
            ("Documentos", str(d.get("documents_processed", "?"))),
        ],
        "graficos": graficos,
    }


def construir_embeddings(d: dict) -> dict:
    puntos = []
    for p in d.get("points", []) or []:
        try:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                puntos.append((float(p[0]), float(p[1]), str(p[2]) if len(p) > 2 else "0"))
            elif isinstance(p, dict):
                puntos.append((float(p.get("x")), float(p.get("y")),
                               str(p.get("cluster", p.get("label", "0")))))
        except (TypeError, ValueError, IndexError):
            continue
    graficos = []
    png = graficar_dispersion([(x, y, c) for x, y, c in puntos])
    if png:
        graficos.append(png)
    return {
        "titulo": "Visualización de embeddings",
        "descripcion": "Proyección 2D de los documentos por similitud.",
        "metricas": [
            ("Documentos totales", str(d.get("total", "?"))),
            ("Puntos graficados", str(d.get("displayed", len(puntos)))),
            ("Clústeres", str(len({c for _, _, c in puntos}))),
            ("Atípicos", str(d.get("outliers", "?"))),
        ],
        "graficos": graficos,
    }


def construir_emociones(d: dict) -> dict:
    graficos = []
    corpus = d.get("corpus_stats", {}) or {}
    pares = []
    if isinstance(corpus, dict):
        for emo, stats in corpus.items():
            if isinstance(stats, dict) and stats.get("mean") is not None:
                try:
                    pares.append((str(emo), float(stats["mean"])))
                except (TypeError, ValueError):
                    continue
    pares.sort(key=lambda kv: kv[1], reverse=True)
    png = graficar_barras(pares[:12], "Intensidad media por emoción")
    if png:
        graficos.append(png)
    top = d.get("top_words", {}) or {}
    tablas_datos = []
    if isinstance(top, dict):
        filas_tw = []
        for emo, palabras in top.items():
            if isinstance(palabras, list) and palabras:
                filas_tw.append([str(emo), ", ".join(str(w) for w in palabras[:3])])
        if filas_tw:
            tablas_datos.append(("Top 3 palabras por emoción", ["Emoción", "Palabras"], filas_tw))
    metricas = [(f"Intensidad media · {e}", f"{v:.3f}") for e, v in pares[:6]]
    return {
        "titulo": "Emociones negativas",
        "descripcion": "Distribución de emociones detectadas en el corpus.",
        "metricas": metricas,
        "graficos": graficos,
        "tablas_datos": tablas_datos,
    }


def construir_topicos(d: dict) -> dict:
    topicos = d.get("topics", []) or []
    pares = []
    filas_tabla = []
    for t in topicos:
        if not isinstance(t, dict):
            continue
        nombre = str(t.get("title") or t.get("titulo") or f"Tópico {t.get('topic_id', '?')}")
        try:
            cant = int(t.get("document_count", 0))
        except (TypeError, ValueError):
            cant = 0
        pares.append((nombre, float(cant)))
        kws = t.get("keywords", []) or []
        filas_tabla.append([nombre, str(cant),
                            ", ".join(str(k) for k in kws[:10]) if kws else "—"])
    graficos = []
    png = graficar_barras(sorted(pares, key=lambda kv: kv[1], reverse=True)[:15],
                          "Documentos por tópico", horizontal=True)
    if png:
        graficos.append(png)
    return {
        "titulo": "Tópicos principales",
        "descripcion": f"{len(topicos)} tópicos sobre {d.get('document_count', '?')} documentos.",
        "metricas": [
            ("Tópicos", str(d.get("topic_count", len(topicos)))),
            ("Documentos", str(d.get("document_count", "?"))),
            ("Atípicos", str(d.get("outlier_count", "?"))),
        ],
        "graficos": graficos,
        "tablas_datos": (
            [("Tópicos con palabras clave", ["Tópico", "Docs", "Palabras clave"], filas_tabla)]
            if filas_tabla else []
        ),
    }


def construir_chats(d: dict) -> dict:
    chats = d.get("chats", []) or []
    total_mensajes = int(d.get("total_mensajes", 0) or 0)
    n_usu = n_asi = 0
    tablas_datos = []
    for c in chats:
        mensajes = c.get("mensajes", []) or []
        filas = []
        for i, m in enumerate(mensajes, start=1):
            autor = str(m.get("autor", "?"))
            texto = str(m.get("mensaje", ""))
            if len(texto) > 800:
                texto = texto[:800] + "…"
            filas.append([str(i), autor, texto])
            a = autor.lower()
            if a in ("usuario", "user", "human"):
                n_usu += 1
            else:
                n_asi += 1
        if filas:
            titulo = f"Chat {c.get('chat_id', '?')} — {c.get('fecha_creacion', '')}"
            tablas_datos.append((titulo, ["#", "Autor", "Mensaje"], filas))
    return {
        "titulo": "Chats del chatbot",
        "descripcion": f"{len(chats)} chats con {total_mensajes} mensajes para este dataset.",
        "metricas": [
            ("Chats", str(len(chats))),
            ("Mensajes", str(total_mensajes)),
            ("De usuario", str(n_usu)),
            ("Del asistente", str(n_asi)),
        ],
        "graficos": [],
        "tablas_datos": tablas_datos,
    }


def construir_riesgo(d: dict) -> dict:
    resumen = d.get("resumen", {}) or {}
    counts = resumen.get("counts", {}) or {}
    orden = ["alto", "potencial", "medio", "bajo", "nulo"]
    pares = []
    for nivel in orden:
        try:
            pares.append((nivel, float(counts.get(nivel, 0) or 0)))
        except (TypeError, ValueError):
            continue
    graficos = []
    png = graficar_barras(pares, "Casos por nivel de riesgo")
    if png:
        graficos.append(png)
    metricas = [
        ("Casos analizados", str(resumen.get("total", "?"))),
        ("% alto + potencial", str(resumen.get("pct_alto_potencial", "?"))),
        ("Score medio", str(resumen.get("score_medio", "?"))),
        ("Modelo", str(d.get("modelo_version", "?"))),
    ]
    for nivel in orden:
        metricas.append((f"Nivel {nivel}", str(counts.get(nivel, 0))))
    tablas_datos = []
    por_clase = d.get("por_clase", {}) or {}
    if isinstance(por_clase, dict) and por_clase:
        filas_pc = []
        for clase, agg in por_clase.items():
            if not isinstance(agg, dict):
                continue
            filas_pc.append([str(clase), str(agg.get("total", "?")),
                             str(agg.get("alto", "?")), str(agg.get("potencial", "?"))])
        if filas_pc:
            tablas_datos.append(("Riesgo por clase", ["Clase", "Total", "Alto", "Potencial"], filas_pc))
    top_items = d.get("top_items", []) or []
    if isinstance(top_items, list) and top_items:
        filas_top = []
        for it in top_items[:30]:
            if not isinstance(it, dict):
                continue
            texto = str(it.get("texto", ""))
            if len(texto) > 300:
                texto = texto[:300] + "…"
            trigs = it.get("triggers", []) or []
            filas_top.append([texto, str(it.get("nivel", "?")),
                              str(it.get("score_a", "?")),
                              ", ".join(str(t) for t in trigs[:5]) if trigs else "—"])
        if filas_top:
            tablas_datos.append(("Casos de mayor riesgo", ["Texto", "Nivel", "Score", "Triggers"], filas_top))
    out = {
        "titulo": "Nivel de riesgo",
        "descripcion": "Riesgo suicida detectado por el modelo (solo artefactos guardados).",
        "metricas": metricas,
        "graficos": graficos,
        "tablas_datos": tablas_datos,
    }
    return out


def construir_evolucion(d: dict) -> dict:
    metricas: list[tuple[str, str]] = []
    tablas_datos = []
    for fila in d.get("filas", []) or []:
        payload = fila.get("payload") or {}
        periodos = payload.get("periodos") or payload.get("bloques") or []
        nombre = f"{fila.get('remitente', '?')} ({fila.get('modo', '?')})"
        metricas.append((f"Análisis {nombre}",
                         f"{len(periodos)} bloques/periodos · {fila.get('cantidad_mensajes', '?')} mensajes"))
        filas_tabla = []
        for per in periodos:
            top = [str(w.get("palabra")) for w in (per.get("palabras_top") or [])[:10]
                   if isinstance(w, dict) and w.get("palabra") is not None]
            filas_tabla.append([str(per.get("etiqueta", "?")), ", ".join(top) if top else "—"])
        if filas_tabla:
            tablas_datos.append((f"Top 10 por período — {nombre}", ["Periodo", "Palabras"], filas_tabla))
    return {
        "titulo": "Evolución de palabras",
        "descripcion": "Palabras más usadas por bloque/período.",
        "metricas": metricas,
        "graficos": [],
        "tablas_datos": tablas_datos,
    }


def construir_resumen(dataset: str, fuentes: dict) -> dict:
    metricas: list[tuple[str, str]] = []
    eda = fuentes.get("estadisticas")
    if isinstance(eda, dict):
        corpus = eda.get("corpus_stats", {}) or {}
        for k in ("total_mensajes", "total_documentos", "mensajes", "documentos",
                  "num_clases", "clases", "total_palabras"):
            if corpus.get(k) is not None:
                metricas.append(("Corpus · " + str(k), str(corpus[k])))
                break
    nube = fuentes.get("nube")
    if isinstance(nube, dict):
        metricas.append(("Vocabulario", f"{nube.get('unique_words', '?')} palabras únicas"))
    emo = fuentes.get("emociones")
    if isinstance(emo, dict) and isinstance(emo.get("emotions"), list):
        metricas.append(("Emociones detectadas", str(len(emo["emotions"]))))
    evo = fuentes.get("evolucion")
    if isinstance(evo, dict):
        metricas.append(("Evoluciones guardadas", str(len(evo.get("filas", []) or []))))
    if not metricas:
        metricas.append(("Dataset", dataset))
    return {
        "titulo": "Resumen ejecutivo",
        "descripcion": "Hallazgos principales del análisis realizado sobre el dataset.",
        "metricas": metricas,
        "graficos": [],
    }


CONSTRUCTORES = {
    "estadisticas": construir_estadisticas,
    "nube": construir_nube,
    "embeddings": construir_embeddings,
    "emociones": construir_emociones,
    "evolucion": construir_evolucion,
    "topicos": construir_topicos,
    "chats": construir_chats,
    "riesgo": construir_riesgo,
}


@app.get("/disponibles")
async def disponibles(
    dataset: str,
    x_user_id: str = Header(..., alias="X-User-ID"),
    x_user_token: str = Header(..., alias="X-User-Token"),
):
    supabase = get_supabase_client(x_user_token)
    return inventario(supabase, dataset, x_user_id)


@app.post("/generar")
async def generar(
    data: GenerarRequest,
    x_user_id: str = Header(..., alias="X-User-ID"),
    x_user_token: str = Header(..., alias="X-User-Token"),
):
    supabase = get_supabase_client(x_user_token)
    ds = resolver_dataset(supabase, data.dataset, x_user_id)
    dataset_id = ds.get("dataset_id")

    validos = set(SECCIONES)
    pedidos = [s for s in (data.secciones or []) if s in validos]
    if not pedidos:
        raise HTTPException(status_code=422, detail="No se pidieron secciones válidas.")

    partes: list[dict] = []
    omitidas: list[str] = []
    fuentes: dict[str, object] = {}
    svgs_pedidos: dict[str, list[str]] = data.graficos if isinstance(data.graficos, dict) else {}
    pngs_pedidos: dict[str, list[str]] = data.pngs if isinstance(data.pngs, dict) else {}
    for sid in pedidos:
        fuente_id = FUENTE_POR_SECCION.get(sid)
        if fuente_id is None:
            omitidas.append(TITULOS.get(sid, sid))
            continue
        fuente = leer_fuente(supabase, fuente_id, data.dataset, x_user_id, dataset_id)
        if fuente is None:
            omitidas.append(TITULOS.get(sid, sid))
            continue
        fuentes[sid] = fuente
        try:
            sec = CONSTRUCTORES[fuente_id](fuente)
        except Exception:
            omitidas.append(TITULOS.get(sid, sid))
            continue
        crudos = svgs_pedidos.get(sid) or []
        sec["svgs"] = [s for s in crudos if isinstance(s, str)][:8] if sid != "evolucion" else []
        crudos_png = pngs_pedidos.get(sid) or []
        sec["pngs_bytes"] = []
        for raw in crudos_png[:4]:
            dec = decodificar_png(raw)
            if dec is not None:
                sec["pngs_bytes"].append(dec)
        partes.append(sec)

    if not partes:
        raise HTTPException(
            status_code=422,
            detail="Este dataset no tiene análisis realizados para las secciones pedidas.",
        )

    partes.insert(0, construir_resumen(data.dataset, fuentes))
    pdf = componer_pdf(data.dataset, partes, omitidas)

    fecha = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    nombre = f"informe_{data.dataset}_{fecha}.pdf".replace(" ", "_")
    ruta = os.path.join(tempfile.gettempdir(), nombre)
    with open(ruta, "wb") as f:
        f.write(pdf)
    return FileResponse(path=ruta, media_type="application/pdf", filename=nombre)
