import io
import os
import re
import zipfile

README_DATASET_RE = re.compile(r"-\s*Dataset:\s*\*\*(.+?)\*\*")
README_METRICA_RE = re.compile(r"-\s*(Accuracy|F1-score weighted|Precision weighted|Recall weighted):\s*\*\*([0-9.]+)\*\*")

RESULTADOS_PATH = os.environ.get("RESULTADOS_PATH", "/app/data/resultados")
MODELOS_DIRNAME = "modelos"

SECCIONES = [
    "estadisticas",
    "nube",
    "topicos",
    "riesgo",
    "sentimiento",
    "evolucion",
    "embeddings",
    "chats",
    "emociones",
    "clasificador",
]

TITULOS = {
    "estadisticas": "Estadísticas generales",
    "nube": "Nube de palabras",
    "topicos": "Tópicos principales",
    "riesgo": "Nivel de riesgo",
    "sentimiento": "Sentimiento en el tiempo",
    "evolucion": "Evolución de palabras",
    "embeddings": "Visualización de embeddings",
    "chats": "Chats del chatbot",
    "emociones": "Emociones negativas",
    "clasificador": "Clasificador entrenado",
}

FUENTE_POR_SECCION = {
    "estadisticas": "estadisticas",
    "nube": "nube",
    "embeddings": "embeddings",
    "emociones": "emociones",
    "evolucion": "evolucion",
    "clasificador": "clasificador",
}


def get_supabase_client(user_jwt: str):
    from supabase import create_client
    client = create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_ANON_KEY"),
    )
    client.postgrest.auth(user_jwt)
    return client


def resolver_dataset(supabase, dataset_name: str, user_id: str):
    from fastapi import HTTPException
    response = (
        supabase.table("datasets")
        .select("dataset_id,dataset_name")
        .eq("dataset_name", dataset_name)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not response.data:
        raise HTTPException(
            status_code=404,
            detail=f"Dataset '{dataset_name}' no encontrado o sin permisos.",
        )
    return response.data[0]


def leer_fila(supabase, tabla: str, user_id: str, dataset_id: str, columnas: str):
    try:
        respuesta = (
            supabase.table(tabla)
            .select(columnas)
            .eq("dataset_id", dataset_id)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if respuesta.data:
            return respuesta.data[0]
    except Exception:
        pass
    return None


def modelos_del_dataset(dataset_name: str) -> list[dict]:
    encontrados: list[dict] = []
    base = os.path.join(RESULTADOS_PATH, MODELOS_DIRNAME)
    try:
        archivos = os.listdir(base)
    except OSError:
        return encontrados
    for nombre in sorted(archivos):
        if not nombre.endswith(".zip"):
            continue
        ruta = os.path.join(base, nombre)
        try:
            with zipfile.ZipFile(ruta) as zf:
                if "README.md" not in zf.namelist():
                    continue
                texto = zf.read("README.md").decode("utf-8", errors="ignore")
        except Exception:
            continue
        m = README_DATASET_RE.search(texto)
        if not m or m.group(1).strip() != dataset_name:
            continue
        metricas = {
            k: float(v) for k, v in README_METRICA_RE.findall(texto)
        }
        sidecar = os.path.join(base, nombre[:-4] + ".name")
        nombre_archivo = nombre
        try:
            with open(sidecar, encoding="utf-8") as f:
                leido = f.read().strip()
                if leido:
                    nombre_archivo = leido
        except OSError:
            pass
        encontrados.append({
            "model_id": nombre[:-4],
            "nombre_archivo": nombre_archivo,
            "metricas": metricas,
        })
    return encontrados


def filas_evolucion(supabase, user_id: str, dataset_id: str) -> list[dict]:
    try:
        respuesta = (
            supabase.table("evolution_results")
            .select("remitente,modo,top_k,cantidad_mensajes,updated_at")
            .eq("dataset_id", dataset_id)
            .eq("user_id", user_id)
            .execute()
        )
        return respuesta.data or []
    except Exception:
        return []


def inventario(supabase, dataset_name: str, user_id: str) -> dict:
    ds = resolver_dataset(supabase, dataset_name, user_id)
    dataset_id = ds.get("dataset_id")

    eda = leer_fila(supabase, "eda_results", user_id, dataset_id,
                     "corpus_stats,class_stats")
    nube = leer_fila(supabase, "wordcloud_results", user_id, dataset_id,
                      "total_words,unique_words")
    emb = leer_fila(supabase, "clustering_results", user_id, dataset_id,
                     "total,displayed")
    emo = leer_fila(supabase, "emotions_results", user_id, dataset_id,
                     "emotions")
    evos = filas_evolucion(supabase, user_id, dataset_id)
    modelos = modelos_del_dataset(dataset_name)

    detalles = {
        "estadisticas": "Resumen descriptivo del corpus y distribución por clase." if eda else "",
        "nube": (f"{nube.get('unique_words', '?')} palabras únicas sobre "
                 f"{nube.get('total_words', '?')} totales.") if nube else "",
        "embeddings": (f"{emb.get('displayed', '?')} puntos sobre "
                       f"{emb.get('total', '?')} documentos.") if emb and (emb.get("total") or 0) > 0 else "",
        "emociones": (f"{len(emo.get('emotions', []))} emociones detectadas.") if emo and emo.get("emotions") else "",
        "evolucion": "",
        "clasificador": "",
    }
    if evos:
        remitentes = sorted({str(f.get("remitente", "?")) for f in evos})
        detalles["evolucion"] = (
            f"{len(evos)} análisis guardados "
            f"({', '.join(remitentes[:3])}{'...' if len(remitentes) > 3 else ''})."
        )
    if modelos:
        detalles["clasificador"] = (
            f"{len(modelos)} modelo(s): "
            f"{', '.join(m.get('nombre_archivo', '') for m in modelos[:2])}."
        )

    disponibles = {
        "estadisticas": eda is not None,
        "nube": nube is not None,
        "embeddings": emb is not None and (emb.get("total") or 0) > 0,
        "emociones": emo is not None and bool(emo.get("emotions")),
        "evolucion": len(evos) > 0,
        "clasificador": len(modelos) > 0,
    }

    secciones = []
    for sid in SECCIONES:
        ok = disponibles.get(sid, False)
        secciones.append({
            "id": sid,
            "titulo": TITULOS[sid],
            "detalle": detalles.get(sid) or "",
            "disponible": ok,
            "motivo": "" if ok else "Aún no realizado para este dataset.",
        })

    return {
        "dataset": dataset_name,
        "dataset_id": dataset_id,
        "secciones": secciones,
    }


def leer_fuente(supabase, seccion_id: str, dataset_name: str, user_id: str, dataset_id: str):
    if seccion_id == "estadisticas":
        return leer_fila(supabase, "eda_results", user_id, dataset_id,
                         "corpus_stats,class_stats,sentence_stats,sentence_freq,distribution")
    if seccion_id == "nube":
        return leer_fila(supabase, "wordcloud_results", user_id, dataset_id,
                         "frequencies,total_words,unique_words,max_count,min_count,documents_processed")
    if seccion_id == "embeddings":
        return leer_fila(supabase, "clustering_results", user_id, dataset_id,
                         "points,clusters,total,displayed,outliers,truncated")
    if seccion_id == "emociones":
        return leer_fila(supabase, "emotions_results", user_id, dataset_id,
                         "emotions,corpus_stats,class_stats,correlation,distribution,top_words,class_top_words")
    if seccion_id == "evolucion":
        try:
            respuesta = (
                supabase.table("evolution_results")
                .select("remitente,modo,top_k,cantidad_mensajes,payload,updated_at")
                .eq("dataset_id", dataset_id)
                .eq("user_id", user_id)
                .execute()
            )
            return {"filas": respuesta.data or []}
        except Exception:
            return None
    if seccion_id == "clasificador":
        modelos = modelos_del_dataset(dataset_name)
        detalle = []
        for m in modelos:
            info = {"model_id": m["model_id"], "nombre_archivo": m["nombre_archivo"],
                    "metricas": m["metricas"]}
            datos = leer_checkpoint_clasificador(m["model_id"])
            if datos is not None:
                info.update(datos)
            detalle.append(info)
        return {"modelos": detalle} if detalle else None
    return None


def leer_checkpoint_clasificador(model_id: str):
    import re as _re
    ruta = os.path.join(RESULTADOS_PATH, MODELOS_DIRNAME, f"{model_id}.zip")
    try:
        with zipfile.ZipFile(ruta) as zf:
            texto = zf.read("README.md").decode("utf-8", errors="ignore")
    except Exception:
        return None
    clases = _re.findall(r"-\s*`(.+?)`\s*→\s*`(\d+)`", texto)
    matriz = None
    m = _re.search(r"\| Real / Predicha[^\n]*\n\|(?:---\|)+\n((?:\|[^\n]*\n)+)", texto)
    if m:
        filas = []
        for linea in m.group(1).strip().splitlines():
            celdas = [c.strip() for c in linea.strip().strip("|").split("|")]
            try:
                filas.append([int(x) for x in celdas[1:]])
            except ValueError:
                continue
        if filas and all(len(f) == len(filas) for f in filas):
            matriz = filas
    return {"clases": [c for c, _ in clases], "matriz": matriz}
