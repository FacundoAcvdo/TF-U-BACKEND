import os

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
}

FUENTE_POR_SECCION = {
    "estadisticas": "estadisticas",
    "nube": "nube",
    "embeddings": "embeddings",
    "emociones": "emociones",
    "evolucion": "evolucion",
    "topicos": "topicos",
    "chats": "chats",
    "riesgo": "riesgo",
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


def filas_topicos(supabase, user_id: str, dataset_id: str):
    try:
        respuesta = (
            supabase.table("topicmodeling_results")
            .select("document_count,topic_count,outlier_count,outlier_percentage,topics,max_topics")
            .eq("dataset_id", dataset_id)
            .eq("user_id", user_id)
            .execute()
        )
        filas = respuesta.data or []
        if not filas:
            return None
        return max(filas, key=lambda f: int(f.get("topic_count") or 0))
    except Exception:
        return None


def filas_chats(supabase, user_id: str, dataset_id: str, limite_por_chat: int = 200):
    try:
        resp = (
            supabase.table("chats")
            .select("chat_id,fecha_creacion,dataset_id,activo")
            .eq("dataset_id", dataset_id)
            .eq("user_id", user_id)
            .order("fecha_creacion", desc=False)
            .execute()
        )
        chats = resp.data or []
    except Exception:
        return None
    if not chats:
        return None
    salida = []
    for c in chats:
        cid = c.get("chat_id")
        try:
            rm = (
                supabase.table("chat_mensajes")
                .select("autor,mensaje,created_at")
                .eq("chat_id", cid)
                .eq("user_id", user_id)
                .order("created_at")
                .limit(limite_por_chat)
                .execute()
            )
            mensajes = rm.data or []
        except Exception:
            mensajes = []
        salida.append({
            "chat_id": str(cid)[:8],
            "fecha_creacion": c.get("fecha_creacion"),
            "mensajes": mensajes,
        })
    total_mensajes = sum(len(c["mensajes"]) for c in salida)
    if total_mensajes == 0:
        return None
    return {"chats": salida, "total_chats": len(salida),
            "total_mensajes": total_mensajes}


def filas_riesgo(supabase, user_id: str, dataset_id: str):
    return leer_fila(supabase, "suicide_results", user_id, dataset_id,
                     "modelo_version,trained_at,thresholds,resumen,por_clase,"
                     "skipped_sin_embedding,verificados_b,top_items,disclaimer")


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
    tops = filas_topicos(supabase, user_id, dataset_id)
    rac = filas_chats(supabase, user_id, dataset_id)
    rgo = filas_riesgo(supabase, user_id, dataset_id)

    detalles = {
        "estadisticas": "Resumen descriptivo del corpus y distribución por clase." if eda else "",
        "nube": (f"{nube.get('unique_words', '?')} palabras únicas sobre "
                 f"{nube.get('total_words', '?')} totales.") if nube else "",
        "embeddings": (f"{emb.get('displayed', '?')} puntos sobre "
                       f"{emb.get('total', '?')} documentos.") if emb and (emb.get("total") or 0) > 0 else "",
        "emociones": (f"{len(emo.get('emotions', []))} emociones detectadas.") if emo and emo.get("emotions") else "",
        "evolucion": "",
    }
    if evos:
        remitentes = sorted({str(f.get("remitente", "?")) for f in evos})
        detalles["evolucion"] = (
            f"{len(evos)} análisis guardados "
            f"({', '.join(remitentes[:3])}{'...' if len(remitentes) > 3 else ''})."
        )
    if tops:
        detalles["topicos"] = (
            f"{tops.get('topic_count', '?')} tópicos sobre "
            f"{tops.get('document_count', '?')} documentos."
        )
    if rac:
        detalles["chats"] = (
            f"{rac.get('total_chats', '?')} chats con "
            f"{rac.get('total_mensajes', '?')} mensajes."
        )
    if rgo:
        res = rgo.get("resumen", {}) or {}
        detalles["riesgo"] = (
            f"{res.get('total', '?')} casos, "
            f"{res.get('pct_alto_potencial', '?')}% alto/potencial."
        )

    disponibles = {
        "estadisticas": eda is not None,
        "nube": nube is not None,
        "embeddings": emb is not None and (emb.get("total") or 0) > 0,
        "emociones": emo is not None and bool(emo.get("emotions")),
        "evolucion": len(evos) > 0,
        "topicos": tops is not None,
        "chats": rac is not None,
        "riesgo": rgo is not None,
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
    if seccion_id == "topicos":
        return filas_topicos(supabase, user_id, dataset_id)
    if seccion_id == "chats":
        return filas_chats(supabase, user_id, dataset_id)
    if seccion_id == "riesgo":
        return filas_riesgo(supabase, user_id, dataset_id)
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
    return None
