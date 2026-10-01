import os
import re
import asyncio
from collections import Counter
from typing import Literal, Optional

import chromadb
import pandas as pd
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")

STOPWORDS= set(stopwords.words('spanish')).union(set(stopwords.words('english')))

URL_RE = re.compile(r"https?://\S+|www\.\S+")
TOKEN_RE = re.compile(r"[a-záéíóúñü]{3,}", re.IGNORECASE)

PALABRAS_IGNORADAS_MEDIA = {
    "multimedia", "omitido", "omitida", "audio", "imagen", "video", "sticker",
    "gif", "documento", "contacto", "ubicación", "mensaje", "eliminado",
    "eliminaste", "editado", "llamada", "perdida", "archivo", "adjunto",
    "omitted", "image", "video", "sticker", "gif", "document", "contact",
    "location", "message", "deleted", "edited", "voice", "call", "missed",
    "attachment", "null", "media"
}

REMITENTES_SISTEMA = {
    "sistema",
    "system",
    "whatsapp",
}

USUARIO_GENERICO = "Usuario 1"

TABLA_CACHE = "evolucion_resultados"

OBJETIVO_BLOQUES = 10

app = FastAPI()

origins = ["http://localhost:8080", "http://localhost:3000", "http://127.0.0.1:3000",]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_supabase_client(user_jwt: str):
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    client.postgrest.auth(user_jwt)
    return client


CHROMA_HOST = os.environ.get("CHROMA_HOST", "chroma-service")
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "documents")
CHROMA_PAGE_SIZE = int(os.environ.get("CHROMA_PAGE_SIZE", "2000"))

_chroma_client = None
_chroma_collection = None


def get_chroma_collection():
    """Singleton: no crear HttpClient por request."""
    global _chroma_client, _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    if _chroma_client is None:
        _chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    _chroma_collection = _chroma_client.get_collection(name=CHROMA_COLLECTION)
    return _chroma_collection


def obtener_resultados_paginados(
    collection,
    where_filter: dict,
    include_fields: list,
    limite_por_pagina: int = 10000
) -> dict:
    """Paginado robusto: `ids` como conteo canónico (no include_fields[0])."""
    resultados_acumulados = {campo: [] for campo in include_fields}
    resultados_acumulados["ids"] = []
    offset = 0

    while True:
        pagina = _obtener_pagina_sync(
            collection, where_filter, include_fields, limite_por_pagina, offset
        )

        ids = pagina.get("ids") or []
        if not ids:
            break

        if not datos_pagina:
            hay_mas_resultados = False
        else:
            for campo in include_fields:
                if pagina.get(campo):
                    resultados_acumulados[campo].extend(pagina[campo])

            if len(datos_pagina) < limite_por_pagina:
                hay_mas_resultados = False
            else:
                offset += limite_por_pagina

    return resultados_acumulados


async def obtener_resultados_paginados(
    collection,
    where_filter: dict,
    include_fields: list,
    limite_por_pagina: int = CHROMA_PAGE_SIZE,
) -> dict:
    """Wrapper async: el HttpClient de Chroma es bloqueante → to_thread."""
    return await asyncio.to_thread(
        obtener_resultados_paginados_sync,
        collection,
        where_filter,
        include_fields,
        limite_por_pagina,
    )

def tokenizar(texto: str) -> list[str]:
    texto = URL_RE.sub(" ", texto.lower())
    tokens = TOKEN_RE.findall(texto)
    return [
        t
        for t in tokens
        if t not in STOPWORDS and t not in PALABRAS_IGNORADAS_MEDIA
    ]


def contar_palabras(texto: str) -> int:
    return len((texto or "").split())


def particionar_por_fecha(
    df: pd.DataFrame,
) -> tuple[pd.core.groupby.generic.DataFrameGroupBy, str]:

    fecha_min, fecha_max = df["fecha"].min(), df["fecha"].max()
    dias_totales = (fecha_max - fecha_min).days

    if dias_totales <= 60:
        freq, granularidad = "W", "semanal"
    elif dias_totales <= 400:
        freq, granularidad = "M", "mensual"
    elif dias_totales <= 1500:
        freq, granularidad = "Q", "trimestral"
    else:
        freq, granularidad = "Y", "anual"

    df = df.copy()
    df["periodo"] = df["fecha"].dt.to_period(freq)

    return df.groupby("periodo"), granularidad


def etiquetar_periodo(periodo: pd.Period, granularidad: str) -> str:
    meses = {
        1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
        7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
    }

    if granularidad == "semanal":
        inicio = periodo.start_time.strftime("%d/%m/%Y")
        fin = periodo.end_time.strftime("%d/%m/%Y")
        return f"{inicio} — {fin}"

    if granularidad == "mensual":
        return f"{meses[periodo.month]} {periodo.year}"

    if granularidad == "trimestral":
        mes_inicio = periodo.start_time.month
        mes_fin = periodo.end_time.month
        return f"{meses[mes_inicio]} — {meses[mes_fin]} {periodo.year}"

    return str(periodo.year)


def clave_orden(meta: dict) -> tuple[int, int]:
    try:
        return (0, int((meta or {}).get("id")))
    except (TypeError, ValueError):
        return (1, 0)


def particionar_por_bloques(
    textos_ordenados: list[str],
    objetivo_bloques: int = OBJETIVO_BLOQUES,
) -> list[tuple[int, int, list[str]]]:
    total_palabras = sum(contar_palabras(t) for t in textos_ordenados)
    n = len(textos_ordenados)
    if n == 0:
        return []

    palabras_por_bloque = max(1, total_palabras // objetivo_bloques)

    bloques: list[tuple[int, int, list[str]]] = []
    actual: list[str] = []
    actual_palabras = 0
    inicio = 0
    for i, texto in enumerate(textos_ordenados):
        actual.append(texto)
        actual_palabras += contar_palabras(texto)
        if actual_palabras >= palabras_por_bloque and i < n - 1:
            bloques.append((inicio, i, actual))
            actual = []
            actual_palabras = 0
            inicio = i + 1
    if actual:
        bloques.append((inicio, n - 1, actual))
    return bloques


def palabras_significativas(mensajes: list[str], top_k: int) -> list[dict]:
    contador = Counter()

    for mensaje in mensajes:
        contador.update(tokenizar(mensaje))

    return [
        {"palabra": palabra, "cantidad": cantidad}
        for palabra, cantidad in contador.most_common(top_k)
    ]


def buscar_cache(supabase, x_user_id: str, dataset_id: str, remitente: str,
                 modo: str, top_k: int, cantidad: int):
    try:
        respuesta = (
            supabase.table(TABLA_CACHE)
            .select("*")
            .eq("user_id", x_user_id)
            .eq("dataset_id", dataset_id)
            .eq("remitente", remitente)
            .eq("modo", modo)
            .eq("top_k", top_k)
            .limit(1)
            .execute()
        )
        if respuesta.data:
            fila = respuesta.data[0]
            if int(fila.get("cantidad_mensajes", -1)) == cantidad:
                return fila.get("payload")
    except Exception as e:
        print(f"Aviso: no se pudo leer el caché de evolución: {str(e)}")
    return None


def guardar_cache(supabase, x_user_id: str, dataset_id: str, remitente: str,
                  modo: str, top_k: int, cantidad: int, payload: dict):
    try:
        supabase.table(TABLA_CACHE).upsert(
            {
                "user_id": x_user_id,
                "dataset_id": dataset_id,
                "remitente": remitente,
                "modo": modo,
                "top_k": top_k,
                "cantidad_mensajes": cantidad,
                "payload": payload,
            },
            on_conflict="user_id,dataset_id,remitente,modo,top_k",
        ).execute()
    except Exception as e:
        print(f"Aviso: no se pudo guardar el caché de evolución: {str(e)}")


class UsuariosRequest(BaseModel):
    dataset: str


class EvolucionPalabrasRequest(BaseModel):
    dataset: str
    remitente: str
    top_k: Optional[int] = 10
    modo: Literal["auto", "fecha", "bloques"] = "auto"


@app.post("/users")
async def obtener_usuarios(
    data: UsuariosRequest,
    x_user_id: str = Header(..., alias="X-User-ID"),
    x_user_token: str = Header(..., alias="X-User-Token"),
):
    try:
        supabase = get_supabase_client(x_user_token)

        response = (
            supabase.table("datasets")
            .select("dataset_id")
            .eq("dataset_name", data.dataset)
            .eq("user_id", x_user_id)
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{data.dataset}' no encontrado o sin permisos.",
            )

        dataset_id = response.data[0].get("dataset_id")

        collection = get_chroma_collection()

        filtros = {
            "$and": [
                {"user_id": x_user_id},
                {"dataset_id": dataset_id},
            ]
        }
        resultados_totales = await obtener_resultados_paginados(collection, filtros, ["metadatas"])
        metadatas = resultados_totales.get("metadatas", [])

        if not metadatas:
            raise HTTPException(
                status_code=404,
                detail="No se encontraron mensajes para este dataset.",
            )

        hay_fechas = any(meta.get("fecha") for meta in metadatas)

        remitentes_raw = []
        for meta in metadatas:
            nombre = (meta.get("remitente") or "").strip()
            if not nombre:
                if hay_fechas:
                    continue
                nombre = USUARIO_GENERICO
            if nombre.strip().lower() in REMITENTES_SISTEMA:
                continue
            remitentes_raw.append(nombre)

        if not remitentes_raw:
            raise HTTPException(
                status_code=422,
                detail="No se encontraron participantes válidos en este dataset.",
            )

        conteo = pd.Series(remitentes_raw).value_counts()


        return {
            "usuarios": [
                {
                    "remitente": nombre,
                    "cantidad_mensajes": int(cantidad),
                }
                for nombre, cantidad in conteo.items()
            ],
            "total_mensajes": int(conteo.sum()),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error al obtener usuarios del chat: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


@app.post("/word-evolution")
async def evolucion_palabras(
    data: EvolucionPalabrasRequest,
    x_user_id: str = Header(..., alias="X-User-ID"),
    x_user_token: str = Header(..., alias="X-User-Token"),
):
    try:
        supabase = get_supabase_client(x_user_token)

        response = (
            supabase.table("datasets")
            .select("dataset_id")
            .eq("dataset_name", data.dataset)
            .eq("user_id", x_user_id)
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{data.dataset}' no encontrado o sin permisos.",
            )

        dataset_id = response.data[0].get("dataset_id")

        collection = get_chroma_collection()

        es_generico = data.remitente == USUARIO_GENERICO
        if es_generico:
            filtros = {
                "$and": [
                    {"user_id": x_user_id},
                    {"dataset_id": dataset_id},
                ]
            }
        else:
            filtros = {
                "$and": [
                    {"user_id": x_user_id},
                    {"dataset_id": dataset_id},
                    {"remitente": data.remitente},
                ]
            }
        resultados_totales = obtener_resultados_paginados(collection, filtros, ["documents", "metadatas"])

        documentos = resultados_totales.get("documents", [])
        metadatas = resultados_totales.get("metadatas", [])

        if es_generico:
            pares = [
                (doc, meta) for doc, meta in zip(documentos, metadatas)
                if not (meta.get("remitente") or "").strip()
            ]
            if pares:
                documentos = [doc for doc, _ in pares]
                metadatas = [meta for _, meta in pares]

        if not documentos:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontraron mensajes de '{data.remitente}' en este dataset.",
            )

        top_k = max(1, min(data.top_k or 10, 30))

        fechas_raw = [meta.get("fecha") for meta in metadatas]
        tiene_fechas = all(f is not None for f in fechas_raw)

        modo_pedido = data.modo or "auto"
        if modo_pedido == "auto":
            modo = "fecha" if tiene_fechas else "bloques"
        else:
            modo = modo_pedido

        if modo == "fecha" and not tiene_fechas:
            raise HTTPException(
                status_code=422,
                detail="Algunos mensajes no tienen fecha registrada. No se puede armar la línea de tiempo.",
            )

        cache = buscar_cache(supabase, x_user_id, dataset_id, data.remitente, modo, top_k, len(documentos))
        if cache is not None:
            cache["desde_cache"] = True
            return cache

        if modo == "fecha":
            resultado = evolucion_por_fecha(documentos, fechas_raw, top_k, data)
        else:
            resultado = evolucion_por_bloques(documentos, metadatas, top_k, data)

        resultado["desde_cache"] = False
        guardar_cache(supabase, x_user_id, dataset_id, data.remitente,modo, top_k, len(documentos), resultado)
        return resultado

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error al calcular evolución de palabras: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


def evolucion_por_fecha(documentos: list[str], fechas_raw: list,
                        top_k: int, data: EvolucionPalabrasRequest) -> dict:
    df = pd.DataFrame(
        {
            "texto": documentos,
            "fecha": pd.to_datetime(fechas_raw),
        }
    ).sort_values("fecha")

    grupos_por_periodo, granularidad = particionar_por_fecha(df)

    periodos_respuesta = []

    for periodo, grupo in grupos_por_periodo:
        palabras_top = palabras_significativas(
            grupo["texto"].tolist(),
            top_k,
        )

        if not palabras_top:
            continue

        periodos_respuesta.append(
            {
                "etiqueta": etiquetar_periodo(
                    periodo,
                    granularidad,
                ),
                "fecha_inicio": periodo.start_time.strftime("%Y-%m-%d"),
                "fecha_fin": periodo.end_time.strftime("%Y-%m-%d"),
                "cantidad_mensajes": int(len(grupo)),
                "palabras_top": palabras_top,
            }
        )

    if not periodos_respuesta:
        raise HTTPException(
            status_code=422,
            detail="No quedaron palabras significativas tras limpiar el texto de este remitente.",
        )

    return {
        "remitente": data.remitente,
        "modo_usado": "fecha",
        "granularidad": granularidad,
        "rango_fechas": {
            "desde": df["fecha"].min().strftime("%Y-%m-%d"),
            "hasta": df["fecha"].max().strftime("%Y-%m-%d"),
        },
        "total_mensajes": int(len(df)),
        "periodos": periodos_respuesta,
    }


def evolucion_por_bloques(documentos: list[str], metadatas: list,
                          top_k: int, data: EvolucionPalabrasRequest) -> dict:
    pares = list(zip(documentos, metadatas))
    pares.sort(key=lambda p: clave_orden(p[1]))
    orden_estimado = any(clave_orden(meta)[0] == 1 for _, meta in pares)
    textos_ordenados = [doc for doc, _ in pares]

    bloques = particionar_por_bloques(textos_ordenados)
    total_palabras = sum(contar_palabras(t) for t in textos_ordenados)

    bloques_respuesta = []
    for i, (inicio, fin, textos) in enumerate(bloques, start=1):
        palabras_top = palabras_significativas(textos, top_k)
        if not palabras_top:
            continue
        bloques_respuesta.append(
            {
                "etiqueta": f"Bloque {i} · msgs {inicio + 1}–{fin + 1}",
                "desde_msg": inicio + 1,
                "hasta_msg": fin + 1,
                "cantidad_mensajes": int(len(textos)),
                "cantidad_palabras": int(sum(contar_palabras(t) for t in textos)),
                "palabras_top": palabras_top,
            }
        )

    if not bloques_respuesta:
        raise HTTPException(
            status_code=422,
            detail="No quedaron palabras significativas tras limpiar el texto de este remitente.",
        )

    return {
        "remitente": data.remitente,
        "modo_usado": "bloques",
        "granularidad": "bloques",
        "orden_estimado": orden_estimado,
        "total_mensajes": int(len(textos_ordenados)),
        "total_palabras": int(total_palabras),
        "cantidad_bloques": int(len(bloques_respuesta)),
        "bloques": bloques_respuesta,
    }
