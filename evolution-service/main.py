import os
import re
from collections import Counter
from typing import Optional

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

STOPWORDS_NLTK = set(stopwords.words('spanish')).union(set(stopwords.words('english')))

JERGA_CHAT = {
    "jaja", "jajaja", "jaj", "jajaj", "jajajaja", "ja", "ok", "dale", "che", "vos", 
    "usted", "ustedes", "ah", "eh", "mm", "mmm", "ehh", "ay", "wow", "oh", "asi", 
    "así", "aca", "acá", "ahi", "ahí", "alli", "allí", "solo", "sólo", "pues", 
    "bueno", "bien", "vez", "veces", "cada", "cosa", "cosas", "si", "dijo", "dice", 
    "digo", "decir", "hacer", "hace", "hizo", "haciendo", "puede", "pueden", "puedo", 
    "podemos", "va", "voy", "vas", "vamos", "van", "tengo", "tienes", "tiene", 
    "tenemos", "tenéis", "tienen", "yeah", "yes", "nope", "haha", "lol", "idk", 
    "okay", "omg", "im", "ur", "u", "thx", "thanks", "pls", "please", "rt",
}

STOPWORDS = STOPWORDS_NLTK.union(JERGA_CHAT)

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


def get_chroma_collection():
    client = chromadb.HttpClient(host="chroma-service", port=8000)
    return client.get_collection(name="documents")


def obtener_resultados_paginados(
    collection, 
    where_filter: dict, 
    include_fields: list, 
    limite_por_pagina: int = 10000
) -> dict:
    resultados_acumulados = {campo: [] for campo in include_fields}
    offset = 0
    hay_mas_resultados = True

    while hay_mas_resultados:
        pagina = collection.get(
            where=where_filter,
            include=include_fields,
            limit=limite_por_pagina,
            offset=offset,
        )

        campo_base = include_fields[0]
        datos_pagina = pagina.get(campo_base) or []

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

def tokenizar(texto: str) -> list[str]:
    texto = URL_RE.sub(" ", texto.lower())
    tokens = TOKEN_RE.findall(texto)
    return [
        t
        for t in tokens
        if t not in STOPWORDS and t not in PALABRAS_IGNORADAS_MEDIA
    ]


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


def palabras_significativas(mensajes: list[str], top_k: int) -> list[dict]:
    contador = Counter()

    for mensaje in mensajes:
        contador.update(tokenizar(mensaje))

    return [
        {"palabra": palabra, "cantidad": cantidad}
        for palabra, cantidad in contador.most_common(top_k)
    ]


class UsuariosRequest(BaseModel):
    dataset: str


class EvolucionPalabrasRequest(BaseModel):
    dataset: str
    remitente: str
    top_k: Optional[int] = 10


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
            .select("*")
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
        resultados_totales = obtener_resultados_paginados(collection, filtros, ["metadatas"])
        metadatas = resultados_totales.get("metadatas", [])

        if not metadatas:
            raise HTTPException(
                status_code=404,
                detail="No se encontraron mensajes para este dataset.",
            )

        remitentes_raw = [
            meta.get("remitente")
            for meta in metadatas
            if meta.get("remitente")
            and not meta.get("remitente").strip().lower() in REMITENTES_SISTEMA
        ]

        if not remitentes_raw:
            raise HTTPException(
                status_code=422,
                detail="No se encontraron participantes válidos en este dataset.",
            )

        conteo = pd.Series(remitentes_raw).value_counts()

        if len(conteo) < 2: # puede analizar una persona su propio chat. Pero creo que lo toma como dos personas
            raise HTTPException(
                status_code=422,
                detail=f"Este chat solo tiene un participante ('{conteo.index[0]}'). Se necesitan al menos 2 para comparar.",
            )

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
            .select("*")
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
                {"remitente": data.remitente},
            ]
        }
        resultados_totales = obtener_resultados_paginados(collection, filtros, ["documents", "metadatas"])
        
        documentos = resultados_totales.get("documents", [])
        metadatas = resultados_totales.get("metadatas", [])

        if not documentos:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontraron mensajes de '{data.remitente}' en este dataset.",
            )

        fechas_raw = [meta.get("fecha") for meta in metadatas]

        if any(f is None for f in fechas_raw):
            raise HTTPException(
                status_code=422,
                detail="Algunos mensajes no tienen fecha registrada. No se puede armar la línea de tiempo.",
            )

        df = pd.DataFrame(
            {
                "texto": documentos,
                "fecha": pd.to_datetime(fechas_raw),
            }
        ).sort_values("fecha")

        grupos_por_periodo, granularidad = particionar_por_fecha(df)

        top_k = max(1, min(data.top_k or 10, 30))

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
            "granularidad": granularidad,
            "rango_fechas": {
                "desde": df["fecha"].min().strftime("%Y-%m-%d"),
                "hasta": df["fecha"].max().strftime("%Y-%m-%d"),
            },
            "total_mensajes": int(len(df)),
            "periodos": periodos_respuesta,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error al calcular evolución de palabras: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )