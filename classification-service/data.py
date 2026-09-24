import io
import os
import re
import uuid

import numpy as np


MODEL_STORAGE_PATH = os.environ.get("MODEL_STORAGE_PATH", "/app/data/modelos")

MODEL_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def obtener_datos_dataset(
    collection,
    where_filter: dict,
    incluir_embeddings: bool,
    incluir_documentos: bool,
    limite_por_pagina: int = 2000,
):
    include_fields = ["metadatas"]
    if incluir_embeddings:
        include_fields.append("embeddings")
    if incluir_documentos:
        include_fields.append("documents")

    clases: list[str] = []
    documentos: list[str] = []
    bloques_embeddings: list[np.ndarray] = []

    offset = 0
    while True:
        pagina = collection.get(
            where=where_filter,
            include=include_fields,
            limit=limite_por_pagina,
            offset=offset,
        )

        metadatas = pagina.get("metadatas") or []
        n_pagina = len(metadatas)
        if n_pagina == 0:
            break

        clases.extend(str(m.get("clase", "")).strip() if m else "" for m in metadatas)

        if incluir_documentos:
            documentos.extend(pagina.get("documents") or [])

        if incluir_embeddings:
            emb = pagina.get("embeddings")
            if emb is not None and len(emb) > 0:
                bloques_embeddings.append(np.asarray(emb, dtype=np.float32))

        offset += n_pagina
        if n_pagina < limite_por_pagina:
            break

    x_vectors = (
        np.concatenate(bloques_embeddings, axis=0) if bloques_embeddings else None
    )
    return clases, documentos, x_vectors


def guardar_modelo_en_disco(zip_buffer: io.BytesIO, nombre_archivo: str) -> str:
    os.makedirs(MODEL_STORAGE_PATH, exist_ok=True)
    model_id = uuid.uuid4().hex
    ruta_zip = os.path.join(MODEL_STORAGE_PATH, f"{model_id}.zip")
    with open(ruta_zip, "wb") as f:
        f.write(zip_buffer.getvalue())
    with open(os.path.join(MODEL_STORAGE_PATH, f"{model_id}.name"), "w", encoding="utf-8") as f:
        f.write(nombre_archivo)
    return model_id


def leer_nombre_archivo(model_id: str) -> str:
    try:
        with open(os.path.join(MODEL_STORAGE_PATH, f"{model_id}.name"), encoding="utf-8") as f:
            nombre = f.read().strip()
            if nombre:
                return nombre
    except OSError:
        pass
    return f"{model_id}.zip"
