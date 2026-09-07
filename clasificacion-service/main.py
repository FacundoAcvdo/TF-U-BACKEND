import base64
import io
import os
import zipfile
from datetime import datetime, timezone
from typing import Literal, Optional

import chromadb
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from supabase import create_client
import torch

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-roberta-large-v1"  # Cambiar por el que usemos después

app = FastAPI()

origins = ["http://localhost:8080", "http://localhost:3000", "http://127.0.0.1:3000"]

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


class TrainingRequest(BaseModel):
    dataset: str
    modelo: Literal[
        "Logistic Regression",
        "Support Vector Machine",
        "Random Forest",
    ] = "Logistic Regression"
    modo: Literal["validacion_cruzada", "porcentajes"]
    k: Optional[int] = 5
    porcentaje_entrenamiento: Optional[int] = 50
    porcentaje_prueba: Optional[int] = 50


class ResumenMenciones(BaseModel):
    truePositive: int
    falsePositive: int
    trueNegative: int
    falseNegative: int
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    total_muestras: int
    muestras_train: int
    muestras_test: int


class TrainingResponse(BaseModel):
    resumen: ResumenMenciones
    archivo_base64: str
    nombre_archivo: str


def instanciar_clasificador(nombre: str):
    if nombre == "Support Vector Machine":
        return LinearSVC(random_state=42, max_iter=3000, class_weight="balanced")
    if nombre == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    return LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")


def procesar_etiquetas_automatico(metadatas: list[dict]):
    if not metadatas:
        raise HTTPException(status_code=400, detail="No se encontraron metadatos en ChromaDB.")

    valores_raw = [str(meta.get("clase", "")).strip() for meta in metadatas]

    le = LabelEncoder()
    y = le.fit_transform(valores_raw)

    clases_unicas = list(le.classes_)

    if len(clases_unicas) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"El dataset contiene una única clase ('{clases_unicas[0]}'). Se requieren al menos dos para entrenar.",
        )

    label_mapping = {clase: int(idx) for idx, clase in enumerate(clases_unicas)}
    print(f"[CLASIFICACION] Clases detectadas automáticamente: {label_mapping}")

    return y, label_mapping, clases_unicas


def generar_readme(
    nombre_modelo: str,
    encoder_base: str,
    label_mapping: dict,
    metricas: dict,
    modo: str,
    total_muestras: int,
) -> str:
    mapping_lines = "\n".join(f"- `{clase}` → `{idx}`" for clase, idx in label_mapping.items())

    return f"""# Modelo de clasificación

Generado: {datetime.now(timezone.utc).isoformat()}

## Contenido del artefacto

- `modelo.pt`: checkpoint de PyTorch (`torch.save`/`torch.load`) que contiene:
  - `clasificador`: el modelo de scikit-learn ya entrenado ({nombre_modelo}).
  - `encoder_base`: nombre del modelo de embeddings usado para vectorizar el texto.
  - `label_mapping`: mapeo clase → id numérico.
  - `clases_nombres`: lista de clases en el orden que usa el encoder.
  - Métricas de evaluación (`accuracy`, `f1_score`, `precision`, `recall`).

## Configuración de entrenamiento

- Modelo: **{nombre_modelo}**
- Modo de evaluación: **{modo}**
- Total de muestras: **{total_muestras}**
- Accuracy: **{metricas['accuracy']:.4f}**
- F1-score (weighted): **{metricas['f1_score']:.4f}**
- Precision (weighted): **{metricas['precision']:.4f}**
- Recall (weighted): **{metricas['recall']:.4f}**

## Mapeo de clases

{mapping_lines}

## Cómo usar este modelo para clasificar texto nuevo

Necesitás el mismo modelo de embeddings que se usó para entrenar
(`{encoder_base}`), instalado vía `sentence-transformers`.

```python
import torch
from sentence_transformers import SentenceTransformer

# 1. Cargar el checkpoint
checkpoint = torch.load("modelo.pt", weights_only=False)

clf = checkpoint["clasificador"]
encoder_base = checkpoint["encoder_base"]
label_mapping = checkpoint["label_mapping"]
clases_nombres = checkpoint["clases_nombres"]

# 2. Cargar el mismo encoder de embeddings usado en el entrenamiento
encoder = SentenceTransformer(encoder_base)

# 3. Vectorizar el/los texto(s) nuevo(s)
textos = ["texto de ejemplo a clasificar"]
embeddings = encoder.encode(textos)

# 4. Predecir
prediccion_id = clf.predict(embeddings)
prediccion_clase = [clases_nombres[i] for i in prediccion_id]

print(prediccion_clase)
```

## Requisitos

```
pip install torch scikit-learn sentence-transformers
```

## Notas

- El clasificador espera embeddings de la misma dimensionalidad y del mismo
  encoder con el que fue entrenado (`{encoder_base}`). Usar un encoder distinto
  produce predicciones sin sentido.
- `label_mapping` y `clases_nombres` sirven para traducir la salida numérica
  del clasificador de vuelta al nombre real de la clase.
"""


@app.post("/detectar_menciones", response_model=TrainingResponse)
async def detectar_menciones(
    data: TrainingRequest,
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

        client = chromadb.HttpClient(host="chroma-service", port=8000)
        collection = client.get_collection(name="documents")

        resultados = collection.get(
            where={
                "$and": [
                    {"user_id": x_user_id},
                    {"dataset_id": dataset_id},
                ]
            },
            include=["embeddings", "metadatas"],
        )

        embeddings = resultados.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            raise HTTPException(
                status_code=400,
                detail="No se encontraron embeddings en ChromaDB para este dataset.",
            )

        x_vectors = np.array(embeddings, dtype=np.float32)
        metadatas = resultados.get("metadatas") or []

        y_labels, label_mapping, clases_nombres = procesar_etiquetas_automatico(metadatas)

        total_muestras = len(y_labels)
        clf = instanciar_clasificador(data.modelo)

        if data.modo == "validacion_cruzada":
            k_folds = data.k if data.k in [5, 10] else 5
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            y_pred = cross_val_predict(clf, x_vectors, y_labels, cv=skf)
            y_eval = y_labels
            clf.fit(x_vectors, y_labels)
            muestras_train = total_muestras
            muestras_test = total_muestras
        else:
            porcentaje_prueba = data.porcentaje_prueba or 50
            test_size = max(0.1, min(0.9, porcentaje_prueba / 100.0))
            x_train, x_test, y_train, y_test = train_test_split(
                x_vectors,
                y_labels,
                test_size=test_size,
                random_state=42,
                stratify=y_labels,
            )
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            y_eval = y_test
            muestras_train = len(y_train)
            muestras_test = len(y_test)

        tn, fp, fn, tp = confusion_matrix(y_eval, y_pred, labels=[0, 1]).ravel()
        acc = float(accuracy_score(y_eval, y_pred))
        f1 = float(f1_score(y_eval, y_pred, average="weighted"))
        prec = float(precision_score(y_eval, y_pred, average="weighted", zero_division=0))
        rec = float(recall_score(y_eval, y_pred, average="weighted", zero_division=0))

        metricas = {"accuracy": acc, "f1_score": f1, "precision": prec, "recall": rec}

        checkpoint_buffer = io.BytesIO()
        checkpoint = {
            "clasificador": clf,
            "encoder_base": EMBEDDING_MODEL_NAME,
            "label_mapping": label_mapping,
            "clases_nombres": clases_nombres,
            **metricas,
        }
        torch.save(checkpoint, checkpoint_buffer)
        checkpoint_buffer.seek(0)

        readme_texto = generar_readme(
            nombre_modelo=data.modelo,
            encoder_base=EMBEDDING_MODEL_NAME,
            label_mapping=label_mapping,
            metricas=metricas,
            modo=data.modo,
            total_muestras=total_muestras,
        )

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("modelo.pt", checkpoint_buffer.read())
            zf.writestr("README.md", readme_texto)
        zip_buffer.seek(0)

        archivo_base64 = base64.b64encode(zip_buffer.read()).decode("utf-8")
        file_name = f"{data.modelo.lower().replace(' ', '_')}_modelo.zip"

        return TrainingResponse(
            resumen=ResumenMenciones(
                truePositive=int(tp),
                falsePositive=int(fp),
                trueNegative=int(tn),
                falseNegative=int(fn),
                accuracy=acc,
                f1_score=f1,
                precision=prec,
                recall=rec,
                total_muestras=total_muestras,
                muestras_train=muestras_train,
                muestras_test=muestras_test,
            ),
            archivo_base64=archivo_base64,
            nombre_archivo=file_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error en deteccion de menciones: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))