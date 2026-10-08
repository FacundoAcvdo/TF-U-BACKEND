import asyncio
import os

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from core import get_collection, get_supabase_client
from data import (
    MODEL_ID_RE,
    MODEL_STORAGE_PATH,
    guardar_modelo_en_disco,
    leer_nombre_archivo,
    obtener_datos_dataset,
)
from schemas import ResumenMenciones, TrainingRequest, TrainingResponse
from training import (
    entrenar_bert,
    entrenar_sklearn,
    procesar_etiquetas_automatico,
    validar_tamano_muestras,
)


app = FastAPI()

origins = ["http://localhost:8080", "http://localhost:3000", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
            .select("dataset_id")
            .eq("dataset_name", data.dataset)
            .eq("user_id", x_user_id)
            .limit(1)
            .execute()
        )
        if not response.data:
            raise HTTPException(
                status_code=404,
                detail=f"Dataset '{data.dataset}' no encontrado o sin permisos.",
            )
        dataset_id = response.data[0].get("dataset_id")

        collection = get_collection()

        filtros = {
            "$and": [
                {"user_id": x_user_id},
                {"dataset_id": dataset_id},
            ]
        }

        es_bert = data.modelo == "BERT"
  
        clases, documentos, x_vectors = await asyncio.to_thread(
            obtener_datos_dataset,
            collection,
            filtros,
            not es_bert,
            es_bert,
        )

        y_labels, label_mapping, clases_nombres = procesar_etiquetas_automatico(clases)
        validar_tamano_muestras(y_labels, data)

        if es_bert:
            if not documentos:
                raise HTTPException(status_code=400, detail="No se encontraron textos para entrenar BERT.")
            resultado = entrenar_bert(documentos, y_labels, label_mapping, clases_nombres, data)
        else:
            if x_vectors is None or len(x_vectors) == 0:
                raise HTTPException(status_code=400, detail="No se encontraron embeddings.")
            if len(x_vectors) != len(y_labels):
                raise HTTPException(
                    status_code=500,
                    detail="Inconsistencia entre embeddings y etiquetas recuperados de ChromaDB.",
                )
            resultado = entrenar_sklearn(x_vectors, y_labels, label_mapping, clases_nombres, data)

        (
            tp,
            fp,
            tn,
            fn,
            acc,
            f1,
            prec,
            rec,
            total_muestras,
            muestras_train,
            muestras_test,
            zip_buffer,
            matriz,
        ) = resultado

        file_name = f"{data.modelo.lower().replace(' ', '_')}_modelo.zip"
        model_id = guardar_modelo_en_disco(zip_buffer, file_name)

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
                matriz_confusion=matriz,
                clases=clases_nombres,
            ),
            model_id=model_id,
            nombre_archivo=file_name,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/descargar_modelo")
async def descargar_modelo(model_id: str):
    if not MODEL_ID_RE.match(model_id or ""):
        raise HTTPException(status_code=404, detail="Modelo no encontrado.")
    ruta_zip = os.path.join(MODEL_STORAGE_PATH, f"{model_id}.zip")
    if not os.path.isfile(ruta_zip):
        raise HTTPException(status_code=404, detail="Modelo no encontrado.")
    return FileResponse(
        path=ruta_zip,
        media_type="application/zip",
        filename=leer_nombre_archivo(model_id),
    )
