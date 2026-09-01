import os
import uuid

from sklearn.feature_extraction.text import CountVectorizer
from storage3 import create_client as create_storage_client
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Header
from supabase import create_client
import matplotlib.pyplot as plt
from pydantic import BaseModel
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import json
import io
from io import BytesIO

###############

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
DATASET_STORAGE_PATH = os.environ.get("DATASET_STORAGE_PATH", "./data/datasets")

###############

matplotlib.use('Agg')
app = FastAPI()

origins = ["http://localhost:8080"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##############

class EdaRequest(BaseModel):
    dataset: str

def get_supabase_client(user_jwt: str):
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    client.postgrest.auth(user_jwt)
    return client

def get_storage_client(user_jwt: str):
    storage_url = f"{SUPABASE_URL}/storage/v1"
    headers = {
        "apiKey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {user_jwt}",
    }
    return create_storage_client(storage_url, headers, is_async=False)


def parse_dataset_json(ruta: str) -> pd.DataFrame:
    with open(ruta, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    records = []
    for i, doc in enumerate(json_data.get("documents", [])):
        row = {"texto": doc.get("texto"), "clase": doc.get("clase")}
        records.append(row)

    df = pd.DataFrame(records)

    fechas = [doc.get("fecha") for doc in json_data.get("documents", [])]
    if any(f is not None for f in fechas):
        df["fecha"] = fechas

    return df

###############

def get_dataset_row_analysis(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset['characters'] = dataset['texto'].astype(str).apply(len)
    dataset['words'] = dataset['texto'].apply(lambda x: len(x.split(' ')))
    dataset['sentences'] = dataset['texto'].str.count('[\w][\.!\?]').clip(lower=1)
    dataset['Avg-charperword'] = (dataset['characters']/dataset['words'])
    dataset['Avg-charpersentence'] = (dataset['characters']/dataset['sentences'])

    return dataset

def get_estadisticas_corpus(dataset: pd.DataFrame) -> pd.DataFrame:
    vect = CountVectorizer(token_pattern='[\/\%]+').fit(dataset['texto'])
    cant_texto = int(dataset.shape[0])
    tam_vocab = len(vect.vocabulary_)
    cant_palabras = dataset['words'].sum()
    avg_palabras = int(cant_palabras / cant_texto)
    senten = dataset['sentences']

    DF_CORPUS = pd.DataFrame([{
        'texto': cant_texto,
        'vocabulary': tam_vocab,
        'words': cant_palabras,
        'word_avg': avg_palabras,
        "sentences": senten.sum(),
        "sentence_avg": int(senten.sum() / len(dataset)),
        "sentence_min": senten.min(),
        "sentence_max": senten.max()
    }])

    return DF_CORPUS

def get_estadisticas_clases(dataset: pd.DataFrame) -> pd.DataFrame:
    filas_clases = []
    clases = np.unique(dataset['clase'])

    for clase in clases:
        grupo = dataset.groupby('clase')

        cant_preg = grupo.get_group(clase)['texto'].count()
        preguntas_clase = grupo[['clase', 'texto']].get_group(clase)
        tam_vocab_clase = len(CountVectorizer().fit(preguntas_clase['texto']).vocabulary_)
        cant_palabras = grupo.get_group(clase)['words'].sum()
        cant_sentencias = grupo.get_group(clase)['sentences'].sum()
        prom_palabras = int(cant_palabras / cant_preg)
        prom_sentencias = int(cant_sentencias / cant_preg)
        maxs = grupo.get_group(clase)['sentences'].max()
        mins = grupo.get_group(clase)['sentences'].min()

        filas_clases.append({
            'clase': clase,
            'texto': cant_preg,
            'vocabulary': tam_vocab_clase,
            'words': cant_palabras,
            'word_avg': prom_palabras,
            'sentences': cant_sentencias,
            'sentence_avg': prom_sentencias,
            "sentence_min": mins,
            "sentence_max": maxs
        })

    DF_CLASES = pd.DataFrame(filas_clases)

    return DF_CLASES

def generate_sentence_distribution_plot(dataset: pd.DataFrame):
    imagenes = []

    fig, ax = plt.subplots(figsize=(5.1111, 5.0), dpi=100)
    sns.boxplot(x=dataset["sentences"], ax=ax)
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    imagenes.append((buf.getvalue(), "boxplot"))

    sns.catplot(x=dataset["sentences"])
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    imagenes.append((buf.getvalue(), "catplot"))

    return imagenes

def generate_class_distribution_plot(dataset: pd.DataFrame):
    dataset.plot.bar(x='clase', logy=True)
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close()

    return [(buf.getvalue(), "class_distribution")]

def generate_distribution_plot(dataset: pd.DataFrame):
    clases = np.unique(dataset['clase'])
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(20, 10), dpi=100)

    for i, clase in enumerate(clases):
        grupo = dataset[dataset['clase'] == clase]
        color = plt.cm.tab10(i % 10)
        sns.histplot(grupo['words'], label=clase, ax=axes[0], color=color, kde=True)
        sns.histplot(grupo['sentences'], label=clase, ax=axes[1], color=color, kde=True)

    axes[0].set_title(f'Distribution of Words per text', fontsize=13)
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='x', labelsize=12)
    axes[0].tick_params(axis='y', labelsize=12)
    axes[0].legend()

    axes[1].set_title(f'Distribution of Sentences per text', fontsize=13)
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='x', labelsize=12)
    axes[1].tick_params(axis='y', labelsize=12)
    axes[1].legend()
    axes[1].set_xscale('log')

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)

    return [(buf.getvalue(), "distribution")]

def upload_images(storage, imagenes, user_id):
    image_public_url = dict()
    image_names = []

    for data_imagen, funcion in imagenes:
        image_name = f"{user_id}/{uuid.uuid4()}.png"

        storage.from_("imagenes-generadas").upload(
            path=image_name,
            file=data_imagen,
            file_options={"content-type": "image/png"}
        )

        image_public_url[funcion] = f"http://127.0.0.1:54321/storage/v1/object/public/imagenes-generadas/{image_name}"
        image_names.append(image_name)

    return image_public_url, image_names


def save_image_info_to_db(supabase, user_id, dataset_id, image_public_url, image_names):
    for funcion, public_url in image_public_url.items():
        image_name = image_names.pop(0)
        
        supabase.table("imagenes-generadas").insert({
            "user_id": user_id,
            "dataset_id": dataset_id,
            "function_name": funcion,
            "public_url": public_url,
            "image_path": image_name,
        }).execute()

def save_eda_results_to_db(supabase, user_id, dataset_id, corpus_stats, class_stats):
    corpus_stats = corpus_stats.to_dict(orient="records")[0]
    corpus_stats["dataset_id"] = dataset_id
    corpus_stats["user_id"] = user_id
    supabase.table("corpus_stats").insert(corpus_stats).execute()

    class_stats = class_stats.to_dict(orient="records")
    for class_stat in class_stats:
        class_stat["dataset_id"] = dataset_id
        class_stat["user_id"] = user_id
        supabase.table("class_stats").insert(class_stat).execute()

@app.post("/eda")
async def main(
    data: EdaRequest,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    supabase = get_supabase_client(x_user_token)
    storage = get_storage_client(x_user_token)

    response = (
            supabase
            .table("datasets")
            .select("*")
            .eq("dataset_name", data.dataset)
            .eq("user_id", x_user_id)
            .execute()
        )

    if not response.data:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")

    dataset_info = response.data[0]
    dataset_id = dataset_info["dataset_id"]

    dataset_path = os.path.join(
        DATASET_STORAGE_PATH, "users", x_user_id, f"{dataset_id}.json"
    )
    
    dataset = parse_dataset_json(dataset_path)

    # Análisis de las filas del dataset
    dataset = get_dataset_row_analysis(dataset)
    print(dataset.head())

    # Estadísticas generales del corpus
    DF_CORPUS = get_estadisticas_corpus(dataset)
    print(DF_CORPUS.head())

    # Crear un DataFrame para almacenar las estadísticas por clase
    DF_CLASES = get_estadisticas_clases(dataset)
    print(DF_CLASES.head())

    # Generar las imágenes en buffer
    imagenes = []
    imagenes += generate_sentence_distribution_plot(dataset)
    imagenes += generate_class_distribution_plot(DF_CLASES)
    imagenes += generate_distribution_plot(dataset)

    # Subir las imágenes al bucket
    image_public_url, image_names = upload_images(storage, imagenes, x_user_id)

    # Guardar informacion de imagenes en tabla
    save_image_info_to_db(supabase, x_user_id, dataset_id, image_public_url, image_names)

    # Guardar resultados de EDA en tabla
    save_eda_results_to_db(supabase, x_user_id, dataset_id, DF_CORPUS, DF_CLASES)

    return {
        "corpus_stats": DF_CORPUS.to_dict(orient="records")[0],
        "class_stats": DF_CLASES.to_dict(orient="records"),
        "images": image_public_url,
    }

@app.post("/getResults")
async def get_results(
    data: EdaRequest,
    x_user_id: str = Header(...),
    x_user_token: str = Header(...),
):
    supabase = get_supabase_client(x_user_token)

    response = (
        supabase
        .table("datasets")
        .select("*")
        .eq("dataset_name", data.dataset)
        .eq("user_id", x_user_id)
        .execute()
    )

    if not response.data:
        raise HTTPException(status_code=404, detail="Dataset no encontrado")

    dataset_info = response.data[0]
    dataset_id = dataset_info["dataset_id"]

    response = (
        supabase
        .table("imagenes-generadas")
        .select("*")
        .eq("user_id", x_user_id)
        .eq("dataset_id", dataset_id)
        .in_("function_name", ["boxplot", "catplot", "class_distribution", "distribution"])
        .execute()
    )

    if not response.data:
        raise HTTPException(status_code=404, detail="No se encontraron imágenes generadas para este dataset")

    images = {img["function_name"]: img["public_url"] for img in response.data}

    response = (
        supabase
        .table("class_stats")
        .select("clase, texto, vocabulary, words, word_avg, sentences, sentence_avg, sentence_min, sentence_max")
        .eq("user_id", x_user_id)
        .eq("dataset_id", dataset_id)
        .execute()
    )

    class_stats = response.data

    response = (
        supabase
        .table("corpus_stats")
        .select("texto, vocabulary, words, word_avg, sentences, sentence_avg, sentence_min, sentence_max")
        .eq("user_id", x_user_id)
        .eq("dataset_id", dataset_id)
        .execute()
    )

    corpus_stats = response.data[0] if response.data else None

    return {
        "images": images,
        "class_stats": class_stats,
        "corpus_stats": corpus_stats
    }