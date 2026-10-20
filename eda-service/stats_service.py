import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

def get_dataset_row_analysis(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset['characters'] = dataset['texto'].astype(str).apply(len)
    dataset['words'] = dataset['texto'].apply(lambda x: len(x.split(' ')))
    dataset['sentences'] = dataset['texto'].str.count(r'[\w][.!?]').clip(lower=1)
    dataset['Avg-charperword'] = (dataset['characters']/dataset['words'])
    dataset['Avg-charpersentence'] = (dataset['characters']/dataset['sentences'])

    return dataset

def get_estadisticas_corpus(dataset: pd.DataFrame) -> pd.DataFrame:
    vect = CountVectorizer().fit(dataset['texto'])
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
