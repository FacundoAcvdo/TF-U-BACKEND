"""Entrenamiento offline de la cabeza de riesgo suicida (opción A).

Pipeline:
  dataset HF (EN) + semillas ES -> embeddings E5-large -> LogReg calibrada -> risk_model.pkl

Uso rápido (prototipo, CPU/GPU):
  pip install -r requirements-train.txt
  python train_risk_head.py --quick

Uso completo (232k):
  python train_risk_head.py --dataset vibhorag101/suicide_prediction_dataset_phr

Modelo definitivo bilingüe (EN + 20k ES traducidos):
  python translate_subset.py --per-class 10000 --out ./es_data
  python train_risk_head.py --dataset vibhorag101/suicide_prediction_dataset_phr \
      --extra-csv es_data/suicide_es_train.parquet --es-test es_data/suicide_es_test.parquet

Salidas en ./model_artifacts/:
  risk_model.pkl, thresholds.json, metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import logging
from logging_utils import setup_service_logger
logger = setup_service_logger("risk-service")


ARTIFACTS_DIR = Path(__file__).parent / "model_artifacts"


THRESHOLDS = {
    "alto": 0.85,
    "potencial": 0.65,
    "medio": 0.40,
    "bajo": 0.15,

}


SEEDS_ES_SUICIDA = [
    "quiero morir, ya no puedo más",
    "estoy pensando en quitarme la vida",
    "no quiero seguir viviendo así",
    "a veces pienso que sería más fácil no despertar",
    "siento que soy una carga para todos",
    "ya no le encuentro sentido a nada de lo que hago",
    "me quiero ir, no doy más",
    "pienso en desaparecer para siempre",
    "no veo salida, quiero terminar con todo",
    "me gustaría no existir más",
]
SEEDS_ES_NO_SUICIDA = [
    "hoy estoy cansado pero voy a descansar un rato",
    "tengo mucho trabajo esta semana",
    "me siento triste por la noticia pero estoy con mi familia",
    "quiero seguir adelante con mis proyectos",
    "la terapia me está ayudando mucho",
    "pasé una buena tarde con amigos",
    "estoy preocupado por el examen pero voy a estudiar",
    "qué lindo día para salir a caminar",
    "extraño a mi familia pero hablamos seguido",
    "tengo esperanza de que las cosas mejoren",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena cabeza de riesgo E5-large + LogReg")
    p.add_argument("--dataset", default="ourafla/Mental-Health_Text-Classification_Dataset",
                   help="ID HF del dataset")
    p.add_argument("--embedding-model", default="intfloat/multilingual-e5-large")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=0,
                   help="0 = todo; ej 5000 para prueba rápida")
    p.add_argument("--quick", action="store_true",
                   help="atajo: max-samples=5000")
    p.add_argument("--no-seeds-es", action="store_true",
                   help="no agregar semillas ES")
    p.add_argument("--extra-csv", action="append", default=[],
                   help="CSV/parquet extra ES (text,label). Repetible. Ej: es_data/suicide_es_train.parquet")
    p.add_argument("--es-test", default="",
                   help="parquet ES reservado para medir F1 en español. Ej: es_data/suicide_es_test.parquet")
    p.add_argument("--out", default=str(ARTIFACTS_DIR))
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--no-fp16", action="store_true")
    return p.parse_args()


def resolve_device(arg: str) -> str:
    if arg == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if arg == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_texts_labels(dataset_id: str, max_samples: int) -> tuple[list[str], list[int]]:
    from datasets import load_dataset

    pass


    data_files = None
    if dataset_id == "ourafla/Mental-Health_Text-Classification_Dataset":
        data_files = {"train": "mental_heath_unbanlanced.csv"}
    ds = load_dataset(dataset_id, data_files=data_files)

    split = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()
    pass


    cols_lower = {c.lower(): c for c in df.columns}
    text_col = cols_lower.get("text") or cols_lower.get("texto") or cols_lower.get("sentence")
    label_col = cols_lower.get("status") or cols_lower.get("label") or cols_lower.get("class") or cols_lower.get("clase")
    if text_col is None or label_col is None:
        raise ValueError(f"Columnas no reconocidas: {list(df.columns)}")

    texts = df[text_col].astype(str).tolist()
    raw = df[label_col].tolist()

    labels = []
    for v in raw:
        s = str(v).strip().lower()
        labels.append(1 if s in ("1", "suicidal", "suicide", "label_1") else 0)

    if max_samples and len(texts) > max_samples:

        xs, _, ys, _ = train_test_split(
            texts, labels, train_size=max_samples,
            stratify=labels, random_state=42,
        )
        texts, labels = xs, ys
        pass

    pos = sum(labels)
    pass
    return texts, labels


def embed_texts(texts: list[str], model_name: str, batch_size: int, device: str, fp16: bool) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    pass
    kwargs: dict = {}
    if fp16 and device == "cuda":
        kwargs["torch_dtype"] = torch.float16
    model = SentenceTransformer(model_name, device=device, model_kwargs=kwargs or None)

    prefixed = [t if t.startswith("passage: ") else f"passage: {t}" for t in texts]
    emb = model.encode(
        prefixed, batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def main() -> None:
    args = parse_args()
    if args.quick:
        args.max_samples = 5000
    device = resolve_device(args.device)
    fp16 = not args.no_fp16

    texts, labels = load_texts_labels(args.dataset, args.max_samples)

    if not args.no_seeds_es:
        texts = texts + SEEDS_ES_SUICIDA + SEEDS_ES_NO_SUICIDA
        labels = labels + [1] * len(SEEDS_ES_SUICIDA) + [0] * len(SEEDS_ES_NO_SUICIDA)
        pass


    for extra in args.extra_csv or []:
        import pandas as pd
        dfx = pd.read_parquet(extra) if str(extra).endswith(".parquet") else pd.read_csv(extra)
        cols = {c.lower(): c for c in dfx.columns}
        tc, lc = cols.get("text", cols.get("texto")), cols.get("label", cols.get("clase"))
        if tc is None or lc is None:
            raise ValueError(f"Columnas no reconocidas en {extra}: {list(dfx.columns)}")
        xt = dfx[tc].astype(str).tolist()
        yl = [1 if str(v).strip().lower() in ("1", "suicidal", "suicide") else 0
              for v in dfx[lc].tolist()]
        texts += xt
        labels += yl
        pass

    X = embed_texts(texts, args.embedding_model, args.batch_size, device, fp16)
    pass
    y = np.asarray(labels, dtype=int)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    base = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    pass
    clf.fit(Xtr, ytr)

    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(yte, pred)),
        "f1": float(f1_score(yte, pred, zero_division=0)),
        "precision": float(precision_score(yte, pred, zero_division=0)),
        "recall": float(recall_score(yte, pred, zero_division=0)),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "embedding_model": args.embedding_model,
        "embedding_dim": int(X.shape[1]),
        "dataset": args.dataset,
        "extra_csv": args.extra_csv,
        "date": datetime.now(timezone.utc).isoformat(),
    }


    if args.es_test:
        import pandas as pd
        dfe = (pd.read_parquet(args.es_test) if str(args.es_test).endswith(".parquet")
               else pd.read_csv(args.es_test))
        cols = {c.lower(): c for c in dfe.columns}
        tc = cols.get("text", cols.get("texto"))
        lc = cols.get("label", cols.get("clase"))
        xte_es = dfe[tc].astype(str).tolist()
        yte_es = np.array([1 if str(v).strip().lower() in ("1", "suicidal", "suicide") else 0
                           for v in dfe[lc].tolist()], dtype=int)
        Xe = embed_texts(xte_es, args.embedding_model, args.batch_size, device, fp16)
        pe = clf.predict_proba(Xe)[:, 1]
        pr = (pe >= 0.5).astype(int)
        metrics["es_test"] = {
            "n": int(len(yte_es)),
            "accuracy": float(accuracy_score(yte_es, pr)),
            "f1": float(f1_score(yte_es, pr, zero_division=0)),
            "precision": float(precision_score(yte_es, pr, zero_division=0)),
            "recall": float(recall_score(yte_es, pr, zero_division=0)),
        }
        pass
    pass

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"clf": clf, "thresholds": THRESHOLDS,
         "embedding_model": args.embedding_model,
         "embedding_dim": int(X.shape[1]),
         "trained_at": metrics["date"]},
        out / "risk_model.pkl",
    )
    (out / "thresholds.json").write_text(json.dumps(THRESHOLDS, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pass


if __name__ == "__main__":
    main()
