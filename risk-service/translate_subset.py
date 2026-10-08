"""Traduce una muestra estratificada EN->ES para el entrenamiento bilingüe.

Modelo: Helsinki-NLP/opus-mt-en-es (liviano, GPU). Genera:
  suicide_es_train.parquet  (text,label) -> para mezclar con --extra-csv
  suicide_es_test.parquet   (text,label) -> reservado para medir F1 ES (--es-test)

Uso (en Docker GPU):
  python translate_subset.py --per-class 10000 --es-test-n 2000 --out ./es_data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Traduce muestra EN->ES (Helsinki)")
    p.add_argument("--dataset", default="vibhorag101/suicide_prediction_dataset_phr")
    p.add_argument("--model", default="Helsinki-NLP/opus-mt-en-es")
    p.add_argument("--per-class", type=int, default=10000,
                   help="textos por clase para train ES")
    p.add_argument("--es-test-n", type=int, default=2000,
                   help="textos ES reservados solo para test")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default="./es_data")
    p.add_argument("--max-length", type=int, default=256,
                   help="trunca origen para acelerar (suficiente para ideación)")
    return p.parse_args()


def binarize(v) -> int:
    return 1 if str(v).strip().lower() in ("1", "suicidal", "suicide", "label_1") else 0


def main() -> None:
    args = parse_args()
    from datasets import load_dataset
    from transformers import pipeline

    print(f"[tr] cargando {args.dataset} ...", flush=True)
    ds = load_dataset(args.dataset)
    split = "train" if "train" in ds else list(ds.keys())[0]
    df = ds[split].to_pandas()
    cols = {c.lower(): c for c in df.columns}
    text_col = cols.get("text") or cols.get("sentence")
    label_col = cols.get("label") or cols.get("status") or cols.get("class")
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["label"] = df["label"].map(binarize)
    df["text"] = df["text"].astype(str).str.slice(0, 1000)

    # Muestra estratificada total = 2*per_class + es_test_n
    total = 2 * args.per_class + args.es_test_n
    _, sample = train_test_split(df, train_size=min(total, len(df)),
                                 stratify=df["label"], random_state=42)
    pos = sample[sample.label == 1]
    neg = sample[sample.label == 0]

    test_es = pd.concat([
        pos.sample(n=min(args.es_test_n // 2, len(pos)), random_state=1),
        neg.sample(n=min(args.es_test_n // 2, len(neg)), random_state=1),
    ])
    rest = sample.drop(test_es.index)
    train_es = pd.concat([
        rest[rest.label == 1].sample(n=min(args.per_class, (rest.label == 1).sum()), random_state=2),
        rest[rest.label == 0].sample(n=min(args.per_class, (rest.label == 0).sum()), random_state=2),
    ])
    print(f"[tr] train ES={len(train_es)} test ES={len(test_es)}", flush=True)

    device = 0 if torch.cuda.is_available() else -1
    print(f"[tr] Helsinki en {'GPU' if device == 0 else 'CPU'} ...", flush=True)
    model_kwargs: dict = {}
    if device == 0:
        model_kwargs["torch_dtype"] = torch.float16  # 2-3x en RTX 3060, mitad VRAM
    pipe = pipeline("translation_en_to_es", model=args.model, device=device,
                    truncation=True, max_length=args.max_length,
                    model_kwargs=model_kwargs or None)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    try:
        from tqdm import tqdm
        _pbar = tqdm
    except ImportError:
        _pbar = None

    for name, frame in (("suicide_es_train.parquet", train_es),
                        ("suicide_es_test.parquet", test_es)):
        texts = frame["text"].tolist()
        # Batches manuales con progreso visible (el pipeline no muestra barra)
        bs = args.batch_size
        out_texts: list[str] = []
        rng = range(0, len(texts), bs)
        if _pbar is not None:
            rng = _pbar(rng, desc=f"[tr] {name}")
        for s in rng:
            batch = texts[s:s + bs]
            res = pipe(batch, batch_size=len(batch))
            out_texts.extend(r["translation_text"] for r in res)
        frame = frame.copy()
        frame["text"] = out_texts
        frame[["text", "label"]].to_parquet(out / name, index=False)
        print(f"[tr] OK -> {out / name} ({len(frame)})", flush=True)


if __name__ == "__main__":
    main()
