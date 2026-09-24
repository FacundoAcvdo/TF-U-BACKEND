import pandas as pd


def build_dataframe_from_chroma(documents: list, metadatas: list) -> pd.DataFrame:
    records = []
    for i, texto in enumerate(documents or []):
        meta = metadatas[i] if i < len(metadatas or []) else {}
        records.append({
            "texto": texto,
            "clase": (meta or {}).get("clase"),
        })

    df = pd.DataFrame(records)

    fechas = [(m or {}).get("fecha") for m in (metadatas or [])]
    if any(f is not None for f in fechas):
        df["fecha"] = fechas

    return df