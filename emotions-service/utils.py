import json
import pandas as pd

def parse_dataset_json(ruta: str) -> pd.DataFrame:
    with open(ruta, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    records = []
    for doc in json_data.get("documents", []):
        row = {"texto": doc.get("texto"), "clase": doc.get("clase")}
        records.append(row)

    df = pd.DataFrame(records)

    fechas = [doc.get("fecha") for doc in json_data.get("documents", [])]
    if any(f is not None for f in fechas):
        df["fecha"] = fechas

    return df
