import inspect
import io
import os
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone

import numpy as np
import torch
from fastapi import HTTPException
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
from sklearn.utils.class_weight import compute_class_weight
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from core import DEVICE, EMBEDDING_MODEL_NAME
from schemas import TrainingRequest

MAX_CLASES = int(os.environ.get("MAX_CLASES", 7))


class _WeightedTrainer(Trainer):
    """Trainer con CrossEntropy ponderada por clase (desbalance 65/35)."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        etiquetas = inputs.get("labels")
        resto = {k: v for k, v in inputs.items() if k != "labels"}
        salidas = model(**resto)
        logits = salidas.logits if hasattr(salidas, "logits") else salidas[0]
        peso = None
        if self._class_weights is not None:
            peso = self._class_weights.to(logits.device)
        import torch.nn.functional as _F
        perdida = _F.cross_entropy(logits, etiquetas, weight=peso)
        return (perdida, salidas) if return_outputs else perdida

def _flotante(v, default: float, lo: float, hi: float) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, f))


def _entero(v, default: int, lo: int, hi: int) -> int:
    try:
        i = int(float(v))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, i))


def instanciar_clasificador_sklearn(nombre: str, hiper: dict | None = None):
    hiper = hiper or {}
    if nombre == "Support Vector Machine":
        c = _flotante(hiper.get("C"), 1.0, 0.01, 100.0)
        mi = _entero(hiper.get("max_iter"), 3000, 100, 10000)
        return (LinearSVC(random_state=42, max_iter=mi, class_weight="balanced", C=c),
                {"C": c, "max_iter": mi})
    if nombre == "Random Forest":
        ne = _entero(hiper.get("n_estimators"), 100, 10, 1000)
        md_raw = hiper.get("max_depth", None)
        if md_raw in (None, "", "auto", "null", "None"):
            md = None
        else:
            md = _entero(md_raw, 0, 2, 100)
            md = None if md == 0 else md
        msl = _entero(hiper.get("min_samples_leaf"), 1, 1, 10)
        return (RandomForestClassifier(
            n_estimators=ne, max_depth=md, min_samples_leaf=msl,
            random_state=42, class_weight="balanced", n_jobs=-1,
        ), {"n_estimators": ne, "max_depth": md, "min_samples_leaf": msl})
    if nombre == "Naive Bayes":
        vs = _flotante(hiper.get("var_smoothing"), 1e-9, 1e-12, 1e-3)
        return GaussianNB(var_smoothing=vs), {"var_smoothing": vs}
    c = _flotante(hiper.get("C"), 1.0, 0.01, 100.0)
    mi = _entero(hiper.get("max_iter"), 1000, 100, 5000)
    return (LogisticRegression(max_iter=mi, random_state=42, class_weight="balanced", C=c),
            {"C": c, "max_iter": mi})


def procesar_etiquetas_automatico(valores_raw: list[str]):
    if not valores_raw:
        raise HTTPException(status_code=400, detail="No se encontraron metadatos en ChromaDB.")
    faltantes = sum(1 for v in valores_raw if v == "")
    if faltantes > 0:
        raise HTTPException(
            status_code=422,
            detail=f"Este dataset tiene {faltantes} texto(s) sin clase asignada. No se puede clasificar hasta que todos estén etiquetados.",
        )
    le = LabelEncoder()
    y = le.fit_transform(valores_raw)
    clases_unicas = list(le.classes_)
    if len(clases_unicas) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"El dataset contiene una única clase ('{clases_unicas[0]}'). Se requieren al menos dos para entrenar.",
        )

    if len(clases_unicas) > MAX_CLASES:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Este dataset tiene {len(clases_unicas)} clases. "
                f"El microservicio admite como máximo {MAX_CLASES} clases."
            ),
        )
    label_mapping = {clase: int(idx) for idx, clase in enumerate(clases_unicas)}
    return y, label_mapping, clases_unicas


def validar_tamano_muestras(y_labels, data: TrainingRequest):
    conteos = np.bincount(y_labels)

    if len(conteos) == 0:
        raise HTTPException(
            status_code=400,
            detail="No hay muestras etiquetadas para entrenar.",
        )

    clase_minoritaria = int(conteos.min())
    cantidad_clases = len(conteos)

    if data.modo == "validacion_cruzada":
        k_folds = data.k if data.k in [5, 10] else 5

        if clase_minoritaria < k_folds:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"La clase minoritaria tiene solo {clase_minoritaria} muestra(s), "
                    f"pero se pidieron {k_folds} pliegues. Cada clase necesita al menos "
                    f"{k_folds} muestras para validación cruzada."
                ),
            )
        return

    porcentaje_prueba = data.porcentaje_prueba or 50
    test_size = max(0.1, min(0.9, porcentaje_prueba / 100.0))

    if clase_minoritaria < 2:
        raise HTTPException(
            status_code=422,
            detail=(
                f"La clase minoritaria tiene solo {clase_minoritaria} muestra(s). "
                "Se necesitan al menos 2 por clase para entrenamiento y prueba estratificados."
            ),
        )

    total_muestras = len(y_labels)
    cantidad_test = int(np.ceil(total_muestras * test_size))
    cantidad_train = total_muestras - cantidad_test

    if cantidad_test < cantidad_clases:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Hay {cantidad_clases} clases, pero el porcentaje de prueba genera "
                f"solo {cantidad_test} muestra(s) para test. Necesitás al menos "
                f"{cantidad_clases} muestras de prueba, una por cada clase."
            ),
        )

    if cantidad_train < cantidad_clases:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Hay {cantidad_clases} clases, pero el porcentaje de entrenamiento "
                f"deja solo {cantidad_train} muestra(s) para train. Necesitás al menos "
                f"{cantidad_clases} muestras de entrenamiento, una por cada clase."
            ),
        )


def calcular_metricas(y_true, y_pred, cantidad_clases: int):
    labels = list(range(cantidad_clases))
    matriz = confusion_matrix(y_true, y_pred, labels=labels)

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    if cantidad_clases == 2:
        tn, fp, fn, tp = matriz.ravel()
        return (
            int(tp), int(fp), int(tn), int(fn),
            acc, f1, prec, rec, matriz.tolist(),
        )

    total = int(matriz.sum())
    tp = int(np.trace(matriz))
    fp = int(np.sum(matriz.sum(axis=0) - np.diag(matriz)))
    fn = int(np.sum(matriz.sum(axis=1) - np.diag(matriz)))

    tn_por_clase = []
    for i in range(cantidad_clases):
        tp_i = int(matriz[i, i])
        fn_i = int(matriz[i, :].sum() - tp_i)
        fp_i = int(matriz[:, i].sum() - tp_i)
        tn_por_clase.append(total - tp_i - fn_i - fp_i)

    tn = int(sum(tn_por_clase))

    return (
        tp, fp, tn, fn,
        acc, f1, prec, rec, matriz.tolist(),
    )


def entrenar_sklearn(x_vectors, y_labels, label_mapping, clases_nombres, data: TrainingRequest):
    total_muestras = len(y_labels)
    cantidad_clases = len(clases_nombres)
    clf, hiper_usados = instanciar_clasificador_sklearn(data.modelo, data.hiperparametros)

    if data.modo == "validacion_cruzada":
        k_folds = data.k if data.k in [5, 10] else 5
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

        y_pred = cross_val_predict(
            clf,
            x_vectors,
            y_labels,
            cv=skf,
            n_jobs=-1,
        )
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

    (
        tp, fp, tn, fn,
        acc, f1, prec, rec,
        matriz,
    ) = calcular_metricas(y_eval, y_pred, cantidad_clases)

    checkpoint_buffer = io.BytesIO()
    checkpoint = {
        "clasificador": clf,
        "encoder_base": EMBEDDING_MODEL_NAME,
        "label_mapping": label_mapping,
        "clases_nombres": clases_nombres,
        "matriz_confusion": matriz,
        "num_clases": cantidad_clases,
        "accuracy": acc,
        "f1_score": f1,
        "precision": prec,
        "recall": rec,
        "hiperparametros": hiper_usados,
    }
    torch.save(checkpoint, checkpoint_buffer)
    checkpoint_buffer.seek(0)

    fecha = datetime.now(timezone.utc).isoformat()
    hiper_lineas = "\n".join(f"- `{k}`: **{v}**" for k, v in hiper_usados.items())
    mapping_lines = "\n".join(
        f"- `{clase}` → `{idx}`"
        for clase, idx in label_mapping.items()
    )
    matriz_tabla = (
        "| Real / Predicha | "
        + " | ".join(str(i) for i in range(cantidad_clases))
        + " |\n"
        + "|" + "---|" * (cantidad_clases + 1)
        + "\n"
        + "\n".join(
            f"| {i} | " + " | ".join(map(str, fila)) + " |"
            for i, fila in enumerate(matriz)
        )
    )

    readme_texto = f"""# Modelo de clasificación de menciones ({data.modelo})

- Dataset: **{data.dataset}**
- Fecha: **{fecha}**
- Cantidad de clases: **{cantidad_clases}**

## Configuración y métricas

- Modo: **{data.modo}**
- Total de muestras: **{total_muestras}**
- Muestras de entrenamiento: **{muestras_train}**
- Muestras de evaluación: **{muestras_test}**
- Accuracy: **{acc:.4f}**
- F1-score weighted: **{f1:.4f}**
- Precision weighted: **{prec:.4f}**
- Recall weighted: **{rec:.4f}**

## Hiperparámetros

{hiper_lineas}

## Mapeo de clases

{mapping_lines}

## Matriz de confusión

Las filas son las clases reales y las columnas las clases predichas.

{matriz_tabla}

## Uso en Python

```python
import torch
from sentence_transformers import SentenceTransformer

checkpoint = torch.load("modelo.pt", weights_only=False)
clf = checkpoint["clasificador"]
encoder = SentenceTransformer("{EMBEDDING_MODEL_NAME}")

textos = ["ejemplo de texto a clasificar"]
embeddings = encoder.encode(textos, normalize_embeddings=True)
prediccion_id = clf.predict(embeddings)

for texto, clase_id in zip(textos, prediccion_id):
    prediccion = (texto, checkpoint["clases_nombres"][int(clase_id)])

# Matriz guardada en el checkpoint:
matriz = checkpoint["matriz_confusion"]
```

## Dependencias

```bash
pip install torch scikit-learn sentence-transformers
```
"""

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("modelo.pt", checkpoint_buffer.read())
        zf.writestr("README.md", readme_texto)
    zip_buffer.seek(0)

    return (
        tp, fp, tn, fn,
        acc, f1, prec, rec,
        total_muestras, muestras_train, muestras_test,
        zip_buffer, matriz,
    )


class TextDataset(torch.utils.data.Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = int(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


MODELOS_BERT = {
    "es": "dccuchile/distilbert-base-spanish-uncased",
    "en": "distilbert-base-uncased",
    "es_base": "dccuchile/bert-base-spanish-wwm-uncased",
    "en_base": "google-bert/bert-base-uncased",
    "en_mental": "mental/mental-roberta-base",
}

_TOKENIZERS: dict = {}

_MARCAS_ES = (" que ", " de ", " la ", " el ", " en ", " una ", " para ",
              " con ", " los ", " del ", " no ", " por ", " este ", " esta ")
_MARCAS_EN = (" the ", " and ", " to ", " of ", " you ", " that ", " with ",
              " for ", " this ", " are ", " was ", " have ", " not ")


def detectar_idioma(textos: list[str], muestra: int = 100) -> str:
    pts_es = pts_en = 0
    for t in (textos or [])[:max(1, muestra)]:
        s = f" {(t or '').lower()} "
        if "ñ" in s or "á" in s or "é" in s or "í" in s or "ó" in s or "ú" in s or "¿" in s or "¡" in s:
            pts_es += 2
        pts_es += sum(1 for m in _MARCAS_ES if m in s)
        pts_en += sum(1 for m in _MARCAS_EN if m in s)
    return "en" if pts_en > pts_es else "es"


def get_tokenizer(model_name: str):
    tok = _TOKENIZERS.get(model_name)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(model_name)
        _TOKENIZERS[model_name] = tok
    return tok


def _ajustes_gpu() -> tuple[int, int, bool]:
    try:
        mayor, _ = torch.cuda.get_device_capability(0)
    except Exception:
        return 16, 32, True
    fp16 = mayor >= 7
    free_gb = 99.0
    try:
        free_b, _total_b = torch.cuda.mem_get_info(0)
        free_gb = free_b / (1024 ** 3)
    except Exception:
        pass
    if free_gb >= 6:
        batch = 32
    elif free_gb >= 3:
        batch = 8
    elif free_gb >= 1.5:
        batch = 4
    else:
        raise HTTPException(
            status_code=503,
            detail="Servicio ocupado, intenta más tarde.",
        )
    return batch, batch * 2, fp16


def entrenar_bert(textos, y_labels, label_mapping, clases_nombres, data: TrainingRequest):
    hiper = getattr(data, "hiperparametros", None) or {}
    pedido = str(hiper.get("bert_modelo", "auto") or "auto").lower()
    if pedido not in ("auto", "es_base", "en_mental"):
        pedido = "auto"
    if pedido in ("es_base", "en_mental"):
        idioma = "en" if pedido == "en_mental" else "es"
        model_name = MODELOS_BERT[pedido]
    else:
        idioma = detectar_idioma(textos)
        model_name = MODELOS_BERT["en_mental"] if idioma == "en" else MODELOS_BERT["es_base"]
    modo_rapido = bool(hiper.get("bert_rapido", False))
    cantidad_clases = len(clases_nombres)
    es_cv = (data.modo == "validacion_cruzada")
    k_folds = data.k if data.k in [5, 10] else 5
    y_list = list(y_labels)

    tokenizer = get_tokenizer(model_name)
    model = None
    if not es_cv:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=cantidad_clases,
            id2label={int(v): k for k, v in label_mapping.items()},
            label2id={k: int(v) for k, v in label_mapping.items()},
        )
        model.to(DEVICE)
        if modo_rapido:
            for n, p in model.named_parameters():
                if "classifier" not in n and "pre_classifier" not in n and "score" not in n:
                    p.requires_grad = False

    largos = sorted(len(t or "") for t in textos) if textos else [0]
    p90 = largos[min(len(largos) - 1, int(len(largos) * 0.9))]
    max_len = 128 if p90 < 250 else 256
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    if modo_rapido:
        epocas = 10
    elif len(textos) < 200:
        epocas = 1
    else:
        epocas = 4
    temp_dir = tempfile.mkdtemp()
    bt, be, usar_fp16 = _ajustes_gpu() if DEVICE == "cuda" else (8, 16, False)
    if DEVICE == "cuda":
        try:
            pass
        except Exception:
            pass

    def _nuevo_modelo():
        m = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=cantidad_clases,
            id2label={int(v): k for k, v in label_mapping.items()},
            label2id={k: int(v) for k, v in label_mapping.items()},
        )
        m.to(DEVICE)
        if modo_rapido:
            for n, p in m.named_parameters():
                if "classifier" not in n and "pre_classifier" not in n and "score" not in n:
                    p.requires_grad = False
        return m

    def _args_en(tmp: str):
        args_deseados = {
            "output_dir": tmp,
            "num_train_epochs": epocas,
            "per_device_train_batch_size": bt,
            "per_device_eval_batch_size": be,
            "eval_strategy": "epoch",
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "save_total_limit": 1,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
            "learning_rate": 2e-5,
            "warmup_ratio": 0.1,
            "use_cpu": DEVICE == "cpu",
            "fp16": usar_fp16,
            "weight_decay": 0.01,
            "group_by_length": True,
            "dataloader_num_workers": 2 if DEVICE == "cuda" else 0,
            "dataloader_pin_memory": DEVICE == "cuda",
            "report_to": "none",
        }
        soportados = set(inspect.signature(TrainingArguments.__init__).parameters)
        args_filtrados = {k: v for k, v in args_deseados.items() if k in soportados}
        if "eval_strategy" in args_filtrados and "evaluation_strategy" in args_filtrados:
            args_filtrados.pop("evaluation_strategy")
        return TrainingArguments(**args_filtrados)

    historial_loss: list[str] = []

    def _entrenar_y_predecir(modelo, tr_enc, y_tr, te_enc, guardar_en=None):
        tmp = tempfile.mkdtemp()
        try:
            clases_ord = sorted(set(int(v) for v in y_tr))
            conteo = compute_class_weight(
                "balanced",
                classes=np.asarray(clases_ord),
                y=np.asarray(y_tr, dtype=int),
            )
            por_indice = {c: w for c, w in zip(clases_ord, conteo)}
            pesos = torch.tensor(
                [por_indice.get(c, 1.0) for c in range(cantidad_clases)],
                dtype=torch.float32,
            )
            trainer_kwargs = {
                "model": modelo,
                "args": _args_en(tmp),
                "train_dataset": TextDataset(tr_enc, y_tr),
                "eval_dataset": TextDataset(te_enc, [0] * len(te_enc["input_ids"])),
                "data_collator": collator,
            }
            if "processing_class" in set(inspect.signature(Trainer.__init__).parameters):
                trainer_kwargs["processing_class"] = tokenizer
            else:
                trainer_kwargs["tokenizer"] = tokenizer
            tr = _WeightedTrainer(class_weights=pesos, **trainer_kwargs)
            tr.train()
            perdidas_ent = [h.get("loss") for h in tr.state.log_history if h.get("loss") is not None]
            perdidas_val = [h.get("eval_loss") for h in tr.state.log_history if h.get("eval_loss") is not None]
            historial_loss.append(
                f"- corrida {len(historial_loss) + 1}: "
                f"loss_entrenamiento={perdidas_ent[-1]:.4f} "
                f"loss_validacion={perdidas_val[-1]:.4f}"
                if perdidas_ent and perdidas_val else
                f"- corrida {len(historial_loss) + 1}: sin registros de pérdida"
            )
            pred = tr.predict(TextDataset(te_enc, [0] * len(te_enc["input_ids"])))
            yph = np.asarray(pred.predictions).argmax(axis=-1)
            if guardar_en:
                tr.save_model(guardar_en)
                tokenizer.save_pretrained(guardar_en)
            return yph, tr
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    try:
        model_dir = os.path.join(temp_dir, "modelo_bert")
        if es_cv:
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            folds = list(skf.split(np.zeros(len(y_list)), np.asarray(y_list)))
            y_pred_full = np.zeros(len(y_list), dtype=int)
            for j, (tr_i, te_i) in enumerate(folds):
                m = _nuevo_modelo()
                try:
                    yph, _ = _entrenar_y_predecir(
                        m,
                        tokenizer([textos[i] for i in tr_i], truncation=True, max_length=max_len),
                        [y_list[i] for i in tr_i],
                        tokenizer([textos[i] for i in te_i], truncation=True, max_length=max_len),
                        guardar_en=model_dir if j == len(folds) - 1 else None,
                    )
                    y_pred_full[np.asarray(te_i)] = np.asarray(yph).ravel()
                finally:
                    try:
                        del m
                    except Exception:
                        pass
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
            y_pred = y_pred_full
            y_eval = np.asarray(y_list)
            n_tr = n_te = len(y_list)
            detalle_modo = f"validación cruzada (k={k_folds}, se guarda último pliegue)"
        else:
            test_size = max(0.1, min(0.9, (data.porcentaje_prueba or 20) / 100.0))
            x_train, x_test, y_train, y_test = train_test_split(
                textos,
                y_list,
                test_size=test_size,
                stratify=y_list,
                random_state=42,
            )
            train_encodings = tokenizer(x_train, truncation=True, max_length=max_len)
            test_encodings = tokenizer(x_test, truncation=True, max_length=max_len)
            y_pred, trainer = _entrenar_y_predecir(
                model, train_encodings, list(y_train), test_encodings,
                guardar_en=model_dir,
            )
            y_eval = np.asarray(y_test)
            n_tr, n_te = len(y_train), len(y_test)
            detalle_modo = "porcentajes"

        (
            tp, fp, tn, fn,
            acc, f1, prec, rec,
            matriz,
        ) = calcular_metricas(y_eval, y_pred, cantidad_clases)

        fecha = datetime.now(timezone.utc).isoformat()
        lineas_loss = "\n".join(historial_loss) if historial_loss else "- Sin registros."
        mapping_lines = "\n".join(
            f"- `{clase}` → `{idx}`"
            for clase, idx in label_mapping.items()
        )
        matriz_tabla = (
            "| Real / Predicha | "
            + " | ".join(str(i) for i in range(cantidad_clases))
            + " |\n"
            + "|" + "---|" * (cantidad_clases + 1)
            + "\n"
            + "\n".join(
                f"| {i} | " + " | ".join(map(str, fila)) + " |"
                for i, fila in enumerate(matriz)
            )
        )

        readme_texto = f"""# Modelo BERT (Fine-Tuning)

- Dataset: **{data.dataset}**
- Arquitectura base: **{model_name}**
- Idioma detectado: **{idioma}**
- Modo: **{"rápido (encoder congelado)" if modo_rapido else "completo"}**
- Épocas: **{epocas}**
- Max length: **{max_len}**
- Fecha: **{fecha}**
- Cantidad de clases: **{cantidad_clases}**

## Configuración y métricas

- Validación: **{detalle_modo}**
- Muestras de entrenamiento: **{n_tr}**
- Muestras de evaluación: **{n_te}**
- Accuracy: **{acc:.4f}**
- F1-score weighted: **{f1:.4f}**
- Precision weighted: **{prec:.4f}**
- Recall weighted: **{rec:.4f}**

## Pérdida por corrida (divergencias a la vista)

{lineas_loss}

## Mapeo de clases

{mapping_lines}

## Matriz de confusión

{matriz_tabla}

## Uso en Python

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from logging_utils import setup_service_logger
logger = setup_service_logger("classification-service")


ruta_modelo = "./modelo_bert"
tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)
model = AutoModelForSequenceClassification.from_pretrained(ruta_modelo)

textos = ["ejemplo de texto a clasificar"]
inputs = tokenizer(
    textos,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length={max_len},
)

with torch.no_grad():
    logits = model(**inputs).logits
    predicciones = logits.argmax(dim=-1)

for texto, clase_id in zip(textos, predicciones):
    clase_id = int(clase_id)
    prediccion = (texto, model.config.id2label[clase_id])
```

## Dependencias

```bash
pip install torch transformers accelerate
```
"""

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(model_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, model_dir)
                    zf.write(file_path, f"modelo_bert/{arcname}")
            zf.writestr("README.md", readme_texto)
        zip_buffer.seek(0)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        try:
            del model
        except Exception:
            pass
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass

    return (
        tp, fp, tn, fn,
        acc, f1, prec, rec,
        len(y_labels), n_tr, n_te,
        zip_buffer, matriz,
    )
