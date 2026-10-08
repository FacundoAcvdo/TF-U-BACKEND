from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel


class TrainingRequest(BaseModel):
    dataset: str
    modelo: Literal[
        "Logistic Regression",
        "Support Vector Machine",
        "Random Forest",
        "Naive Bayes",
        "BERT",
    ] = "Logistic Regression"
    modo: Literal["validacion_cruzada", "porcentajes"]
    k: Optional[int] = 5
    porcentaje_entrenamiento: Optional[int] = 50
    porcentaje_prueba: Optional[int] = 50
    hiperparametros: Optional[Dict[str, Any]] = None


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
    matriz_confusion: list[list[int]]
    clases: list[str]


class TrainingResponse(BaseModel):
    resumen: ResumenMenciones
    model_id: str
    nombre_archivo: str
