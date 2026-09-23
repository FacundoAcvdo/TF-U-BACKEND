import io
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd
import polars as pl

from models import Document


def _normalizar_fecha_a_iso(valor: Any, formato_fecha: str | None) -> str | None:
    """Convierte cualquier valor de fecha a YYYY-MM-DD.

    Si formato_fecha es DMY/MDY/YMD y el valor contiene '/', se respeta ese orden.
    Si no, intenta parseo robusto con pandas y falla con ValueError si es inválida.
    """
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None

    # Si ya es datetime / Timestamp / date
    if isinstance(valor, (pd.Timestamp, datetime)):
        try:
            dt = pd.to_datetime(valor)
            if pd.isna(dt):
                return None
            return dt.strftime("%Y-%m-%d")
        except Exception:
            return None
    # Polars puede entregar date objects
    if hasattr(valor, "isoformat") and not isinstance(valor, str):
        try:
            s = str(valor)
            dt = pd.to_datetime(s, errors="coerce")
            if pd.isna(dt):
                return None
            return dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    s = str(valor).strip()
    if not s:
        return None

    # Ya ISO YYYY-MM-DD o con hora
    m_iso = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})", s)
    if m_iso:
        try:
            dt = pd.to_datetime(s, errors="coerce")
            if pd.isna(dt):
                raise ValueError(f"Fecha ISO inválida: {s}")
            return dt.strftime("%Y-%m-%d")
        except Exception as e:
            raise ValueError(f"Fecha inválida '{s}': {e}") from e

    # Con barras: usar formatoFecha explícito si viene del front
    if "/" in s and formato_fecha in ("DMY", "MDY", "YMD"):
        # tomar solo la parte de fecha antes de espacio/coma
        fecha_part = re.split(r"[\s,]+", s)[0]
        parts = fecha_part.split("/")
        if len(parts) == 3:
            try:
                nums = [int(p.strip()) for p in parts]
            except ValueError:
                raise ValueError(f"Fecha con barras inválida: {s}")
            if formato_fecha == "DMY":
                d, m, y = nums
            elif formato_fecha == "MDY":
                m, d, y = nums
            else:  # YMD
                y, m, d = nums
            if y < 100:
                y += 2000
            # validación básica
            if not (1 <= m <= 12 and 1 <= d <= 31):
                raise ValueError(
                    f"Fecha '{s}' con formato {formato_fecha} da mes={m} día={d} inválido. "
                    f"¿Seleccionaste el formato correcto? (DMY=13/08/2026, MDY=08/13/2026)"
                )
            try:
                dt = datetime(y, m, d)
                return dt.strftime("%Y-%m-%d")
            except ValueError as e:
                raise ValueError(f"Fecha inválida '{s}' ({formato_fecha}): {e}") from e

    # Fallback genérico: intentar con pandas mixed (dayfirst según formato)
    # Si formato es MDY, dayfirst=False; DMY -> True
    dayfirst = True if formato_fecha in (None, "DMY") else False if formato_fecha == "MDY" else False
    # YMD ya manejado, pero por si acaso
    try:
        dt = pd.to_datetime(s, errors="coerce", format="mixed", dayfirst=dayfirst)
        if pd.isna(dt):
            # último intento sin format mixed
            dt = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
        if pd.isna(dt):
            raise ValueError(f"No se pudo parsear fecha '{s}' (formatoFecha={formato_fecha})")
        return dt.strftime("%Y-%m-%d")
    except Exception as e:
        raise ValueError(f"Fecha inválida '{s}': {e}") from e


class BaseDocumentParser(ABC):

    @abstractmethod
    def parse(
        self,
        content: bytes,
        columna_texto: str,
        columna_clase: str | None = None,
        columna_fecha: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
        formato_fecha: str | None = None,
    ) -> list[Document]:

        raise NotImplementedError


class DataFrameParser(
    BaseDocumentParser
):

    def dataframe_to_documents(
        self,
        df: pd.DataFrame,
        columna_texto: str,
        columna_clase: str | None,
        columna_fecha: str | None,
        extra_metadata: dict[str, Any] | None,
        formato_fecha: str | None = None,
    ) -> list[Document]:

        if columna_texto not in df.columns:

            raise ValueError(
                f"La columna de texto "
                f"'{columna_texto}' no existe. "
                f"Columnas disponibles: "
                f"{list(df.columns)}"
            )

        columnas_mapeadas = {
            columna_texto
        }

        if columna_clase:

            if columna_clase not in df.columns:

                raise ValueError(
                    f"La columna de clase "
                    f"'{columna_clase}' no existe."
                )

            columnas_mapeadas.add(
                columna_clase
            )

        if columna_fecha:

            if columna_fecha not in df.columns:

                raise ValueError(
                    f"La columna de fecha "
                    f"'{columna_fecha}' no existe."
                )

            columnas_mapeadas.add(
                columna_fecha
            )

        columnas_extra = [
            column
            for column in df.columns
            if column not in columnas_mapeadas
        ]

        documents: list[Document] = []

        for index, row in df.iterrows():

            raw_text = row[columna_texto]

            if pd.isna(raw_text):
                continue

            text = str(
                raw_text
            ).strip()

            if not text:
                continue

            metadata: dict[str, Any] = {}

            for column in columnas_extra:

                value = row[column]

                if pd.isna(value):
                    continue

                metadata[column] = value

            if extra_metadata:

                metadata.update(
                    extra_metadata
                )

            clase = None

            if columna_clase:

                value = row[
                    columna_clase
                ]

                if not pd.isna(value):

                    clase = str(value)

            fecha = None

            if columna_fecha:

                value = row[
                    columna_fecha
                ]

                if not pd.isna(value):
                    # Normaliza siempre a YYYY-MM-DD; lanza ValueError si es inválida
                    normalized = _normalizar_fecha_a_iso(value, formato_fecha)
                    fecha = normalized

            documents.append(
                Document(
                    texto=text,
                    clase=clase,
                    fecha=fecha,
                    metadata=metadata,
                )
            )

        return documents


class CSVParser(
    DataFrameParser
):

    ENCODINGS_A_PROBAR = (
        "utf-8-sig",
        "utf-8",
        "cp1252",
        "latin-1",
    )

    def parse(
        self,
        content: bytes,
        columna_texto: str,
        columna_clase: str | None = None,
        columna_fecha: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
        formato_fecha: str | None = None,
    ) -> list[Document]:

        df = self._leer_csv(content)

        return self.dataframe_to_documents(
            df=df,
            columna_texto=columna_texto,
            columna_clase=columna_clase,
            columna_fecha=columna_fecha,
            extra_metadata=extra_metadata,
            formato_fecha=formato_fecha,
        )

    def _leer_csv(
        self,
        content: bytes,
    ) -> pd.DataFrame:
        ultimo_error: Exception | None = None

        for encoding in self.ENCODINGS_A_PROBAR:
            try:
                if encoding.lower() in ("utf-8", "utf8"):
                    df_pl = pl.read_csv(
                        io.BytesIO(content),
                        infer_schema_length=0,
                        ignore_errors=True,
                        truncate_ragged_lines=True,
                    )
                else:
                    texto = content.decode(encoding)
                    df_pl = pl.read_csv(
                        io.StringIO(texto),
                        infer_schema_length=0,
                        ignore_errors=True,
                        truncate_ragged_lines=True,
                    )

                nuevas_columnas = {
                    col: f"unnamed_{i}"
                    for i, col in enumerate(df_pl.columns)
                    if not col or not col.strip()
                }
                if nuevas_columnas:
                    df_pl = df_pl.rename(nuevas_columnas)

                return (
                    df_pl.to_pandas()
                    if hasattr(df_pl, "to_pandas")
                    else pd.DataFrame(df_pl.to_dicts())
                )

            except UnicodeDecodeError as exc:
                ultimo_error = exc
                continue
            except Exception as exc:
                ultimo_error = exc
                continue

        raise ValueError(
            f"No se pudo leer el CSV: no se reconoce la codificación (último error: {ultimo_error})"
        )


class XLSXParser(
    DataFrameParser
):

    def parse(
        self,
        content: bytes,
        columna_texto: str,
        columna_clase: str | None = None,
        columna_fecha: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
        formato_fecha: str | None = None,
    ) -> list[Document]:

        try:
            df = pd.read_excel(
                io.BytesIO(content),
                engine="openpyxl",
            )

        except Exception as exc:

            raise ValueError(
                f"No se pudo leer el Excel: {exc}"
            ) from exc

        return self.dataframe_to_documents(
            df=df,
            columna_texto=columna_texto,
            columna_clase=columna_clase,
            columna_fecha=columna_fecha,
            extra_metadata=extra_metadata,
            formato_fecha=formato_fecha,
        )


class ParserFactory:

    _registry: dict[
        str,
        BaseDocumentParser,
    ] = {}

    @classmethod
    def register(
        cls,
        extension: str,
        parser: BaseDocumentParser,
    ) -> None:

        extension = extension.lower()

        if not extension.startswith("."):
            extension = f".{extension}"

        cls._registry[
            extension
        ] = parser

    @classmethod
    def get_parser(
        cls,
        filename: str,
    ) -> BaseDocumentParser:

        extension = os.path.splitext(
            filename
        )[1].lower()

        parser = cls._registry.get(
            extension
        )

        if not parser:

            supported = ", ".join(
                sorted(
                    cls._registry.keys()
                )
            )

            raise ValueError(
                f"Extensión '{extension}' "
                f"no soportada. "
                f"Formatos válidos: "
                f"{supported}"
            )

        return parser


ParserFactory.register(
    ".csv",
    CSVParser(),
)

ParserFactory.register(
    ".xlsx",
    XLSXParser(),
)