import io
import os
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
import polars as pl

from models import Document


def _normalizar_fechas_a_iso(
    serie: pd.Series,
    formato_fecha: str | None,
) -> list[str | None]:
    """Normaliza una columna entera de fechas a YYYY-MM-DD (vectorizado).

    Respeta formato_fecha DMY/MDY/YMD para valores con '/'.
    Lanza ValueError si algún valor no vacío es inválido (fail-fast,
    igual que el parseo por fila anterior).
    """


    dayfirst = formato_fecha in (None, "DMY")

    original = serie.astype(object)
    vacia = original.isna() | (original.astype(str).str.strip() == "")

    parsed = pd.to_datetime(
        serie, errors="coerce", format="mixed", dayfirst=dayfirst
    )

    invalidas = ~vacia & parsed.isna()
    if invalidas.any():
        ejemplos = (
            original[invalidas].astype(str).str.slice(0, 40).unique()[:3]
        )
        raise ValueError(
            "Fechas inválidas "
            f"(formatoFecha={formato_fecha}): "
            f"{list(ejemplos)}"
        )

    iso = parsed.dt.strftime("%Y-%m-%d")
    return [None if v else f for v, f in zip(vacia.tolist(), iso.tolist())]


class BaseDocumentParser(ABC):

    @abstractmethod
    def parse(
        self,
        content: bytes,
        columna_texto: str,
        columna_clase: str | None = None,
        columna_fecha: str | None = None,
        columna_autor: str | None = None,
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
        columna_autor: str | None = None,
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

        if columna_autor:

            if columna_autor not in df.columns:

                raise ValueError(
                    f"La columna de autor "
                    f"'{columna_autor}' no existe."
                )

            columnas_mapeadas.add(
                columna_autor
            )

        columnas_extra = [
            column
            for column in df.columns
            if column not in columnas_mapeadas
        ]


        fechas_iso: list[str | None] | None = None
        if columna_fecha:
            fechas_iso = _normalizar_fechas_a_iso(
                df[columna_fecha], formato_fecha
            )


        registros = df.to_dict(orient="records")

        documents: list[Document] = []

        for i, row in enumerate(registros):

            raw_text = row.get(columna_texto)

            if raw_text is None or (isinstance(raw_text, float) and pd.isna(raw_text)):
                continue

            text = str(raw_text).strip()

            if not text:
                continue

            metadata: dict[str, Any] = {}

            for column in columnas_extra:

                value = row.get(column)

                if value is None or (isinstance(value, float) and pd.isna(value)):
                    continue

                try:
                    if pd.isna(value):
                        continue
                except (TypeError, ValueError):
                    pass

                metadata[column] = value

            if extra_metadata:

                metadata.update(
                    extra_metadata
                )

            clase = None

            if columna_clase:

                value = row.get(columna_clase)

                if value is not None and not (
                    isinstance(value, float) and pd.isna(value)
                ):
                    try:
                        es_nan = pd.isna(value)
                    except (TypeError, ValueError):
                        es_nan = False

                    if not es_nan:
                        clase = str(value)

            fecha = fechas_iso[i] if fechas_iso is not None else None

            autor = None

            if columna_autor:

                value = row.get(columna_autor)

                if value is not None and not (
                    isinstance(value, float) and pd.isna(value)
                ):
                    try:
                        es_nan = pd.isna(value)
                    except (TypeError, ValueError):
                        es_nan = False

                    if not es_nan and str(value).strip():
                        autor = str(value).strip()

            documents.append(
                Document(
                    texto=text,
                    clase=clase,
                    fecha=fecha,
                    autor=autor,
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
        columna_autor: str | None = None,
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
            columna_autor=columna_autor,
            formato_fecha=formato_fecha,
        )

    def _detectar_separador(self, muestra: str) -> str:
        candidatos = [";", ",", "\t", "|"]
        lineas = [l for l in muestra.splitlines() if l.strip()][:10]
        if not lineas:
            return ","
        mejor, mejor_score = ",", -1
        for sep in candidatos:
            conteos = [l.count(sep) for l in lineas]
            if sum(conteos) == 0:
                continue
            score = sum(conteos) - (max(conteos) - min(conteos)) * 10
            if score > mejor_score:
                mejor, mejor_score = sep, score
        return mejor

    def _leer_csv(self, content: bytes) -> pd.DataFrame:
        muestra = content[:8192].decode("utf-8", errors="replace")
        sep = self._detectar_separador(muestra)

        try:
            df_pl = pl.read_csv(
                io.BytesIO(content),
                encoding="utf8-lossy",
                separator=sep,
                infer_schema_length=0,
                ignore_errors=True,
                truncate_ragged_lines=True,
                null_values=["", "NULL", "#¿NOMBRE?"],
            )
            if len(df_pl.columns) > 1:
                return df_pl.to_pandas()
        except Exception:
            pass

        for enc in ("cp1252", "latin-1"):
            try:
                texto = content.decode(enc)
                sep2 = self._detectar_separador(texto[:8192])
                df_pl = pl.read_csv(
                    io.StringIO(texto),
                    separator=sep2,
                    infer_schema_length=0,
                    ignore_errors=True,
                    truncate_ragged_lines=True,
                )
                if len(df_pl.columns) > 1:
                    return df_pl.to_pandas()
            except Exception:
                continue

        raise ValueError("No se pudo leer el CSV con ; , \\t |")


class XLSXParser(
    DataFrameParser
):

    def parse(
        self,
        content: bytes,
        columna_texto: str,
        columna_clase: str | None = None,
        columna_fecha: str | None = None,
        columna_autor: str | None = None,
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
            columna_autor=columna_autor,
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
