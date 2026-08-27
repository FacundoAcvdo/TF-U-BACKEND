import io
import os
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from models import Document


class BaseDocumentParser(ABC):

    @abstractmethod
    def parse(
        self,
        content: bytes,
        columna_texto: str,
        columna_clase: str | None = None,
        columna_fecha: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
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

                    fecha = str(value)

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
    ) -> list[Document]:

        df = self._leer_csv(content)

        return self.dataframe_to_documents(
            df=df,
            columna_texto=columna_texto,
            columna_clase=columna_clase,
            columna_fecha=columna_fecha,
            extra_metadata=extra_metadata,
        )

    def _leer_csv(
        self,
        content: bytes,
    ) -> pd.DataFrame:

        ultimo_error: Exception | None = None

        for encoding in self.ENCODINGS_A_PROBAR:

            try:

                return pd.read_csv(
                    io.BytesIO(content),
                    sep=None,
                    engine="python",
                    dtype=str,
                    encoding=encoding,
                )

            except UnicodeDecodeError as exc:

                ultimo_error = exc
                continue

            except Exception as exc:
                raise ValueError(f"No se pudo leer el CSV: {exc}") from exc

        raise ValueError(
            "No se pudo leer el CSV: no se "
            "reconoce la codificación del "
            "archivo "
            f"(último error: {ultimo_error})"
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