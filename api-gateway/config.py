import os
from urllib.parse import urlsplit


SERVICE_URL_ENV = {
    "auth": "AUTH_SERVICE_URL",
    "clustering": "CLUSTERING_SERVICE_URL",
    "dataset": "DATASET_SERVICE_URL",
    "eda": "EDA_SERVICE_URL",
    "emotions": "EMOTIONS_SERVICE_URL",
    "topicmodeling": "TOPIC_MODELING_SERVICE_URL",
    "wordcloud": "WORDCLOUD_SERVICE_URL",
    "classification": "CLASSIFICATION_SERVICE_URL",
    "rag": "RAG_SERVICE_URL",
    "datacleaning": "DATA_CLEANING_SERVICE_URL",
    "evolution": "EVOLUTION_SERVICE_URL",
    "suicide": "SUICIDE_SERVICE_URL",
    "report": "REPORT_SERVICE_URL",
}


class ServiceConfigurationError(ValueError):
    """Raised when a configured service URL is missing or invalid."""


def resolve_service_url(service: str) -> str:
    """Return a normalized service URL, validating its configured value."""
    try:
        environment_variable = SERVICE_URL_ENV[service]
    except KeyError as exc:
        raise ServiceConfigurationError(f"Servicio no configurado: {service}") from exc

    configured_url = os.getenv(environment_variable)
    if not configured_url or not configured_url.strip():
        raise ServiceConfigurationError(
            f"Falta la variable {environment_variable} para el servicio {service}"
        )

    normalized_url = configured_url.strip().rstrip("/")
    try:
        parsed_url = urlsplit(normalized_url)
    except ValueError as exc:
        raise ServiceConfigurationError(
            f"La variable {environment_variable} no contiene una URL válida"
        ) from exc
    if (
        parsed_url.scheme not in {"http", "https"}
        or not parsed_url.netloc
        or parsed_url.query
        or parsed_url.fragment
        or any(character.isspace() for character in normalized_url)
    ):
        raise ServiceConfigurationError(
            f"La variable {environment_variable} no contiene una URL válida"
        )

    return normalized_url


def service_url(service: str, path: str) -> str:
    """Join a validated service URL and path without duplicate slashes."""
    return f"{resolve_service_url(service)}/{path.lstrip('/')}"
