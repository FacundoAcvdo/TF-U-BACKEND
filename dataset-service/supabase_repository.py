from http.client import HTTPException
from typing import Any
from pathlib import Path
import httpx

from supabase import create_client

from config import get_settings


class SupabaseRepository:

    def __init__(self) -> None:

        settings = get_settings()

        if not settings.supabase_url:
            raise RuntimeError(
                "SUPABASE_URL no está configurado."
            )


        self.base_url = (
            settings.supabase_url.rstrip("/")
        )

        self.headers = {
            "apikey": (
                settings.supabase_secret_key
            ),
            "Authorization": (
                "Bearer "
                + settings.supabase_secret_key
            ),
            "Content-Type": "application/json",
        }

        self.table = (
            settings.supabase_datasets_table
        )

        self.anon_key = (
            settings.supabase_anon_key
        )

    def get_supabase_client(self, user_jwt: str):
        client = create_client(self.base_url, self.anon_key)
        client.postgrest.auth(user_jwt)
        return client

    async def create_dataset(
        self,
        *,
        dataset_id: str,
        user_id: str,
        filename: str,
        user_token: str,
        dataset_size=int,
        embedding_size=int,
        embedding_chunked_size=int,
        tipo: str = "dataset",
    ) -> None:
        
        supabase = self.get_supabase_client(user_token)

        payload = {
            "dataset_id": dataset_id,
            "user_id": user_id,
            "dataset_name": filename,
            "dataset_size": dataset_size,
            "embedding_size": embedding_size,
            "embedding_chunked_size": embedding_chunked_size,
            "tipo": tipo,
        }

        try: 
            (supabase.table(self.table)
                .insert(payload)
                .execute())
            


        except Exception as e:
            raise RuntimeError(
                "No se pudo registrar el dataset "
                "en Supabase."
                f"Error: {e}"
            ) from e

    def get_available_dataset_name(
        self,
        user_id: str,
        filename: str,
        user_token: str
    ) -> str:

        try:
            supabase = self.get_supabase_client(user_token)

            response = (
                supabase
                .table(self.table)
                .select("dataset_name")
                .eq("user_id", user_id)
                .execute()
            )

            existing_names = [
                row["dataset_name"]
                for row in response.data
            ]

        except Exception as e:
            raise RuntimeError(
                f"No se pudieron consultar los datasets "
                f"del usuario en Supabase: {e}"
            ) from e

        path = Path(filename)

        base_name = path.stem
        extension = path.suffix

        if filename not in existing_names:
            return filename

        counter = 1

        while f"{base_name} ({counter}){extension}" in existing_names:
            counter += 1

        return f"{base_name} ({counter}){extension}"

    async def delete_dataset(
        self,
        *,
        dataset_id: str,
        user_id: str,
        user_token: str,
    ) -> None:

        supabase = self.get_supabase_client(
            user_token
        )

        try:

            (
                supabase
                .table(self.table)
                .delete()
                .eq("dataset_id", dataset_id)
                .eq("user_id", user_id)
                .execute()
            )

        except Exception as e:

            raise RuntimeError(
                "No se pudo eliminar el dataset "
                "de Supabase."
                f" Error: {e}"
            ) from e

    async def delete_datasets_by_user(
        self,
        user_id: str,
        user_token: str,
    ) -> None:

        supabase = self.get_supabase_client(
            user_token
        )

        try:

            (
                supabase
                .table(self.table)
                .delete()
                .eq("user_id", user_id)
                .execute()
            )

        except Exception as e:

            raise RuntimeError(
                "No se pudieron eliminar los datasets "
                "del usuario de Supabase."
                f" Error: {e}"
            ) from e

    async def get_datasets_by_user(
        self,
        user_id: str,
        user_token: str,
    ) -> list[dict[str, Any]]:

        supabase = self.get_supabase_client(user_jwt=user_token)

        response = (
            supabase.table("datasets")
            .select("*")
            .eq("user_id", user_id)
            .execute()
        )

        return response.data