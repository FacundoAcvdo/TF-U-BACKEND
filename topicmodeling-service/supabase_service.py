from typing import Any

from supabase import create_client

from config import get_settings

class SupabaseService:

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
            "apikey": settings.supabase_secret_key,
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

    def get_supabase_client(
        self,
        user_jwt: str,
    ):

        client = create_client(
            self.base_url,
            self.anon_key,
        )

        client.postgrest.auth(user_jwt)

        return client

    async def get_dataset_by_name(
        self,
        *,
        dataset_name: str,
        user_id: str,
        user_token: str,
    ) -> dict[str, Any] | None:
        print(f"Obteniendo dataset '{dataset_name}' para user_id: {user_id}")
        supabase = self.get_supabase_client(
            user_jwt=user_token,
        )
        print(f"Consultando dataset '{dataset_name}' para user_id: {user_id}")
        response = (
            supabase
            .table(self.table)
            .select("dataset_id, dataset_name")
            .eq("user_id", user_id)
            .eq("dataset_name", dataset_name)
            .limit(1)
            .execute()
        )

        if not response.data:

            return None

        return response.data[0]