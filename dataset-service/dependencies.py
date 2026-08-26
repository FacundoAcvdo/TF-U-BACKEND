from dataclasses import dataclass

from fastapi import Header, HTTPException, status

from config import get_settings


@dataclass
class RequestContext:
    user_id: str


async def get_request_context(
    x_user_id: str | None = Header(
        default=None,
        alias="X-User-Id",
    ),
) -> RequestContext:

    if not x_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-User-Id requerido.",
        )

    user_id = x_user_id.strip()

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-User-Id no puede estar vacío.",
        )

    return RequestContext(
        user_id=user_id
    )