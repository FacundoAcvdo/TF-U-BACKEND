from fastapi import FastAPI, Request, HTTPException, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from auth import verify_jwt, has_accepted_terms_claim, mark_terms_accepted
from config import SERVICE_URL_ENV, ServiceConfigurationError, service_url
import httpx
import logging
import time
import uuid

from logging_utils import setup_service_logger


logger = setup_service_logger("api-gateway")
logger.setLevel(logging.INFO)
for handler in logger.handlers:
    handler.setLevel(logging.INFO)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    inicio = time.perf_counter()
    try:
        response = await call_next(request)
        latencia_ms = int((time.perf_counter() - inicio) * 1000)
        logger.info(
            f"{request.method} {request.url.path} "
            f"status={response.status_code} latencia_ms={latencia_ms} "
            f"req={request_id}"
        )
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as e:
        latencia_ms = int((time.perf_counter() - inicio) * 1000)
        logger.exception(
            f"[api-gateway] fallo procesando {request.method} {request.url.path} "
            f"req={request_id} latencia_ms={latencia_ms} causa={type(e).__name__}: {e}"
        )
        raise


def _passthrough_response(upstream: httpx.Response) -> Response:
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=dict(upstream.headers),
        media_type=upstream.headers.get("content-type"),
    )


def _map_upstream_error(request: Request, e: Exception) -> HTTPException:
    request_id = getattr(request.state, "request_id", "-")
    if isinstance(e, httpx.ConnectError):
        logger.exception(
            f"[api-gateway] microservicio no alcanzable | "
            f"{request.method} {request.url.path} req={request_id} causa={type(e).__name__}"
        )
        return HTTPException(status_code=503, detail="Servicio no disponible. Intenta más tarde.")
    if isinstance(e, httpx.TimeoutException):
        logger.exception(
            f"[api-gateway] timeout del microservicio | "
            f"{request.method} {request.url.path} req={request_id} causa={type(e).__name__}"
        )
        return HTTPException(status_code=504, detail="El servicio tardó demasiado. Intenta más tarde.")
    logger.exception(
        f"[api-gateway] error de red contra microservicio | "
        f"{request.method} {request.url.path} req={request_id} causa={type(e).__name__}: {e}"
    )
    return HTTPException(status_code=502, detail="Error de comunicación con el servicio.")


async def verify_jwt_and_terms(request: Request, user: dict = Depends(verify_jwt)) -> dict:
    if has_accepted_terms_claim(user):
        return user

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            upstream = await client.get(
                service_url("auth", "/auth/me"),
                headers={
                    "X-User-Token": user["raw_token"],
                    "X-User-ID": user["sub"],
                    "X-Request-ID": getattr(request.state, "request_id", ""),
                },
            )
        if upstream.status_code == 200:
            data = upstream.json()
            if (data.get("user") or {}).get("terms_accepted") is True:
                mark_terms_accepted(user["sub"])
                return user
    except Exception as e:
        logger.warning(f"[api-gateway] fallo consultando terms_accepted en auth-service: {e}")

    raise HTTPException(
        status_code=403,
        detail="Debes aceptar los términos y condiciones antes de utilizar la plataforma.",
    )


@app.post("/auth/login")
async def login_proxy(request: Request, response: Response):
    try:
        async with httpx.AsyncClient() as client:
            upstream = await client.post(
                service_url("auth", "/auth/login"),
                json=await request.json(),
            )
    except ServiceConfigurationError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        raise _map_upstream_error(request, e)

    for cookie_header in upstream.headers.get_list("set-cookie"):
        response.headers.append("set-cookie", cookie_header)

    if upstream.status_code >= 400:
        logger.warning(
            f"[api-gateway] login rechazado por auth-service | "
            f"status={upstream.status_code} req={getattr(request.state, 'request_id', '-')}"
        )
    else:
        try:
            body_json = upstream.json()
            user_obj = body_json.get("user") or {}
            if user_obj.get("terms_accepted") is True and user_obj.get("id"):
                mark_terms_accepted(user_obj["id"])
        except Exception:
            pass

    return _passthrough_response(upstream)


@app.get("/auth/me")
async def me_proxy(request: Request, user: dict = Depends(verify_jwt)):
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            upstream = await client.get(
                service_url("auth", "/auth/me"),
                headers={
                    "X-User-Token": user["raw_token"],
                    "X-User-ID": user["sub"],
                    "X-Request-ID": getattr(request.state, "request_id", ""),
                },
            )
    except ServiceConfigurationError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        raise _map_upstream_error(request, e)

    if upstream.status_code == 200:
        try:
            data = upstream.json()
            if (data.get("user") or {}).get("terms_accepted") is True:
                mark_terms_accepted(user["sub"])
        except Exception:
            pass

    return _passthrough_response(upstream)


@app.post("/auth/terms/accept")
async def accept_terms_proxy(request: Request, user: dict = Depends(verify_jwt)):
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            upstream = await client.post(
                service_url("auth", "/auth/terms/accept"),
                headers={
                    "X-User-Token": user["raw_token"],
                    "X-User-ID": user["sub"],
                    "X-Request-ID": getattr(request.state, "request_id", ""),
                },
            )
    except ServiceConfigurationError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        raise _map_upstream_error(request, e)

    if upstream.status_code < 400:
        mark_terms_accepted(user["sub"])

    return _passthrough_response(upstream)


@app.post("/auth/logout")
async def logout_proxy(request: Request, response: Response, user: dict = Depends(verify_jwt)):
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            upstream = await client.post(
                service_url("auth", "/auth/logout"),
                headers={
                    "X-User-Token": user["raw_token"],
                    "X-User-ID": user["sub"],
                    "X-Request-ID": getattr(request.state, "request_id", ""),
                },
            )
    except ServiceConfigurationError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        raise _map_upstream_error(request, e)

    for cookie_header in upstream.headers.get_list("set-cookie"):
        response.headers.append("set-cookie", cookie_header)

    return _passthrough_response(upstream)


def _auth_headers(request: Request, user: dict) -> dict:
    return {
        "X-User-Token": user["raw_token"],
        "X-User-ID": user["sub"],
        "X-Request-ID": getattr(request.state, "request_id", ""),
    }


async def _proxy_auth(request: Request, method: str, path: str, user: dict) -> Response:
    body = await request.body()
    headers = _auth_headers(request, user)
    if request.headers.get("content-type"):
        headers["Content-Type"] = request.headers["content-type"]
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            upstream = await client.request(
                method=method,
                url=service_url("auth", path),
                headers=headers,
                content=body if method in ["POST", "PUT", "PATCH", "DELETE"] else None,
                params=dict(request.query_params),
            )
    except ServiceConfigurationError as e:
        raise HTTPException(500, str(e))
    except Exception as e:
        raise _map_upstream_error(request, e)
    if upstream.status_code >= 500:
        logger.error(
            f"[api-gateway] auth-service devolvio {upstream.status_code} | "
            f"{method} {path} req={getattr(request.state, 'request_id', '-')} user_id={user.get('sub')}"
        )
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=dict(upstream.headers),
        media_type=upstream.headers.get("content-type"),
    )


@app.get("/auth/admin/users")
async def admin_users_list_proxy(request: Request, user: dict = Depends(verify_jwt_and_terms)):
    return await _proxy_auth(request, "GET", "/auth/admin/users", user)


@app.post("/auth/admin/users")
async def admin_users_create_proxy(request: Request, user: dict = Depends(verify_jwt_and_terms)):
    return await _proxy_auth(request, "POST", "/auth/admin/users", user)


@app.patch("/auth/admin/users/{user_id}/role")
async def admin_users_role_proxy(request: Request, user_id: str, user: dict = Depends(verify_jwt_and_terms)):
    return await _proxy_auth(request, "PATCH", f"/auth/admin/users/{user_id}/role", user)


@app.patch("/auth/admin/users/{user_id}/password")
async def admin_users_password_proxy(request: Request, user_id: str, user: dict = Depends(verify_jwt_and_terms)):
    return await _proxy_auth(request, "PATCH", f"/auth/admin/users/{user_id}/password", user)


@app.delete("/auth/admin/users/{user_id}")
async def admin_users_delete_proxy(request: Request, user_id: str, user: dict = Depends(verify_jwt_and_terms)):
    return await _proxy_auth(request, "DELETE", f"/auth/admin/users/{user_id}", user)


@app.api_route("/{service}/{operation}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def gateway(service: str, operation: str, request: Request, user: dict = Depends(verify_jwt_and_terms)):
    return await _proxy_service(request, service, operation, user)


@app.patch("/rag/chat/mode")
async def rag_chat_mode_proxy(request: Request, user: dict = Depends(verify_jwt_and_terms)):
    return await _proxy_service(request, "rag", "chat/mode", user)


@app.api_route("/{service}/jobs", methods=["GET"])
async def jobs_list_proxy(service: str, request: Request, user: dict = Depends(verify_jwt_and_terms)):
    return await _proxy_service(request, service, "jobs", user)


@app.api_route("/{service}/jobs/{job_id}", methods=["GET"])
async def job_status_proxy(service: str, job_id: str, request: Request, user: dict = Depends(verify_jwt_and_terms)):
    return await _proxy_service(request, service, f"jobs/{job_id}", user)


@app.api_route("/{service}/descargar/{job_id}", methods=["GET"])
async def descargar_proxy(service: str, job_id: str, request: Request, user: dict = Depends(verify_jwt_and_terms)):
    return await _proxy_service(request, service, f"descargar/{job_id}", user)


async def _proxy_service(request: Request, service: str, operation: str, user: dict) -> Response:
    if service not in SERVICE_URL_ENV or service == "auth":
        raise HTTPException(404, "Servicio no encontrado")

    try:
        target_url = service_url(service, operation)
    except ServiceConfigurationError as e:
        raise HTTPException(500, str(e))

    EXCLUDED_HEADERS = {
        "host", "content-length", "connection",
        "transfer-encoding", "authorization",
    }

    forwarded_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in EXCLUDED_HEADERS
    }

    forwarded_headers["X-User-Token"] = user["raw_token"]
    forwarded_headers["X-User-ID"] = user["sub"]
    forwarded_headers["X-Request-ID"] = getattr(request.state, "request_id", "")

    body = await request.body()

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            upstream = await client.request(
                method=request.method,
                url=target_url,
                headers=forwarded_headers,
                content=body if request.method in ["POST", "PUT", "PATCH", "DELETE"] else None,
                params=dict(request.query_params),
            )
    except Exception as e:
        raise _map_upstream_error(request, e)

    if upstream.status_code >= 500:
        logger.error(
            f"[api-gateway] {service}/{operation} devolvio {upstream.status_code} | "
            f"user_id={user.get('sub')} req={getattr(request.state, 'request_id', '-')}"
        )

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=dict(upstream.headers),
        media_type=upstream.headers.get("content-type"),
    )
