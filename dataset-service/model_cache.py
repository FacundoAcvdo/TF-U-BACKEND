"""Cache de modelos con TTL + control de admision (copia local por servicio).

COPIAS IDENTICAS en: emotions-service, evolution-service, dataset-service,
rag-service, risk-service, topicmodeling-service, classification-service,
clustering-service. No editar copias a mano: modificar aqui y replicar.

Uso:
    slot = ModelSlot(name="emociones-es", vram_mb=800, load_fn=cargar, unload_fn=descargar)
    with slot.use() as modelo:
        ... inferencia ...
    # al salir: refcount-- y eviccion si paso el TTL sin uso.

Reglas:
- get(): si hay lugar (admision) y no esta cargado, carga con single-flight.
- Sin lugar -> ModelBusyError (el caller lo mapea a HTTP 503).
- maybe_evict(): descarga si refcount==0 y pasaron MODEL_TTL_SECONDS sin uso.
- WARMUP=false en el servicio => no precargar al arrancar (lazy igual).
"""
import gc
import logging
import os
import threading
import time

logger = logging.getLogger("model-cache")


def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def warmup_enabled():
    return os.environ.get("WARMUP", "true").lower() not in ("false", "0", "no", "off")


def ttl_seconds():
    return _env_int("MODEL_TTL_SECONDS", 600)


def vram_margin_mb():
    return _env_int("MODEL_VRAM_MARGIN_MB", 512)


def cuda_free_mb():
    """MB libres en GPU, o None si no hay CUDA (sin admision en CPU)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free_b, _total_b = torch.cuda.mem_get_info()
        return int(free_b // (1024 * 1024))
    except Exception:
        return None


def cuda_cleanup():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    gc.collect()


class ModelBusyError(RuntimeError):
    """No hay VRAM para cargar el modelo: reintentar mas tarde (=> HTTP 503)."""


class ModelSlot:
    """Un modelo con carga perezosa, TTL y control de admision."""

    def __init__(self, name, vram_mb, load_fn, unload_fn=None):
        self.name = name
        self.vram_mb = int(vram_mb)
        self._load_fn = load_fn
        self._unload_fn = unload_fn
        self._lock = threading.RLock()
        self._model = None
        self._users = 0
        self._last_used = 0.0

    @property
    def loaded(self):
        with self._lock:
            return self._model is not None

    def _admit_or_raise(self):
        free = cuda_free_mb()
        if free is None:
            return
        need = self.vram_mb + vram_margin_mb()
        if free < need:
            raise ModelBusyError(
                f"Servicio ocupado ({self.name}): libre={free}MB requerido={need}MB. Intenta mas tarde."
            )

    def get(self):
        """Devuelve el modelo cargado (carga si hace falta). Suma un usuario."""
        with self._lock:
            if self._model is None:
                self._admit_or_raise()
                logger.warning(f"[model-cache] cargando {self.name} (~{self.vram_mb}MB)...")
                t0 = time.perf_counter()
                self._model = self._load_fn()
                logger.warning(
                    f"[model-cache] {self.name} listo en {time.perf_counter() - t0:.1f}s.")
            self._users += 1
            self._last_used = time.perf_counter()
            return self._model

    def release(self):
        """Libera un usuario y evicta si paso el TTL."""
        with self._lock:
            self._users = max(0, self._users - 1)
            self.maybe_evict_locked()

    def maybe_evict_locked(self):
        if self._model is None or self._users > 0:
            return False
        idle = time.perf_counter() - self._last_used
        if idle < ttl_seconds():
            return False
        try:
            if self._unload_fn is not None:
                self._unload_fn(self._model)
        except Exception as e:
            logger.warning(f"[model-cache] fallo descargando {self.name}: {e}")
        self._model = None
        cuda_cleanup()
        logger.warning(f"[model-cache] {self.name} liberado tras {idle:.0f}s sin uso.")
        return True

    def maybe_evict(self):
        with self._lock:
            return self.maybe_evict_locked()

    def touch(self):
        with self._lock:
            self._last_used = time.perf_counter()

    def use(self):
        """Context manager: with slot.use() as modelo: ..."""
        slot = self

        class _Uso:
            def __enter__(self):
                return slot.get()

            def __exit__(self, *args):
                slot.release()
                return False

        return _Uso()


def check_fits_or_raise(name, vram_mb):
    """Admision puntual para trabajos sin slot (entrenamientos, BERTopic...)."""
    free = cuda_free_mb()
    if free is None:
        return 0
    need = int(vram_mb) + vram_margin_mb()
    if free < need:
        raise ModelBusyError(
            f"Servicio ocupado ({name}): libre={free}MB requerido={need}MB. Intenta mas tarde."
        )
    return free
