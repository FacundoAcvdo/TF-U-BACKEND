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
- Cada ModelSlot corre un hilo daemon que llama a maybe_evict() cada
  MODEL_EVICT_POLL_SECONDS (defecto: min(30, TTL/2)). Sin este hilo la
  liberacion solo ocurria en release()/proxima request y la VRAM quedaba
  ocupada indefinidamente en idle (bug 2026-10-16).
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
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    except Exception:
        pass


def _free_gpu_object(obj):
    """Intento best-effort de bajar a CPU los pesos antes de soltar la ref.

    `del pipe` dentro de un unload_fn NO libera nada por si solo (solo borra
    el nombre local del argumento). Lo que libera es soltar TODAS las refs
    (slot._model=None + gc) + empty_cache. Mover a CPU primero reduce la
           memoria pico retenida por el caching allocator.
    """
    try:
        inner = getattr(obj, "model", None)
        if inner is not None:
            for meth in ("to", "cpu"):
                try:
                    if meth == "to":
                        inner.to("cpu")
                    else:
                        fn = getattr(inner, "cpu", None)
                        if callable(fn):
                            fn()
                    break
                except Exception:
                    continue
        for meth in ("to", "cpu"):
            try:
                if meth == "to":
                    fn = getattr(obj, "to", None)
                    if callable(fn):
                        fn("cpu")
                else:
                    fn = getattr(obj, "cpu", None)
                    if callable(fn):
                        fn()
                break
            except Exception:
                continue
    except Exception:
        pass
    for attr in ("model", "_model", "backend_model", "pipe"):
        try:
            if hasattr(obj, attr):
                setattr(obj, attr, None)
        except Exception:
            pass


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
        self._stop_evictor = False
        self._evictor = None
        self._start_evictor()

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
        # Soltar la ref del slot ANTES de llamar al unload_fn: `del pipe`
        # dentro del unload solo borra el nombre local, asi que el objeto
        # se liberaria recien al hacer self._model=None despues. Separando
        # la ref primero, el unload + _free_gpu_object trabajan sobre el
        # ultimo dueño real y el empty_cache posterior si devuelve a nvidia-smi.
        model = self._model
        self._model = None
        try:
            if self._unload_fn is not None:
                try:
                    self._unload_fn(model)
                except Exception as e:
                    logger.warning(f"[model-cache] fallo descargando {self.name}: {e}")
            _free_gpu_object(model)
        finally:
            try:
                del model
            except Exception:
                pass
            cuda_cleanup()
        logger.warning(f"[model-cache] {self.name} liberado tras {idle:.0f}s sin uso.")
        return True

    def maybe_evict(self):
        with self._lock:
            return self.maybe_evict_locked()

    def touch(self):
        with self._lock:
            self._last_used = time.perf_counter()

    def _poll_seconds(self):
        try:
            forced = int(os.environ.get("MODEL_EVICT_POLL_SECONDS", "0"))
            if forced > 0:
                return max(1, forced)
        except (TypeError, ValueError):
            pass
        try:
            ttl = int(ttl_seconds())
        except Exception:
            ttl = 600
        if ttl <= 0:
            return 5
        return max(5, min(30, ttl // 2 if ttl >= 10 else 5))

    def _start_evictor(self):
        # Hilo daemon por slot: sin esto, maybe_evict() solo se llamaba en
        # release() (idle ~0s, nunca >= TTL) y al inicio de la SIGUIENTE
        # inferencia. Resultado: sin trafico nuevo, la VRAM jamas se liberaba.
        if self._evictor is not None:
            return
        def _loop():
            while not self._stop_evictor:
                try:
                    time.sleep(self._poll_seconds())
                except Exception:
                    time.sleep(5)
                try:
                    self.maybe_evict()
                except Exception:
                    pass
        try:
            t = threading.Thread(target=_loop, name=f"evict-{self.name}", daemon=True)
            t.start()
            self._evictor = t
        except Exception as e:
            logger.warning(f"[model-cache] no se pudo iniciar evictor de {self.name}: {e}")

    def stop_evictor(self):
        self._stop_evictor = True

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
