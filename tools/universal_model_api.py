"""
tools/universal_model_api.py

Production-oriented HTTP API for UniversalModel inference.
No external web framework required.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import urlparse

if __package__ in (None, ""):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from models.universal_model import UniversalModel


def _is_loopback_host(host: str) -> bool:
    h = str(host or "").strip().lower()
    return h in {"127.0.0.1", "localhost", "::1"}


def _requires_api_key(host: str, api_key: str, allow_insecure_no_auth: bool) -> bool:
    if bool(allow_insecure_no_auth):
        return False
    if str(api_key or "").strip():
        return False
    return not _is_loopback_host(host)


def _json_response(
    handler: BaseHTTPRequestHandler,
    status: int,
    payload: Dict[str, Any],
    *,
    request_id: Optional[str] = None,
) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    if request_id:
        handler.send_header("X-Request-Id", request_id)
    handler.end_headers()
    handler.wfile.write(body)


def _internal_error_payload(*, request_id: str) -> Dict[str, Any]:
    return {"ok": False, "error": "internal_error", "request_id": str(request_id)}


class UniversalModelAPIHandler(BaseHTTPRequestHandler):
    model: UniversalModel = None  # type: ignore[assignment]
    model_dir: str = ""
    api_key: Optional[str] = None
    max_body_bytes: int = 1_000_000
    require_nonempty_memory: bool = False
    start_time_ms: int = int(time.time() * 1000)
    predict_calls_total: int = 0
    predict_errors_total: int = 0
    reload_calls_total: int = 0
    reload_errors_total: int = 0
    allow_reload_endpoint: bool = False
    rate_limit_enabled: bool = False
    rate_limit_rps: float = 20.0
    rate_limit_burst: int = 40
    rate_limit_state_max_entries: int = 10000
    rate_limit_state: Dict[str, Dict[str, float]] = {}
    rate_limit_lock = threading.Lock()

    def _request_id(self) -> str:
        incoming = str(self.headers.get("X-Request-Id", "")).strip()
        return incoming if incoming else uuid.uuid4().hex

    def _log_event(self, event: Dict[str, Any]) -> None:
        try:
            print(json.dumps(event, ensure_ascii=False), flush=True)
        except Exception:
            pass

    def _auth_ok(self) -> bool:
        api_key = self.api_key
        if not api_key:
            return True
        auth = str(self.headers.get("Authorization", "")).strip()
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
            if hmac.compare_digest(token, api_key):
                return True
        x_key = str(self.headers.get("X-API-Key", "")).strip()
        return hmac.compare_digest(x_key, api_key)

    def _memory_has_entries(self, path: str) -> bool:
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        return True
        except Exception:
            return False
        return False

    def _client_ip(self) -> str:
        xff = str(self.headers.get("X-Forwarded-For", "")).strip()
        if xff:
            return str(xff.split(",")[0]).strip() or "unknown"
        try:
            return str(self.client_address[0] or "unknown")
        except Exception:
            return "unknown"

    def _rate_limit_ok(self) -> tuple[bool, int]:
        if not bool(self.rate_limit_enabled):
            return True, 0
        key = self._client_ip()
        now = time.monotonic()
        rps = max(1e-6, float(self.rate_limit_rps))
        burst = max(1.0, float(self.rate_limit_burst))
        with self.rate_limit_lock:
            bucket = self.rate_limit_state.get(key)
            tokens = burst
            last = now
            if isinstance(bucket, dict):
                tokens = float(bucket.get("tokens", burst))
                last = float(bucket.get("last", now))
            elapsed = max(0.0, float(now - last))
            tokens = min(burst, float(tokens) + (elapsed * rps))
            if tokens >= 1.0:
                tokens -= 1.0
                self.rate_limit_state[key] = {"tokens": float(tokens), "last": float(now)}
                if len(self.rate_limit_state) > int(self.rate_limit_state_max_entries):
                    stale_before = float(now - 300.0)
                    for ip in list(self.rate_limit_state.keys()):
                        rec = self.rate_limit_state.get(ip) or {}
                        if float(rec.get("last", now)) < stale_before:
                            self.rate_limit_state.pop(ip, None)
                    if len(self.rate_limit_state) > int(self.rate_limit_state_max_entries):
                        self.rate_limit_state.pop(next(iter(self.rate_limit_state)), None)
                return True, 0
            retry_after_ms = int(max(1.0, ((1.0 - tokens) / rps) * 1000.0))
            self.rate_limit_state[key] = {"tokens": float(tokens), "last": float(now)}
            return False, retry_after_ms

    def _readiness(self) -> Dict[str, Any]:
        issues = []
        memory_dir = ""
        if self.model is None:
            issues.append("model_not_loaded")
        else:
            memory_dir = str(self.model.config.memory_dir or "")
            if not memory_dir or not os.path.isdir(memory_dir):
                issues.append("memory_dir_missing")
            else:
                mem_file = os.path.join(memory_dir, "memories.jsonl")
                if not os.path.isfile(mem_file):
                    issues.append("memory_file_missing")
                elif self.require_nonempty_memory and not self._memory_has_entries(mem_file):
                    issues.append("memory_file_empty")

        ok = not issues
        return {
            "ok": ok,
            "issues": issues,
            "model_dir": self.model_dir,
            "memory_dir": memory_dir,
            "require_nonempty_memory": bool(self.require_nonempty_memory),
        }

    def _metrics_text(self) -> str:
        uptime_seconds = max(0.0, (time.time() * 1000.0 - float(self.start_time_ms)) / 1000.0)
        lines = [
            "# TYPE universal_model_api_predict_calls_total counter",
            f"universal_model_api_predict_calls_total {int(self.predict_calls_total)}",
            "# TYPE universal_model_api_predict_errors_total counter",
            f"universal_model_api_predict_errors_total {int(self.predict_errors_total)}",
            "# TYPE universal_model_api_reload_calls_total counter",
            f"universal_model_api_reload_calls_total {int(self.reload_calls_total)}",
            "# TYPE universal_model_api_reload_errors_total counter",
            f"universal_model_api_reload_errors_total {int(self.reload_errors_total)}",
            "# TYPE universal_model_api_uptime_seconds gauge",
            f"universal_model_api_uptime_seconds {uptime_seconds:.3f}",
        ]
        return "\n".join(lines) + "\n"

    def do_GET(self) -> None:
        rid = self._request_id()
        path = urlparse(self.path).path
        if path == "/health":
            return _json_response(
                self,
                200,
                {
                    "ok": True,
                    "model_dir": self.model_dir,
                    "memory_dir": self.model.config.memory_dir,
                },
                request_id=rid,
            )
        if path == "/ready":
            readiness = self._readiness()
            status = 200 if readiness.get("ok") else 503
            return _json_response(self, status, readiness, request_id=rid)
        if path == "/metrics":
            body = self._metrics_text().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Request-Id", rid)
            self.end_headers()
            self.wfile.write(body)
            return
        _json_response(self, 404, {"ok": False, "error": "not found"}, request_id=rid)

    def do_POST(self) -> None:
        rid = self._request_id()
        path = urlparse(self.path).path
        if not self._auth_ok():
            return _json_response(self, 401, {"ok": False, "error": "unauthorized"}, request_id=rid)
        if path == "/predict":
            ok_rate, retry_after_ms = self._rate_limit_ok()
            if not ok_rate:
                return _json_response(
                    self,
                    429,
                    {"ok": False, "error": "rate_limited", "retry_after_ms": int(retry_after_ms)},
                    request_id=rid,
                )
            return self._predict(rid)
        if path == "/reload":
            if not bool(self.allow_reload_endpoint):
                return _json_response(self, 404, {"ok": False, "error": "not found"}, request_id=rid)
            return self._reload_model(rid)
        _json_response(self, 404, {"ok": False, "error": "not found"}, request_id=rid)

    def _read_json(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length", "0") or 0)
        if n < 0:
            raise ValueError("invalid content-length")
        if n > int(self.max_body_bytes):
            raise ValueError(f"payload too large ({n} > {self.max_body_bytes})")
        raw = self.rfile.read(n) if n > 0 else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON: {e.msg}") from e
        if not isinstance(data, dict):
            raise ValueError("JSON body must be an object")
        return data

    def _validate_predict_payload(self, payload: Dict[str, Any]) -> None:
        if "obs" not in payload:
            raise ValueError("missing field 'obs'")
        obs = payload.get("obs")
        if isinstance(obs, bool) or not isinstance(obs, (dict, list, int, float)):
            raise ValueError("'obs' must be JSON dict/list/number")
        top_k = payload.get("top_k")
        if top_k is not None:
            if isinstance(top_k, bool) or not isinstance(top_k, int):
                raise ValueError("'top_k' must be integer")
            if top_k < 1 or top_k > 5000:
                raise ValueError("'top_k' out of range [1, 5000]")
        min_score = payload.get("min_score")
        if min_score is not None and (isinstance(min_score, bool) or not isinstance(min_score, (int, float))):
            raise ValueError("'min_score' must be numeric")
        verse = payload.get("verse")
        if verse is not None and not isinstance(verse, str):
            raise ValueError("'verse' must be string")
        recent_history = payload.get("recent_history")
        if recent_history is not None and not isinstance(recent_history, list):
            raise ValueError("'recent_history' must be a list")

    def _predict(self, request_id: str) -> None:
        start = time.perf_counter()
        self.__class__.predict_calls_total += 1
        try:
            payload = self._read_json()
            self._validate_predict_payload(payload)

            pred = self.model.predict(
                obs=payload.get("obs"),
                verse_name=payload.get("verse"),
                top_k=payload.get("top_k"),
                min_score=payload.get("min_score"),
                recent_history=payload.get("recent_history"),
            )
            latency_ms = int((time.perf_counter() - start) * 1000)
            self._log_event(
                {
                    "event": "predict",
                    "request_id": request_id,
                    "latency_ms": latency_ms,
                    "matched": int(pred.get("matched", 0)),
                    "confidence": float(pred.get("confidence", 0.0)),
                    "strategy": pred.get("strategy"),
                    "model_dir": self.model_dir,
                }
            )
            return _json_response(
                self,
                200,
                {"ok": True, "prediction": pred, "request_id": request_id, "latency_ms": latency_ms},
                request_id=request_id,
            )
        except ValueError as e:
            self.__class__.predict_errors_total += 1
            self._log_event({"event": "predict_error", "request_id": request_id, "error": str(e)})
            return _json_response(self, 400, {"ok": False, "error": str(e)}, request_id=request_id)
        except Exception as e:
            self.__class__.predict_errors_total += 1
            self._log_event({"event": "predict_error", "request_id": request_id, "error": str(e)})
            return _json_response(self, 500, _internal_error_payload(request_id=request_id), request_id=request_id)

    def _reload_model(self, request_id: str) -> None:
        self.__class__.reload_calls_total += 1
        try:
            payload = self._read_json()
            model_dir = str(payload.get("model_dir") or self.model_dir)
            self.__class__.model = UniversalModel.load(model_dir)
            self.__class__.model_dir = model_dir
            return _json_response(
                self,
                200,
                {
                    "ok": True,
                    "model_dir": model_dir,
                    "memory_dir": self.model.config.memory_dir,
                    "request_id": request_id,
                },
                request_id=request_id,
            )
        except ValueError as e:
            self.__class__.reload_errors_total += 1
            self._log_event({"event": "reload_error", "request_id": request_id, "error": str(e)})
            return _json_response(self, 400, {"ok": False, "error": str(e)}, request_id=request_id)
        except Exception as e:
            self.__class__.reload_errors_total += 1
            self._log_event({"event": "reload_error", "request_id": request_id, "error": str(e)})
            return _json_response(self, 500, _internal_error_payload(request_id=request_id), request_id=request_id)

    def log_message(self, format: str, *args: Any) -> None:
        # Keep API output concise.
        return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, default="models/universal_model")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--api_key", type=str, default=os.environ.get("UNIVERSAL_MODEL_API_KEY", ""))
    ap.add_argument("--max_body_bytes", type=int, default=int(os.environ.get("UNIVERSAL_MODEL_MAX_BODY_BYTES", "1000000")))
    ap.add_argument(
        "--enable_rate_limit",
        action="store_true",
        help="Enable token-bucket rate limiting for POST /predict.",
    )
    ap.add_argument(
        "--rate_limit_rps",
        type=float,
        default=float(os.environ.get("UNIVERSAL_MODEL_RATE_LIMIT_RPS", "20")),
    )
    ap.add_argument(
        "--rate_limit_burst",
        type=int,
        default=int(os.environ.get("UNIVERSAL_MODEL_RATE_LIMIT_BURST", "40")),
    )
    ap.add_argument(
        "--enable_reload_endpoint",
        action="store_true",
        help="Enable POST /reload endpoint (disabled by default).",
    )
    ap.add_argument(
        "--allow_insecure_no_auth",
        action="store_true",
        help="Allow empty API key on non-loopback hosts (not recommended).",
    )
    ap.add_argument(
        "--require_nonempty_memory",
        action="store_true",
        help="Treat readiness as failed when central memory has zero records.",
    )
    args = ap.parse_args()

    if _requires_api_key(args.host, args.api_key, bool(args.allow_insecure_no_auth)):
        raise SystemExit(
            "Refusing to bind non-loopback host without API key. "
            "Set --api_key (or UNIVERSAL_MODEL_API_KEY), or use --allow_insecure_no_auth."
        )

    UniversalModelAPIHandler.model = UniversalModel.load(args.model_dir)
    UniversalModelAPIHandler.model_dir = args.model_dir
    UniversalModelAPIHandler.api_key = str(args.api_key).strip() or None
    UniversalModelAPIHandler.max_body_bytes = max(1024, int(args.max_body_bytes))
    UniversalModelAPIHandler.allow_reload_endpoint = bool(args.enable_reload_endpoint)
    UniversalModelAPIHandler.rate_limit_enabled = bool(args.enable_rate_limit or (not _is_loopback_host(args.host)))
    UniversalModelAPIHandler.rate_limit_rps = max(1e-6, float(args.rate_limit_rps))
    UniversalModelAPIHandler.rate_limit_burst = max(1, int(args.rate_limit_burst))
    UniversalModelAPIHandler.rate_limit_state = {}
    UniversalModelAPIHandler.require_nonempty_memory = bool(args.require_nonempty_memory)
    UniversalModelAPIHandler.start_time_ms = int(time.time() * 1000)

    server = ThreadingHTTPServer((args.host, int(args.port)), UniversalModelAPIHandler)
    print(f"UniversalModel API listening on http://{args.host}:{args.port}")
    if bool(args.enable_reload_endpoint):
        print("Endpoints: GET /health, GET /ready, GET /metrics, POST /predict, POST /reload")
    else:
        print("Endpoints: GET /health, GET /ready, GET /metrics, POST /predict")
    print(
        f"Rate limit: enabled={bool(UniversalModelAPIHandler.rate_limit_enabled)} "
        f"rps={float(UniversalModelAPIHandler.rate_limit_rps):.3f} "
        f"burst={int(UniversalModelAPIHandler.rate_limit_burst)}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
