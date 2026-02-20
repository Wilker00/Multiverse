import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from models.universal_model import UniversalModel
from tools.universal_model_api import UniversalModelAPIHandler, _requires_api_key


def _write_model_dir(root: str, *, nonempty_memory: bool) -> str:
    model_dir = os.path.join(root, "model")
    memory_dir = os.path.join(model_dir, "central_memory")
    os.makedirs(memory_dir, exist_ok=True)

    row = {
        "run_id": "run_x",
        "episode_id": "0",
        "step_idx": 0,
        "t_ms": 0,
        "verse_name": "line_world",
        "action": 1,
        "reward": 1.0,
        "obs": [1, 2, 3],
        "obs_vector": [1.0, 2.0, 3.0],
    }
    with open(os.path.join(memory_dir, "memories.jsonl"), "w", encoding="utf-8") as f:
        if nonempty_memory:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(memory_dir, "dedupe_index.json"), "w", encoding="utf-8") as f:
        json.dump([], f)

    config = {
        "memory_dir": "central_memory",
        "default_top_k": 5,
        "default_min_score": 0.0,
        "default_verse_name": None,
        "meta_model_path": None,
        "meta_confidence_threshold": 0.35,
        "prefer_meta_policy": False,
        "meta_history_len": 6,
    }
    with open(os.path.join(model_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f)
    return model_dir


def _http_request(url: str, *, method: str = "GET", payload=None, headers=None):
    body = None
    req_headers = dict(headers or {})
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url=url, data=body, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read()
            text = raw.decode("utf-8")
            ctype = str(resp.headers.get("Content-Type", ""))
            data = json.loads(text) if "application/json" in ctype else text
            return resp.status, data
    except urllib.error.HTTPError as e:
        raw = e.read()
        text = raw.decode("utf-8")
        ctype = str(e.headers.get("Content-Type", ""))
        data = json.loads(text) if "application/json" in ctype else text
        return e.code, data


class TestUniversalModelAPI(unittest.TestCase):
    def _start_server(
        self,
        model_dir: str,
        *,
        api_key: str = "",
        require_nonempty_memory: bool = False,
        allow_reload_endpoint: bool = False,
        rate_limit_enabled: bool = False,
        rate_limit_rps: float = 20.0,
        rate_limit_burst: int = 40,
    ):
        class _Handler(UniversalModelAPIHandler):
            pass

        _Handler.model = UniversalModel.load(model_dir)
        _Handler.model_dir = model_dir
        _Handler.api_key = api_key or None
        _Handler.max_body_bytes = 1_000_000
        _Handler.require_nonempty_memory = require_nonempty_memory
        _Handler.allow_reload_endpoint = allow_reload_endpoint
        _Handler.rate_limit_enabled = rate_limit_enabled
        _Handler.rate_limit_rps = float(rate_limit_rps)
        _Handler.rate_limit_burst = int(rate_limit_burst)
        _Handler.rate_limit_state = {}

        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_address[1]}"
        return server, base_url

    def test_ready_fails_when_memory_empty_if_required(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = _write_model_dir(td, nonempty_memory=False)
            server, base = self._start_server(model_dir, require_nonempty_memory=True)
            try:
                code, payload = _http_request(f"{base}/ready")
                self.assertEqual(code, 503)
                self.assertFalse(payload["ok"])
                self.assertIn("memory_file_empty", payload["issues"])
            finally:
                server.shutdown()
                server.server_close()

    def test_predict_requires_auth_when_key_set(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = _write_model_dir(td, nonempty_memory=True)
            server, base = self._start_server(model_dir, api_key="secret")
            try:
                code, payload = _http_request(
                    f"{base}/predict",
                    method="POST",
                    payload={"obs": [1, 2, 3], "verse": "line_world"},
                )
                self.assertEqual(code, 401)
                self.assertFalse(payload["ok"])

                code_ok, payload_ok = _http_request(
                    f"{base}/predict",
                    method="POST",
                    payload={"obs": [1, 2, 3], "verse": "line_world"},
                    headers={"X-API-Key": "secret"},
                )
                self.assertEqual(code_ok, 200)
                self.assertTrue(payload_ok["ok"])
            finally:
                server.shutdown()
                server.server_close()

    def test_predict_validates_payload(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = _write_model_dir(td, nonempty_memory=True)
            server, base = self._start_server(model_dir)
            try:
                code, payload = _http_request(f"{base}/predict", method="POST", payload={})
                self.assertEqual(code, 400)
                self.assertFalse(payload["ok"])

                code2, payload2 = _http_request(
                    f"{base}/predict",
                    method="POST",
                    payload={"obs": "not_supported"},
                )
                self.assertEqual(code2, 400)
                self.assertFalse(payload2["ok"])
            finally:
                server.shutdown()
                server.server_close()

    def test_api_key_required_for_non_loopback_bindings(self):
        self.assertTrue(_requires_api_key("0.0.0.0", "", False))
        self.assertTrue(_requires_api_key("10.0.0.1", "", False))
        self.assertFalse(_requires_api_key("127.0.0.1", "", False))
        self.assertFalse(_requires_api_key("localhost", "", False))
        self.assertFalse(_requires_api_key("::1", "", False))
        self.assertFalse(_requires_api_key("0.0.0.0", "secret", False))
        self.assertFalse(_requires_api_key("0.0.0.0", "", True))

    def test_reload_endpoint_is_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = _write_model_dir(td, nonempty_memory=True)
            server, base = self._start_server(model_dir)
            try:
                code, payload = _http_request(f"{base}/reload", method="POST", payload={})
                self.assertEqual(code, 404)
                self.assertFalse(payload["ok"])
            finally:
                server.shutdown()
                server.server_close()

    def test_reload_endpoint_can_be_enabled(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = _write_model_dir(td, nonempty_memory=True)
            server, base = self._start_server(model_dir, allow_reload_endpoint=True)
            try:
                code, payload = _http_request(f"{base}/reload", method="POST", payload={})
                self.assertEqual(code, 200)
                self.assertTrue(payload["ok"])
            finally:
                server.shutdown()
                server.server_close()

    def test_predict_internal_errors_are_sanitized(self):
        class _BrokenModel:
            class _Cfg:
                memory_dir = "central_memory"

            config = _Cfg()

            def predict(self, **kwargs):
                _ = kwargs
                raise RuntimeError("db_password=supersecret")

        with tempfile.TemporaryDirectory() as td:
            model_dir = _write_model_dir(td, nonempty_memory=True)

            class _Handler(UniversalModelAPIHandler):
                pass

            _Handler.model = _BrokenModel()  # type: ignore[assignment]
            _Handler.model_dir = model_dir
            _Handler.api_key = None
            _Handler.max_body_bytes = 1_000_000
            _Handler.require_nonempty_memory = False
            _Handler.allow_reload_endpoint = False

            server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_address[1]}"
            try:
                code, payload = _http_request(
                    f"{base}/predict",
                    method="POST",
                    payload={"obs": [1, 2, 3], "verse": "line_world"},
                )
                self.assertEqual(code, 500)
                self.assertFalse(payload["ok"])
                self.assertEqual(str(payload.get("error")), "internal_error")
                self.assertNotIn("password", json.dumps(payload))
            finally:
                server.shutdown()
                server.server_close()

    def test_predict_rate_limit_returns_429(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = _write_model_dir(td, nonempty_memory=True)
            server, base = self._start_server(
                model_dir,
                rate_limit_enabled=True,
                rate_limit_rps=0.01,
                rate_limit_burst=1,
            )
            try:
                code1, payload1 = _http_request(
                    f"{base}/predict",
                    method="POST",
                    payload={"obs": [1, 2, 3], "verse": "line_world"},
                )
                self.assertEqual(code1, 200)
                self.assertTrue(payload1["ok"])

                code2, payload2 = _http_request(
                    f"{base}/predict",
                    method="POST",
                    payload={"obs": [1, 2, 3], "verse": "line_world"},
                )
                self.assertEqual(code2, 429)
                self.assertFalse(payload2["ok"])
                self.assertEqual(str(payload2.get("error")), "rate_limited")
                self.assertGreaterEqual(int(payload2.get("retry_after_ms", 0)), 1)
            finally:
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
