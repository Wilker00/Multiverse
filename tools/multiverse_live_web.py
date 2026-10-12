"""
tools/multiverse_live_web.py

Local browser view of the same live snapshot the terminal map uses.

Stdlib HTTP only. The page polls ``/api/snapshot`` about once a second.
This is a product surface for the shared intelligence layer, not a training UI.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

if __package__ in (None, ""):
    _ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)

from core.live_state import build_live_snapshot


_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Multiverse live map</title>
<style>
  body { margin: 0; font: 14px/1.4 ui-sans-serif, Segoe UI, sans-serif; background: #0e141b; color: #d7e2ea; }
  header { padding: 16px 20px; border-bottom: 1px solid #243140; }
  h1 { margin: 0; font-size: 18px; letter-spacing: 0.04em; }
  main { display: grid; grid-template-columns: 280px 1fr 320px; gap: 16px; padding: 16px; }
  section { background: #15202b; border: 1px solid #243140; border-radius: 8px; padding: 12px; }
  h2 { margin: 0 0 8px; font-size: 13px; color: #8fb4c9; text-transform: uppercase; }
  .verse { padding: 6px 0; border-bottom: 1px solid #243140; }
  .active { color: #7ddea0; }
  .idle { color: #8aa0b2; }
  .edge { font-family: ui-monospace, Consolas, monospace; font-size: 12px; padding: 4px 0; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header>
  <h1>Multiverse live map</h1>
  <div id="summary">loading</div>
</header>
<main>
  <section><h2>Central memory</h2><div id="memory"></div></section>
  <section><h2>Verses</h2><div id="verses"></div></section>
  <section><h2>Flows</h2><div id="flows"></div></section>
</main>
<script>
async function tick() {
  const res = await fetch("/api/snapshot");
  const data = await res.json();
  const counts = data.counts || {};
  const memory = data.memory || {};
  document.getElementById("summary").textContent =
    "verses " + (counts.registered_verses || 0) +
    " | active runs " + (counts.active_runs || 0) +
    " | transfers " + (counts.recent_transfers || 0);
  document.getElementById("memory").textContent =
    (memory.exists ? "bank present" : "bank missing") +
    " | rows " + (memory.row_count || 0) +
    " | bytes " + (memory.bytes || 0);
  const verses = (data.verses || []).filter(function (row) {
    return (row.run_count || 0) > 0 || (row.recall_count || 0) > 0 ||
      (row.transfer_count || 0) > 0 || (row.safety_count || 0) > 0;
  }).slice(0, 24);
  const shown = verses.length ? verses : (data.verses || []).slice(0, 8);
  document.getElementById("verses").innerHTML = shown.map(function (row) {
    const tags = (row.concept_tags || []).join(", ");
    return '<div class="verse ' + row.status + '">' + row.verse_name +
      " [" + row.status + "] " + tags +
      " runs=" + (row.run_count || 0) +
      " reward=" + Number(row.reward_sum || 0).toFixed(2) + "</div>";
  }).join("");
  const flows = data.flows || {};
  const lines = [];
  (flows.retrievals || []).forEach(function (row) {
    lines.push('<div class="edge">recall ' + row.verse_name + " matches=" + row.match_count + "</div>");
  });
  (flows.transfers || []).forEach(function (row) {
    lines.push('<div class="edge">transfer ' + row.source_verse + " -> " + row.target_verse + "</div>");
  });
  (flows.safety || []).forEach(function (row) {
    lines.push('<div class="edge">safety ' + row.verse_name + " " + row.mode + "</div>");
  });
  document.getElementById("flows").innerHTML = lines.join("") || '<div class="edge">no recent traffic</div>';
}
tick();
setInterval(tick, 1000);
</script>
</body>
</html>
"""


def render_page() -> str:
    return _PAGE


def make_handler(runs_root: str, memory_root: str):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(int(code))
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = str(self.path or "/").split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, render_page().encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/snapshot":
                payload = build_live_snapshot(runs_root=runs_root, memory_root=memory_root)
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
                return
            self._send(404, b"not found", "text/plain; charset=utf-8")

    return Handler


def serve(
    *,
    runs_root: str = "runs",
    memory_root: str = "central_memory",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    server = ThreadingHTTPServer((str(host), int(port)), make_handler(str(runs_root), str(memory_root)))
    print(f"Multiverse live map at http://{host}:{int(port)}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nLive map closed.")
    finally:
        server.server_close()


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Browser view of the multiverse live snapshot.")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--memory-root", default="central_memory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    serve(
        runs_root=str(args.runs_root),
        memory_root=str(args.memory_root),
        host=str(args.host),
        port=int(args.port),
    )


if __name__ == "__main__":
    main()
