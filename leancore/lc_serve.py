#!/usr/bin/env python3
"""lc_serve.py — минимальный OpenAI-подобный HTTP-сервис поверх lc_stream.
POST /v1/completions {"prompt": str, "max_tokens": int, "temperature": float, "top_k": int}
GET  /health"""
import json, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from lc_repl import Engine, tok, detok

import os
MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/champ3_qat_kl8.lcw2")
ENG = None
LOCK = threading.Lock()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, code, obj):
        b = json.dumps(obj).encode(); self.send_response(code)
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health": self._json(200, {"status": "ok", "model": MODEL})
        else: self._json(404, {"error": "unknown"})
    def do_POST(self):
        if self.path != "/v1/completions": return self._json(404, {"error": "unknown"})
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with LOCK:
            ENG.reset()
            if "tau" in body: ENG.tau(float(body["tau"]))
            ENG.step([1] + tok(body.get("prompt", "")))
            ids = ENG.gen(int(body.get("max_tokens", 64)), float(body.get("temperature", 0.8)),
                          int(body.get("top_k", 40)))
        self._json(200, {"choices": [{"text": detok(ids), "finish_reason": "length"}],
                         "usage": {"completion_tokens": len(ids)}})

if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    if len(sys.argv) > 2: MODEL = sys.argv[2]
    ENG = Engine(MODEL)
    print(f"lc_serve :{port} model={MODEL}")
    HTTPServer(("0.0.0.0", port), H).serve_forever()
