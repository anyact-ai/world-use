"""The pilot's model endpoint on the evaluation machine: it forwards each Responses request to the model's own
endpoint with the API key, which only this process holds, and refuses requests once the run's spending reaches its
cap.

Agents are given http://127.0.0.1:PORT/v1 and a placeholder key. Each request's token usage and estimated cost go
to ledger.jsonl; nothing of a request's or response's content is kept.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RESERVE_USD = 2.0  # a request is refused once the next one could take spending past the cap


class Ledger:
    def __init__(self, path: Path, cap_usd: float, rates: dict):
        self.path, self.cap, self.rates = path, cap_usd, rates  # rates: USD per million tokens
        self.spent, self.lock = 0.0, threading.Lock()

    def open(self) -> bool:
        with self.lock:
            return self.spent + RESERVE_USD <= self.cap

    def charge(self, status: int, body) -> None:
        usage = body.get("usage") if isinstance(body, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        tokens_in = int(usage.get("input_tokens") or 0)
        cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
        tokens_out = int(usage.get("output_tokens") or 0)
        cost = ((tokens_in - cached) * self.rates["input"] + cached * self.rates["cached_input"]
                + tokens_out * self.rates["output"]) / 1e6
        with self.lock:
            self.spent += cost
            row = dict(at=time.time(), status=status, model=body.get("model") if isinstance(body, dict) else None,
                       input_tokens=tokens_in, cached_input_tokens=cached, output_tokens=tokens_out,
                       usd=round(cost, 6), spent_usd=round(self.spent, 6))
            with self.path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")


def serve(port: int, upstream: str, key: str, ledger: Ledger, *, ready=None):
    upstream = upstream.rstrip("/")

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["content-length"]))
            if not self.path.endswith("/responses"):
                return self.reply(404, {"error": {"message": "only /responses is forwarded"}})
            if not ledger.open():
                # Not retryable for the agents' loops: the turn ends and the attempt with it.
                return self.reply(400, {"error": {"message": f"the run's spending cap of ${ledger.cap:g} is reached"}})
            request = urllib.request.Request(upstream + "/responses", data=body, method="POST",
                                             headers={"api-key": key, "content-type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=1800) as response:
                    status, payload = response.status, response.read()
            except urllib.error.HTTPError as error:
                status, payload = error.code, error.read()
            except (urllib.error.URLError, TimeoutError) as error:
                return self.reply(502, {"error": {"message": f"upstream unreachable: {type(error).__name__}"}})
            try:
                parsed = json.loads(payload)
            except ValueError:
                parsed = None
            if status == 200:
                ledger.charge(status, parsed)
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def reply(self, status, value):
            payload = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    if ready is not None:
        ready(server)
    server.serve_forever()


if __name__ == "__main__":
    import os
    import sys

    # The parent passes the key and the upstream in its environment for this process alone.
    port, ledger_path, cap = int(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3])
    rates = json.loads(sys.argv[4])
    serve(port, os.environ.pop("MODEL_UPSTREAM"), os.environ.pop("MODEL_KEY"), Ledger(ledger_path, cap, rates))
