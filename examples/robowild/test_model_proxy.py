"""model_proxy.py against a stand-in model endpoint.

    uv run pytest examples/robowild/test_model_proxy.py
"""

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from model_proxy import Ledger, serve

USAGE = dict(input_tokens=100_000, input_tokens_details=dict(cached_tokens=50_000), output_tokens=10_000)
RATES = dict(input=8.0, cached_input=0.4, output=40.0)  # 50k fresh, 50k cached, 10k out: $0.82 a request


@pytest.fixture
def endpoint(tmp_path):
    seen = []

    class Model(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(dict(path=self.path, key=self.headers.get("api-key"),
                             body=self.rfile.read(int(self.headers["content-length"]))))
            payload = json.dumps(dict(model="m", output=[], usage=USAGE)).encode()
            self.send_response(200)
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    model = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    threading.Thread(target=model.serve_forever, daemon=True).start()
    ledger = Ledger(tmp_path / "ledger.jsonl", cap_usd=4.0, rates=RATES)
    servers = []
    upstream = f"http://127.0.0.1:{model.server_address[1]}/openai/v1"
    threading.Thread(target=serve, args=(0, upstream, "secret-key", ledger), kwargs=dict(ready=servers.append),
                     daemon=True).start()
    while not servers:
        pass
    yield f"http://127.0.0.1:{servers[0].server_address[1]}/v1/responses", seen, ledger
    servers[0].shutdown()
    model.shutdown()


def post(url, body=b'{"input": "the agent\'s request"}'):
    request = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Authorization": "Bearer local", "content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_requests_reach_the_model_with_its_key_and_stop_at_the_spending_cap(endpoint):
    url, seen, ledger = endpoint
    statuses = [post(url)[0] for _ in range(4)]
    # $0.82 each: the third leaves $1.54, less than the $2 a next request is allowed to take.
    assert statuses == [200, 200, 200, 400]
    assert len(seen) == 3 and all(r["key"] == "secret-key" and r["path"] == "/openai/v1/responses" for r in seen)
    assert ledger.spent == pytest.approx(3 * 0.82)


def test_the_ledger_keeps_usage_and_cost_and_nothing_of_the_conversation(endpoint):
    url, _, ledger = endpoint
    post(url)
    (row,) = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    assert row["usd"] == pytest.approx(0.82) and row["cached_input_tokens"] == 50_000
    assert "the agent" not in ledger.path.read_text() and "secret-key" not in ledger.path.read_text()


def test_instructions_reach_the_model_as_a_leading_developer_message(endpoint):
    url, seen, _ = endpoint
    user = dict(type="message", role="user", content=[dict(type="input_text", text="go")])
    post(url, json.dumps(dict(model="m", instructions="the rules", input=[user], store=False)).encode())
    forwarded = json.loads(seen[0]["body"])
    assert "instructions" not in forwarded and forwarded["store"] is False
    assert forwarded["input"] == [dict(type="message", role="developer",
                                       content=[dict(type="input_text", text="the rules")]), user]
