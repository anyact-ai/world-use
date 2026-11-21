"""A stand-in Responses endpoint that plays a fixed list of tool calls, to check a setup end to end without a paid
model: each request gets the next call, and once the list is spent it answers in text with no call.

    python scripted_model.py PORT PLAN.json      PLAN.json: [{"name": ..., "arguments": {...}}, ...]

Every request's tool names and latest tool result are appended to requests.jsonl beside PLAN.json.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def serve(port: int, plan_path: Path):
    plan = json.loads(plan_path.read_text())
    log = plan_path.with_name("requests.jsonl")
    played = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            outputs = [item for item in body.get("input", []) if item.get("type") == "function_call_output"]
            with log.open("a") as stream:
                stream.write(json.dumps(dict(tools=[t["name"] for t in body.get("tools", [])],
                                             last_output=outputs[-1]["output"] if outputs else None)) + "\n")
            index = len(played)
            played.append(index)
            if body.get("tool_choice") == "none" or index >= len(plan):
                output = [dict(type="message", role="assistant",
                               content=[dict(type="output_text", text="The scripted plan is over.")])]
            else:
                output = [dict(type="function_call", call_id=f"call_{index}", name=plan[index]["name"],
                               arguments=json.dumps(plan[index]["arguments"]))]
            reply = json.dumps(dict(id=f"resp_{index}", model="scripted", output=output,
                                    usage=dict(input_tokens=0, output_tokens=0, total_tokens=0))).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)

        def log_message(self, *args):
            pass

    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    serve(int(sys.argv[1]), Path(sys.argv[2]))
