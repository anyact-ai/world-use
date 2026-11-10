"""Fail-closed, single-process Azure transport with a durable reservation journal."""

from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from PIL import Image

BASE_URL = os.environ.get("LIBERO_AZURE_BASE_URL", "https://example.openai.azure.com/openai/v1").rstrip("/")
HOST = urlsplit(BASE_URL).hostname
MODEL = "gpt-6.1-sol"
# Published LLM_LIBERO price row; estimates, not an Azure billing reconciliation.
RATES = dict(input=2.0, cached_input=0.1, conservative_input=2.5, output=10.0)


class BudgetStop(RuntimeError):
    pass


def estimate_input(body):
    images = 0

    def walk(value):
        nonlocal images
        if isinstance(value, dict):
            if value.get("type") == "input_image":
                url = value.get("image_url", "")
                if not url.startswith("data:image/png;base64,"):
                    raise BudgetStop("only inline PNG observations are allowed")
                raw = base64.b64decode(url.split(",", 1)[1], validate=True)
                with Image.open(io.BytesIO(raw)) as image:
                    if image.size != (256, 256):
                        raise BudgetStop("unexpected image size")
                images += 1
                value["detail"] = "high"
                return {"type": "input_image", "image_url": "[image]", "detail": "high"}
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    text = walk(body)
    if images > 4:
        raise BudgetStop("more than two observations of two cameras")
    tokens = len(json.dumps(text, ensure_ascii=False).encode()) + images * 8192 + 4096
    if tokens > 200000:
        raise BudgetStop("conservative short-context ceiling")
    return tokens


class AzureBudget(httpx.BaseTransport):
    def __init__(
        self, folder: Path, *, key: str, total=15.0, per_arm=7.4, seconds=7200, inner=None, max_output_tokens=3072
    ):
        folder.mkdir(parents=True, exist_ok=False)
        endpoint = urlsplit(BASE_URL)
        if (
            endpoint.scheme != "https"
            or endpoint.path != "/openai/v1"
            or endpoint.query
            or endpoint.username
            or endpoint.password
            or endpoint.port not in (None, 443)
            or not endpoint.hostname.endswith(".openai.azure.com")
        ):
            raise ValueError("LIBERO_AZURE_BASE_URL must name an HTTPS Azure OpenAI /openai/v1 endpoint")
        self.folder, self.key = folder, key
        self.total, self.per_arm = total, per_arm
        self.deadline = time.monotonic() + seconds
        self.max_output_tokens = max_output_tokens
        # An explicit HTTPTransport otherwise ignores HTTPX's client proxy defaults.
        self.inner = inner or httpx.HTTPTransport(
            proxy=os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"), retries=0
        )
        self.used = self.estimated = 0.0
        self.arm_used = self.arm_estimated = 0.0
        self.arm = None
        self.calls = self.arm_calls = 0
        self.uncertain = False

    def begin(self, arm):
        if self.uncertain:
            raise BudgetStop("previous request usage unknown")
        self.arm, self.arm_used, self.arm_estimated, self.arm_calls = arm, 0.0, 0.0, 0

    def record(self, **row):
        with (self.folder / "ledger.jsonl").open("a") as f:
            f.write(json.dumps(dict(at=time.time(), arm=self.arm, **row)) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def handle_request(self, request):
        if self.uncertain or self.arm is None:
            raise BudgetStop("unknown usage or no active arm")
        if time.monotonic() >= self.deadline:
            raise BudgetStop("campaign deadline")
        if self.arm_calls >= (1 if self.arm == "preflight" else 80):
            raise BudgetStop("call limit")
        if (request.method, request.url.scheme, request.url.host, request.url.path) != (
            "POST",
            "https",
            HOST,
            "/openai/v1/responses",
        ) or request.url.query:
            raise BudgetStop("endpoint not allowed")
        body = json.loads(request.read())
        if body.get("model") != MODEL or any(t.get("type") != "function" for t in body.get("tools", [])):
            raise BudgetStop("model or tool type not allowed")
        body.update(store=False, max_output_tokens=self.max_output_tokens)
        body.pop("service_tier", None)
        tokens = estimate_input(body)
        reserve = (tokens * RATES["conservative_input"] + self.max_output_tokens * RATES["output"]) / 1e6
        arm_limit = 0.15 if self.arm == "preflight" else self.per_arm
        if self.used + reserve > self.total or self.arm_used + reserve > arm_limit:
            raise BudgetStop("request would exceed budget")
        self.calls += 1
        self.arm_calls += 1
        ident = f"{self.calls:04d}-{self.arm}"
        (self.folder / f"{ident}-request.json").write_text(json.dumps(body))
        self.used += reserve
        self.arm_used += reserve
        self.uncertain = True
        self.record(event="reserved", request=ident, reserved_usd=reserve, total_charge=self.used)
        started = time.monotonic()
        try:
            remaining = max(0.1, self.deadline - started)
            outgoing = httpx.Request(
                "POST",
                str(request.url),
                json=body,
                headers={"api-key": self.key, "Content-Type": "application/json"},
                extensions={
                    "timeout": dict(
                        connect=min(15.0, remaining),
                        read=min(120.0, remaining),
                        write=min(30.0, remaining),
                        pool=min(15.0, remaining),
                    )
                },
            )
            response = self.inner.handle_request(outgoing)
            response.read()
            if response.status_code != 200:
                self.record(event="usage_unknown", request=ident, http_status=response.status_code)
                raise BudgetStop(f"HTTP {response.status_code}; no retry, reservation retained")
            payload = response.json()
            (self.folder / f"{ident}-response.json").write_bytes(response.content)
            usage = payload.get("usage") or {}
            inp, out = usage.get("input_tokens"), usage.get("output_tokens")
            cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
            if any(type(n) is not int or n < 0 for n in (inp, out, cached)) or cached > inp:
                raise BudgetStop("invalid or missing usage; reservation retained")
            if inp > tokens or out > self.max_output_tokens:
                raise BudgetStop("actual usage exceeded reservation; campaign stopped")
            charge = (inp * RATES["conservative_input"] + out * RATES["output"]) / 1e6
            cost = ((inp - cached) * RATES["input"] + cached * RATES["cached_input"] + out * RATES["output"]) / 1e6
            self.used += charge - reserve
            self.arm_used += charge - reserve
            self.estimated += cost
            self.arm_estimated += cost
            self.uncertain = False
            self.record(
                event="settled",
                request=ident,
                usage=usage,
                charge_usd=charge,
                estimated_usd=cost,
                total_charge=self.used,
                latency_s=time.monotonic() - started,
                response_status=payload.get("status"),
            )
            if payload.get("status") != "completed":
                raise BudgetStop("response incomplete; do not execute partial output")
            return response
        except Exception as exc:
            self.record(event="stopped", request=ident, error_type=type(exc).__name__, unknown_usage=self.uncertain)
            raise BudgetStop(str(exc) if isinstance(exc, BudgetStop) else "request failed; usage unknown") from None

    def close(self):
        pass  # Each policy owns a client; the campaign owns this shared transport.

    def dispose(self):
        self.inner.close()
