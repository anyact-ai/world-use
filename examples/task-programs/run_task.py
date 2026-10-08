"""Run one frozen Python task, offline or against an already-running simulation."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import traceback
from pathlib import Path

# Also needed with python -I when executing the snapshotted launcher.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_record import Record, RecordedClient, prepare


def launch(task: Path, output: Path, *, parameters: dict, sources=(), url=None, parent=None) -> int:
    task = task.resolve()
    root = task.parent
    paths = [task, *[Path(source).resolve() for source in sources]]
    if any(not path.is_relative_to(root) for path in paths):
        raise ValueError("declared sources must be inside the task's directory")
    bundle = {str(path.relative_to(root)): path for path in paths}
    if any(Path(name).parts[0] == "__runner__" for name in bundle):
        raise ValueError("__runner__ is reserved for the frozen launcher")
    for name in ("run_task.py", "task_record.py"):
        bundle[f"__runner__/{name}"] = Path(__file__).with_name(name)
    output = prepare(output, bundle, entrypoint=task.name, parameters=parameters, url=url, parent=parent)
    print(f"invocation: {output}", flush=True)
    # Never import the mutable original task. A killed launcher must not cause an automatic retry.
    return subprocess.run([sys.executable, "-I", str(output / "source/__runner__/run_task.py"),
                           "--execute", str(output)], cwd=output / "source", check=False).returncode


def execute(output: Path) -> int:
    record = Record(output)
    manifest = record.manifest
    client = None
    result = None
    failure = None
    final_status = None
    try:
        if manifest["url"]:
            client = RecordedClient(manifest["url"], record)
            client.begin()
        path = record.root / "source" / manifest["entrypoint"]
        sys.path[:0] = [str(path.parent), str(record.root / "source")]
        spec = importlib.util.spec_from_file_location("task_program", path)
        if spec is None or spec.loader is None:
            raise ValueError("entrypoint must be a Python module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        record.append("task_start", entrypoint=manifest["entrypoint"])
        returned = module.run(client, manifest["parameters"], record)
        if not isinstance(returned, dict) or returned.get("verdict") not in ("pass", "fail", "inconclusive"):
            raise ValueError("run(robot, params, record) must return a dict with an explicit task verdict")
        result = returned
    except BaseException:
        failure = traceback.format_exc()
        print(failure, file=sys.stderr)
    finally:
        if client is not None:
            try:
                final_status = client.status()
                if client.flight is not None:
                    if final_status["recording"]["path"] != client.flight:
                        raise RuntimeError("daemon record changed; do not attribute the new session to this task")
                    client.record(context=dict(invocation=manifest["invocation"], phase="end",
                                               revision=manifest["revision"],
                                               verdict=None if result is None else result["verdict"],
                                               exception=failure))
                record.append("end", status=final_status)
            except Exception:
                failure = (failure or "") + traceback.format_exc()
        saved = record.finish(result, status=final_status, exception=failure)
    ok = result is not None and result["verdict"] == "pass"
    ok &= not (failure or saved["recording_error"] or saved["flight_recording_error"])
    if client is not None:
        ok &= (final_status is not None and final_status["enabled"] is False
               and final_status["power_uncertain"] is False
               and not (final_status.get("job") or final_status.get("queued")))
    print(f"{'passed' if ok else 'needs attention'}: {record.root / 'result.json'}", flush=True)
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--params", type=Path, help="JSON parameters; use absolute paths for external inputs")
    parser.add_argument("--source", action="append", type=Path, default=[], help="additional local source/config file")
    parser.add_argument("--url", help="explicit URL of an already-running simulation; no daemon is started")
    parser.add_argument("--parent", help="failed/previous invocation ID or record reference")
    parser.add_argument("--execute", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.execute is not None:
        return execute(args.execute)
    if args.task is None or args.output is None:
        parser.error("task and --output are required")
    parameters = {} if args.params is None else json.loads(args.params.read_text())
    if not isinstance(parameters, dict):
        parser.error("--params must contain a JSON object")
    return launch(args.task, args.output, parameters=parameters, sources=args.source, url=args.url, parent=args.parent)


if __name__ == "__main__":
    raise SystemExit(main())
