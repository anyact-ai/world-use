"""Optional, bounded model process owned by a procedure or MCP client, never by the kernel."""
from __future__ import annotations

import multiprocessing
import os
import pickle
import signal
import threading

from .errors import Refused
from .procedures import positive


def _serve(conn, options):
    from .vision import MODEL, REVISION, EdgeTAM
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    parent = multiprocessing.parent_process()
    if parent is not None:
        def orphaned():
            parent.join()
            os._exit(0)
        threading.Thread(target=orphaned, daemon=True).start()
    try:
        import torch  # ty: ignore[unresolved-import]
        torch.set_num_threads(2)
        root = EdgeTAM(**options)
    except Exception as e:
        conn.send((False, str(e)))
        conn.close()
        return
    targets = {}
    conn.send((True, dict(provider="EdgeTAM", device=root.device, model=MODEL,
                         revision=REVISION if options.get("model_path") is None else None,
                         local_model=options.get("model_path") is not None)))
    try:
        while (request := conn.recv()) is not None:
            action, identity, frame, prompt = request
            try:
                if action == "forget":
                    old = targets.pop(identity, None)
                    if old is not None:
                        old.close()
                    result = None
                elif action == "select":
                    if len(targets) >= 4:
                        raise ValueError("at most four active targets")
                    tracker = root.fork()
                    result = tracker.select(frame, **prompt)
                    targets[identity] = tracker
                elif action == "update":
                    result = targets[identity].update(frame)
                else:
                    raise ValueError("unknown tracking operation")
                conn.send((True, result))
            except Exception as e:
                conn.send((False, str(e)))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        for target in targets.values():
            target.close()
        root.close()
        conn.close()


class TrackerProcess:
    """Load once before powered work; one in-flight inference and four independent target histories."""

    def __init__(self, *, device="cpu", model_path=None, max_age_s=15, timeout_s=30, startup_timeout_s=120):
        self.timeout_s = positive(timeout_s, "timeout_s")
        startup_timeout_s = positive(startup_timeout_s, "startup_timeout_s")
        self.lock = threading.Lock()
        self.ready = False
        ctx = multiprocessing.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.process = ctx.Process(target=_serve, args=(child, dict(device=device, model_path=model_path,
                                                                   max_age_s=max_age_s)), daemon=True)
        self.process.start()
        child.close()
        if not self.conn.poll(startup_timeout_s):
            self.close()
            raise Refused("vision provider did not finish loading", "provider_timeout")
        try:
            ok, result = self.conn.recv()
        except EOFError:
            self.close()
            raise Refused("vision provider exited during loading", "provider_unavailable") from None
        if not ok:
            self.close()
            raise Refused(f"vision provider failed to load: {result}", "provider_unavailable")
        self.info = result
        self.ready = True

    def _call(self, action, target, frame=None, **prompt):
        if not self.lock.acquire(blocking=False):
            raise Refused("another perception request is running", "busy", "wait for its result")
        expired = threading.Event()

        def deadline():
            expired.set()
            # Killing the reader also releases a send blocked on a full image pipe.
            if self.process.is_alive():
                self.process.kill()

        timer = threading.Timer(self.timeout_s, deadline)
        timer.daemon = True
        try:
            if not self.ready:
                raise Refused("vision provider is closed; restart it and reselect targets", "provider_unavailable")
            # The model needs RGB and identity, not a second copy of depth or geometric feedback.
            if frame is not None:
                from .cameras import Frame
                frame = Frame(frame.image, frame.camera, frame.id, frame.timestamp,
                              session=frame.session, calibration=frame.calibration)
            timer.start()
            self.conn.send_bytes(pickle.dumps((action, target, frame, prompt)))
            if not self.conn.poll(self.timeout_s):
                self.close(force=True)
                raise Refused("perception deadline exceeded; targets require reselection", "provider_timeout")
            ok, result = self.conn.recv()
            if expired.is_set():
                self.close(force=True)
                raise Refused("perception deadline exceeded; targets require reselection", "provider_timeout")
            if not ok:
                raise Refused(f"perception failed: {result}", "provider_error")
            return result
        except (EOFError, BrokenPipeError, OSError):
            self.close()
            if expired.is_set():
                raise Refused("perception deadline exceeded; targets require reselection", "provider_timeout") from None
            raise Refused("vision provider exited; restart it and reselect targets", "provider_unavailable") from None
        finally:
            timer.cancel()
            if timer.ident is not None:
                timer.join()
            self.lock.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def select(self, target, frame, **prompt):
        return self._call("select", target, frame, **prompt)

    def update(self, target, frame):
        return self._call("update", target, frame)

    def forget(self, target):
        return self._call("forget", target)

    def close(self, *, force=False):
        graceful = not force and self.ready and self.lock.acquire(blocking=False)
        self.ready = False
        try:
            if self.process.is_alive() and graceful:
                try:
                    self.conn.send(None)
                    self.process.join(timeout=3)
                except (EOFError, BrokenPipeError, OSError):
                    pass
            if self.process.is_alive():
                self.process.kill()
            self.process.join(timeout=2)
            self.conn.close()
        finally:
            if graceful:
                self.lock.release()
