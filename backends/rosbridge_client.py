"""Minimal rosbridge v2 client (JSON over a websocket) built on ``websocket-client``.

Replaces ``roslibpy``: no Twisted/autobahn stack, one small pure-Python dependency that Hermes already ships, no
reactor thread. It implements only what hermes-ros needs: call_service, subscribe/unsubscribe, advertise/publish.
Protocol reference: https://github.com/RobotWebTools/rosbridge_suite/blob/ros2/ROSBRIDGE_PROTOCOL.md
"""
from __future__ import annotations

import itertools
import json
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple


class RosbridgeError(RuntimeError):
    pass


class RosbridgeClient:
    def __init__(self, host: str, port: int) -> None:
        self.url = f"ws://{host}:{port}"
        self._ws: Any = None
        self._thread: Optional[threading.Thread] = None
        self._opened = threading.Event()
        self._closed = threading.Event()
        self._open_error: Optional[BaseException] = None
        self._send_lock = threading.Lock()
        self._ids = itertools.count(1)
        self._pending: Dict[str, Tuple[threading.Event, Dict[str, Any]]] = {}
        self._subs: Dict[str, Dict[str, Callable[[Dict[str, Any]], None]]] = {}  # topic -> {sub_id: callback}
        self._advertised: Dict[str, str] = {}

    # ------------------------------------------------------------ lifecycle
    @property
    def is_connected(self) -> bool:
        return self._opened.is_set() and not self._closed.is_set()

    def connect(self, timeout: float = 6.0) -> None:
        import websocket  # websocket-client; lazy so plugin registration stays offline and import-cheap

        def on_open(_ws: Any) -> None:
            self._opened.set()

        def on_error(_ws: Any, err: Any) -> None:
            if not self._opened.is_set():
                self._open_error = err if isinstance(err, BaseException) else RuntimeError(str(err))
                self._closed.set()
                self._opened.set()

        def on_close(_ws: Any, *_a: Any) -> None:
            self._closed.set()
            for ev, _slot in list(self._pending.values()):
                ev.set()  # fail pending calls fast instead of letting them wait out their timeout

        self._ws = websocket.WebSocketApp(self.url, on_open=on_open, on_message=self._on_message,
                                          on_error=on_error, on_close=on_close)
        self._thread = threading.Thread(target=lambda: self._ws.run_forever(ping_interval=20, ping_timeout=10),
                                        name="hermes-ros-rosbridge", daemon=True)
        self._thread.start()
        if not self._opened.wait(timeout):
            self.close()
            raise RosbridgeError(f"timed out after {timeout:.0f}s connecting to {self.url}")
        if self._open_error is not None:
            raise RosbridgeError(f"{type(self._open_error).__name__}: {self._open_error}")

    def close(self) -> None:
        self._closed.set()
        try:
            if self._ws is not None:
                for topic in list(self._advertised):
                    self._send({"op": "unadvertise", "topic": topic}, quiet=True)
                self._ws.close()
        except Exception:
            pass

    # ------------------------------------------------------------- plumbing
    def _send(self, msg: Dict[str, Any], quiet: bool = False) -> None:
        if not self.is_connected and not quiet:
            raise RosbridgeError(f"not connected to {self.url}")
        try:
            with self._send_lock:
                self._ws.send(json.dumps(msg))
        except Exception as exc:
            if not quiet:
                raise RosbridgeError(f"send failed: {exc}") from exc

    def _on_message(self, _ws: Any, raw: Any) -> None:
        try:
            msg = json.loads(raw)
        except Exception:
            return
        op = msg.get("op")
        if op == "publish":
            for cb in list(self._subs.get(msg.get("topic", ""), {}).values()):
                try:
                    cb(msg.get("msg") or {})
                except Exception:
                    pass
        elif op == "service_response":
            entry = self._pending.get(msg.get("id", ""))
            if entry is not None:
                entry[1].update(msg)
                entry[0].set()

    # ------------------------------------------------------------------ API
    def call_service(self, name: str, srv_type: Optional[str], args: Optional[Dict[str, Any]] = None,
                     timeout: float = 8.0) -> Dict[str, Any]:
        cid = f"call_service:{name}:{next(self._ids)}"
        ev, slot = threading.Event(), {}
        self._pending[cid] = (ev, slot)
        try:
            req: Dict[str, Any] = {"op": "call_service", "id": cid, "service": name, "args": args or {}}
            if srv_type:
                req["type"] = srv_type
            self._send(req)
            if not ev.wait(timeout) or not slot:
                raise RosbridgeError(f"no response from {name} within {timeout:.0f}s"
                                     if self.is_connected else f"connection to {self.url} lost")
        finally:
            self._pending.pop(cid, None)
        if not slot.get("result", True):
            raise RosbridgeError(f"service {name} reported failure: {slot.get('values')}")
        values = slot.get("values")
        return values if isinstance(values, dict) else {}

    def subscribe(self, topic: str, msg_type: Optional[str], callback: Callable[[Dict[str, Any]], None],
                  **opts: Any) -> str:
        sid = f"subscribe:{topic}:{next(self._ids)}"
        self._subs.setdefault(topic, {})[sid] = callback
        req: Dict[str, Any] = {"op": "subscribe", "id": sid, "topic": topic}
        if msg_type:
            req["type"] = msg_type
        req.update({k: v for k, v in opts.items() if v is not None})
        self._send(req)
        return sid

    def unsubscribe(self, topic: str, sid: Optional[str] = None) -> None:
        subs = self._subs.get(topic, {})
        subs.pop(sid, None) if sid else subs.clear()
        if not subs:
            self._subs.pop(topic, None)
        self._send({"op": "unsubscribe", "id": sid or f"unsubscribe:{topic}", "topic": topic}, quiet=True)

    def publish(self, topic: str, msg_type: str, message: Dict[str, Any]) -> None:
        if self._advertised.get(topic) != msg_type:
            self._send({"op": "advertise", "id": f"advertise:{topic}:{next(self._ids)}", "topic": topic, "type": msg_type})
            self._advertised[topic] = msg_type
        self._send({"op": "publish", "id": f"publish:{topic}:{next(self._ids)}", "topic": topic, "msg": message})
