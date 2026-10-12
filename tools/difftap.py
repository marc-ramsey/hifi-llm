#!/usr/bin/env python3
"""difftap — a byte-exact HTTP/1.1 tap for finding out what a proxy really does.

It exists to answer one question without guessing: when traffic goes through
hifi instead of straight to llama.cpp, is anything actually different — the
request bytes, the response bytes, the SSE framing, or just the timing?  Those
are three different bugs and they need different fixes.

Run two taps, one per arm of the A/B, and compare:

    # arm A — client straight to the model
    tools/difftap.py serve --listen 127.0.0.1:8099 \\
        --upstream 127.0.0.1:8082 --log /tmp/tap-direct.jsonl --label direct

    # arm B — client through hifi
    tools/difftap.py serve --listen 127.0.0.1:8098 \\
        --upstream 127.0.0.1:8082 --log /tmp/tap-hifi.jsonl --label hifi \\
        --tap-request-only
    #   …point hifi's model url at 127.0.0.1:8098 so the tap sees the bytes
    #   hifi actually puts on the wire, and arm A's client at 127.0.0.1:8099

    # what is different?
    tools/difftap.py compare /tmp/tap-direct.jsonl /tmp/tap-hifi.jsonl

Use ``--capture-bodies`` to keep the exact bytes; ``compare`` then reports the
offset of the first differing byte, not just "hashes differ".

Two readings matter most:
  * request hash differs        → the proxy re-serialises (B1/B2 territory)
  * request identical, response
    hash or frame count differs → the proxy authors stream content (B3–B5)
  * everything identical, only
    timings differ              → lifecycle/backpressure, not content (B6)

Caveats, deliberately narrow: one request per connection (``Connection: close``
is forced on both sides), HTTP/1.1 only, no pipelining, no TLS to the
upstream.  It is a diagnostic tap, not a production proxy — do not put it in
the path you are shipping.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path

MAX_PRINT = 2000  # bytes of a body echoed into the JSONL record


# ────────────────────────────────────────────────────────────────────────
# serve
# ────────────────────────────────────────────────────────────────────────

class Tapper:
    def __init__(self, upstream_host: str, upstream_port: int, log_path: Path,
                 label: str, capture_bodies: bool, tap_request_only: bool) -> None:
        self.upstream = (upstream_host, upstream_port)
        self.log_path = log_path
        self.label = label
        self.capture_bodies = capture_bodies
        self.tap_request_only = tap_request_only
        self.seq = 0
        self.bodies_dir = log_path.with_suffix(log_path.suffix + ".bodies")
        self.bodies_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    # ── recording ───────────────────────────────────────────────────────

    def _record(self, rec: dict) -> None:
        rec.setdefault("label", self.label)
        rec.setdefault("ts", round(time.time(), 3))
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with self.log_path.open("a") as fh:
            fh.write(line + "\n")

    def _store_body(self, direction: str, data: bytes) -> dict:
        digest = hashlib.sha256(data).hexdigest()
        out = {"sha256": digest, "len": len(data)}
        printable = data[:MAX_PRINT].decode("utf-8", errors="replace")
        out["preview"] = printable
        if self.capture_bodies:
            self.bodies_dir.mkdir(parents=True, exist_ok=True)
            path = self.bodies_dir / f"{self.seq:05d}-{direction}.bin"
            path.write_bytes(data)
            out["file"] = str(path)
        return out

    # ── connection handling ─────────────────────────────────────────────

    async def handle(self, reader: asyncio.StreamReader, writer) -> None:
        self.seq += 1
        seq = self.seq
        rec: dict = {"seq": seq, "_t0": time.monotonic()}
        up_reader = up_writer = None
        try:
            head = await _read_head(reader)
            if head is None:
                return
            req_head, _, rest = head.partition(b"\r\n\r\n")
            req_body = await _read_body(reader, req_head, rest)
            lines = req_head.split(b"\r\n")
            rec["method"], rec["path"], _ = (lines[0].decode("latin-1").split(" ", 2) + [""])[:3]
            rec["req_headers"] = _header_map(lines[1:])
            rec["request"] = self._store_body("request", req_body)

            # ── forward upstream (verbatim bytes) ───────────────────────
            up_reader, up_writer = await asyncio.open_connection(*self.upstream)
            up_writer.write(req_head + b"\r\n\r\n" + req_body)
            await up_writer.drain()
            t_sent = time.monotonic()

            # ── relay back, capturing raw bytes (framing included) ──────
            resp_head = await _read_head(up_reader)
            if resp_head is None:
                rec["error"] = "upstream closed before response head"
                return
            resp_head_only, _, first = resp_head.partition(b"\r\n\r\n")
            rec["resp_status_line"] = resp_head_only.split(b"\r\n")[0].decode("latin-1")
            rec["resp_headers"] = _header_map(resp_head_only.split(b"\r\n")[1:])

            # Head first, then body bytes — writing the body alone makes the
            # client read a chunk header as a status line.
            raw = bytearray(first)
            writer.write(resp_head_only + b"\r\n\r\n")
            if first:
                writer.write(first)
            await writer.drain()

            # Per-read timing: an SSE stream that goes quiet mid-generation
            # shows up here as a large gap, which is what a stall looks like.
            gaps: list[float] = []
            last = time.monotonic()
            while True:
                piece = await up_reader.read(65536)
                if not piece:
                    break
                now = time.monotonic()
                gaps.append(round((now - last) * 1000, 1))
                last = now
                raw.extend(piece)
                writer.write(piece)
                await writer.drain()

            rec["ttfb_ms"] = round((time.monotonic() - t_sent) * 1000, 1)
            rec["gap_max_ms"] = max(gaps) if gaps else 0
            rec["gap_mean_ms"] = round(sum(gaps) / len(gaps), 1) if gaps else 0

            body_bytes = bytes(raw)
            rec["response"] = self._store_body("response", body_bytes)
            rec["frames"], rec["comment_frames"] = _count_frames(body_bytes)
        except (ConnectionResetError, BrokenPipeError, OSError) as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # keep tapping even when a request is weird
            rec["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            rec["duration_ms"] = round((time.monotonic() - rec.get("_t0", time.monotonic())) * 1000, 1)
            self._record(rec)
            for w in (writer, up_writer):
                if w is None:
                    continue
                try:
                    w.close()
                except Exception:
                    pass


async def _read_head(reader: asyncio.StreamReader) -> bytes | None:
    buf = b""
    while b"\r\n\r\n" not in buf:
        piece = await reader.read(65536)
        if not piece:
            return None
        buf += piece
        if len(buf) > 1 << 20:
            return None
    return buf


async def _read_body(reader: asyncio.StreamReader, head: bytes, have: bytes) -> bytes:
    """Read a request body: Content-Length or chunked, raw."""
    headers = _header_map(head.split(b"\r\n")[1:])
    if headers.get("transfer-encoding", "").lower() == "chunked":
        buf = bytearray(have)
        while b"\r\n0\r\n\r\n" not in bytes(buf) and not bytes(buf).endswith(b"0\r\n\r\n"):
            piece = await reader.read(65536)
            if not piece:
                break
            buf.extend(piece)
        return bytes(buf)
    length = int(headers.get("content-length", "0") or 0)
    body = bytearray(have)
    while len(body) < length:
        piece = await reader.read(length - len(body))
        if not piece:
            break
        body.extend(piece)
    return bytes(body)


def _header_map(lines: list[bytes]) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in lines:
        name, sep, value = line.partition(b":")
        if sep:
            out[name.decode("latin-1").strip().lower()] = value.decode("latin-1").strip()
    return out


def _count_frames(body: bytes) -> tuple[int, int]:
    """(data-event count, comment-line count) — comments are SSE keep-alives."""
    events = comments = 0
    for raw in body.split(b"\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith(b"data:"):
            events += 1
        elif line.startswith(b":"):
            comments += 1
    return events, comments


def _looks_like_chunk_size(line: bytes) -> bool:
    try:
        int(line.split(b";")[0], 16)
        return True
    except ValueError:
        return False


def _split_addr(value: str, flag: str) -> tuple[str, int]:
    """`host:port` -> (host, port), with a readable error instead of a traceback."""
    host, sep, port = value.rpartition(":")
    if not sep or not port.isdigit():
        raise ValueError(f"{flag} must be host:port, got {value!r}")
    return (host or "127.0.0.1"), int(port)


async def _serve(listen: str, upstream: str, log_path: Path, label: str,
                 capture_bodies: bool, tap_request_only: bool) -> None:
    try:
        host, port = _split_addr(listen, "--listen")
        up_host, up_port = _split_addr(upstream, "--upstream")
    except ValueError as exc:
        print(f"difftap: {exc}", file=sys.stderr)
        return 2
    tapper = Tapper(up_host, up_port, log_path, label,
                    capture_bodies, tap_request_only)
    server = await asyncio.start_server(tapper.handle, host or "127.0.0.1", int(port))
    print(f"difftap: {listen} → {upstream}   log={log_path}  label={label}", flush=True)
    print("difftap: one request per connection; body hashing always on, "
          f"capture_bodies={capture_bodies}", flush=True)
    async with server:
        await server.serve_forever()


# ────────────────────────────────────────────────────────────────────────
# compare
# ────────────────────────────────────────────────────────────────────────

def _load(path: str) -> list[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _first_diff(a: Path, b: Path) -> int | None:
    da, db = a.read_bytes(), b.read_bytes()
    if da == db:
        return None
    lo, hi = 0, min(len(da), len(db))
    while lo < hi and da[lo] == db[lo]:
        lo += 1
    return lo


def compare(left: str, right: str, show: int, path_filter: str | None = None) -> int:
    """Diff two tap logs, pairing requests by route (method+path) in order.

    Pairing by route matters: a proxy arm emits extra requests a client arm
    never makes (health probes, retries), and comparing by raw index matches a
    chat completion against a ``GET /v1/models`` and reports nonsense.
    """
    a, b = _load(left), _load(right)

    def by_route(records):
        out: dict[tuple[str, str], list[dict]] = {}
        for rec in records:
            if rec.get("error") and not rec.get("response"):
                continue  # a failed probe is not a comparable exchange
            key = (rec.get("method", ""), rec.get("path", ""))
            if path_filter and path_filter not in key[1]:
                continue
            out.setdefault(key, []).append(rec)
        return out

    ga, gb = by_route(a), by_route(b)
    problems = compared = 0
    for route in sorted(set(ga) | set(gb)):
        ra_list, rb_list = ga.get(route, []), gb.get(route, [])
        if len(ra_list) != len(rb_list):
            print(f"[{route[0]} {route[1]}] request count differs: "
                  f"{len(ra_list)} vs {len(rb_list)} (comparing {min(len(ra_list), len(rb_list))})")
        for i in range(min(len(ra_list), len(rb_list))):
            compared += 1
            notes = _diff_pair(ra_list[i], rb_list[i])
            if notes:
                problems += 1
                print(f"[{route[0]} {route[1]} #{i}]")
                for note in notes:
                    print(f"     - {note}")
            elif show > 0:
                sa = ra_list[i].get("response", {})
                print(f"[{route[0]} {route[1]} #{i}] identical: "
                      f"{sa.get('len')}B, {ra_list[i].get('frames')} frames, "
                      f"{ra_list[i].get('duration_ms')}ms vs {rb_list[i].get('duration_ms')}ms")
                show -= 1

    if compared == 0:
        print("no comparable exchanges — check the logs and --path")
        return 1
    verdict = ("IDENTICAL — any hinkiness is timing/lifecycle, not content"
               if problems == 0 else f"{problems} of {compared} differ — see above")
    print(f"\n{left}  vs  {right}: {verdict}")
    return 0 if problems == 0 else 1


def _diff_pair(ra: dict, rb: dict) -> list[str]:
    notes = []
    qa, qb = ra.get("request", {}), rb.get("request", {})
    if qa.get("sha256") != qb.get("sha256"):
        at = ""
        if qa.get("file") and qb.get("file") and Path(qa["file"]).exists() and Path(qb["file"]).exists():
            off = _first_diff(Path(qa["file"]), Path(qb["file"]))
            at = "" if off is None else f" (first difference at byte {off})"
        notes.append(f"REQUEST BYTES DIFFER{at}: "
                     f"{qa.get('len')}B {qa.get('sha256','')[:12]} vs "
                     f"{qb.get('len')}B {qb.get('sha256','')[:12]}")
    sa, sb = ra.get("response", {}), rb.get("response", {})
    if sa.get("sha256") != sb.get("sha256"):
        at = ""
        if sa.get("file") and sb.get("file") and Path(sa["file"]).exists() and Path(sb["file"]).exists():
            off = _first_diff(Path(sa["file"]), Path(sb["file"]))
            at = "" if off is None else f" (first difference at byte {off})"
        notes.append(f"RESPONSE BYTES DIFFER{at}: "
                     f"{sa.get('len')}B/{ra.get('frames')} frames vs "
                     f"{sb.get('len')}B/{rb.get('frames')} frames")
    if ra.get("frames") != rb.get("frames") and sa.get("sha256") == sb.get("sha256"):
        notes.append(f"frame counting differs: {ra.get('frames')} vs {rb.get('frames')}")
    if ra.get("resp_status_line") != rb.get("resp_status_line"):
        notes.append(f"status {ra.get('resp_status_line')} vs {rb.get('resp_status_line')}")
    if ra.get("error") or rb.get("error"):
        notes.append(f"error on one side: {ra.get('error')!r} / {rb.get('error')!r}")
    return notes


# ────────────────────────────────────────────────────────────────────────
# cli
# ────────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="tap traffic and write a JSONL log")
    serve.add_argument("--listen", default="127.0.0.1:8099")
    serve.add_argument("--upstream", default="127.0.0.1:8082")
    serve.add_argument("--log", default="/tmp/difftap.jsonl")
    serve.add_argument("--label", default="tap")
    serve.add_argument("--capture-bodies", action="store_true",
                       help="keep exact bytes on disk so compare can report byte offsets")
    serve.add_argument("--tap-request-only", action="store_true",
                       help="hash but do not analyse response framing")

    cmp_ = sub.add_parser("compare", help="diff two tap logs")
    cmp_.add_argument("left")
    cmp_.add_argument("right")
    cmp_.add_argument("--show-identical", type=int, default=0)
    cmp_.add_argument("--path", default=None,
                      help="only compare requests whose path contains this (e.g. chat/completions)")

    args = parser.parse_args(argv)

    if args.cmd == "serve":
        try:
            # propagate _serve's status (2 = bad address) instead of swallowing it
            return asyncio.run(_serve(args.listen, args.upstream, Path(args.log), args.label,
                                      args.capture_bodies, args.tap_request_only)) or 0
        except KeyboardInterrupt:
            return 0
    return compare(args.left, args.right, args.show_identical, args.path)


if __name__ == "__main__":
    sys.exit(main())
