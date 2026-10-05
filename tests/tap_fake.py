"""Tiny in-process TAP (UWS async) server for mirror tests, all on 127.0.0.1.

Implements exactly the UWS wire contract pyvo 1.8 uses for
``submit_job -> run -> fetch_result`` and the query shapes the xmatch
mirror emits (:mod:`xmatch.mirror._mirror_tap`):

* page fetch:  ``SELECT ... FROM t AS <alias> ORDER BY t."KEY" LIMIT n OFFSET m``
* count probe: ``SELECT COUNT(*) AS n FROM t AS <alias> WHERE t."KEY" >= lo AND t."KEY" <= hi``
* key probe:   ``SELECT t."KEY" FROM t AS <alias> ORDER BY t."KEY" DESC LIMIT 1``
* continuation: ``SELECT ... FROM t AS <alias> WHERE t."KEY" > v ORDER BY t."KEY" LIMIT n``

The table is in-memory and mutable (:meth:`FakeTAPServer.update`), so
tests can simulate remote mutation (append / window shrink) and assert
the incremental re-sync behaviour.
"""

from __future__ import annotations

import re
import threading
import urllib.parse
import uuid
from collections.abc import Sequence
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_VOT = """<?xml version="1.0"?>
<VOTABLE version="1.4">
<RESOURCE type="results">
<TABLE>
%s
<DATA><TABLEDATA>
%s
</TABLEDATA></DATA>
</TABLE>
</RESOURCE>
</VOTABLE>
"""

_JOB_XML = (
    "<uws:job><uws:jobId>{jid}</uws:jobId><uws:phase>COMPLETED</uws:phase>"
    '<uws:results><uws:result id="result" xlink:href="{base}/tap/async/{jid}/results"/>'
    "</uws:results></uws:job>"
)


def _field_xml(col: str, value: Any) -> str:
    if isinstance(value, bool):
        dtype = "boolean"
    elif isinstance(value, int):
        dtype = "int"
    elif isinstance(value, float):
        dtype = "double"
    else:
        dtype = "char"
    return f'<FIELD name="{col}" datatype="{dtype}"/>'


def _votable(cols: Sequence[str], rows: Sequence[dict[str, Any]]) -> str:
    first = rows[0] if rows else None
    fields = "".join(_field_xml(c, first.get(c) if first else 0) for c in cols)
    lines = [
        "<TR>"
        + "".join(f"<TD>{r.get(c) if r.get(c) is not None else ''}</TD>" for c in cols)
        + "</TR>"
        for r in rows
    ]
    return _VOT % (fields, "\n".join(lines))


def _parse_literal(text: str) -> Any:
    """Unquote an SQL literal (numeric or single-quoted string)."""
    text = text.strip()
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1].replace("''", "'")
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _coerce(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _compare(value: Any, op: str, target: Any) -> bool:
    left, right = _coerce(value), _coerce(target)
    if op == ">=":
        return left >= right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    return left < right


def make_rows(n: int, *, ra0: float = 10.0, dec0: float = -5.0) -> list[dict[str, Any]]:
    """Deterministic ``(id, ra, dec)`` rows ~0.001 deg apart (id-sorted)."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": i,
                "ra": round(ra0 + ((i * 7) % 997) * 0.001, 6),
                "dec": round(dec0 + ((i * 11) % 991) * 0.001, 6),
            }
        )
    return rows


class FakeTAPServer:
    """One mutable table served over the UWS async protocol on 127.0.0.1."""

    def __init__(
        self, rows: Sequence[dict[str, Any]], cols: Sequence[str] = ("id", "ra", "dec")
    ) -> None:
        self.cols = list(cols)
        self.key = "id"
        self.rows: list[dict[str, Any]] = [dict(r) for r in rows]
        self.queries: list[str] = []
        self.jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler_cls())
        self.port = self.httpd.server_address[1]
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/tap"

    def update(self, rows: Sequence[dict[str, Any]]) -> None:
        """Replace the whole table (simulates a remote mutation)."""
        with self._lock:
            self.rows = [dict(r) for r in rows]

    def count_queries(self, needle: str) -> int:
        return sum(1 for q in self.queries if needle in q)

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self._thread.join(timeout=5)

    # ------------------------------------------------------------------ SQL
    def _run(self, query: str) -> list[dict[str, Any]]:
        count_m = re.match(r"\s*SELECT\s+COUNT\(\*\)(?:\s+AS\s+\w+)?", query, re.IGNORECASE)
        if count_m:
            matched = self._collect_rows(query)
            return [{"n": len(matched)}]
        return self._collect_rows(query)

    def _collect_rows(self, query: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self.rows)
        m = re.search(r"ORDER BY\s+\S*\.?\"?(\w+)\"?\s*(ASC|DESC)?", query, re.IGNORECASE)
        order_col = m.group(1) if m else self.key
        order_dir = (m.group(2) or "ASC").upper() if m else "ASC"
        where_m = re.search(
            r"WHERE\s+(.*?)(?=\s+ORDER BY|\s+LIMIT|\s*$)", query, re.IGNORECASE | re.DOTALL
        )
        if where_m:
            for clause in where_m.group(1).split(" AND "):
                cond = re.match(r"\w+\.?\"?(\w+)\"?\s*(>=|<=|>|<)\s*(.+)", clause)
                if not cond:
                    continue
                col, op, raw = cond.group(1), cond.group(2), cond.group(3).strip()
                target = _parse_literal(raw)
                rows = [r for r in rows if r.get(col) is not None and _compare(r[col], op, target)]
        rows.sort(key=lambda r: _coerce(r.get(order_col)), reverse=(order_dir == "DESC"))
        limit_m = re.search(r"LIMIT\s+(\d+)", query, re.IGNORECASE)
        offset_m = re.search(r"OFFSET\s+(\d+)", query, re.IGNORECASE)
        start = int(offset_m.group(1)) if offset_m else 0
        if limit_m:
            return rows[start : start + int(limit_m.group(1))]
        return rows[start:]

    # ------------------------------------------------------------- handler
    def _handler_cls(self) -> type:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # silence
                pass

            def _send(
                self, code: int, body: str, ctype: str = "text/xml", headers: Sequence[tuple] = ()
            ) -> None:
                data = body.encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                for k, v in headers:
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:  # noqa: N802 - http.server API
                path = urllib.parse.urlparse(self.path).path
                m = re.fullmatch(r"/tap/async/([0-9a-f]+)(/phase|/results)?", path)
                if not m or m.group(1) not in server.jobs:
                    self._send(404, "<oops/>")
                    return
                jid, suffix = m.group(1), m.group(2)
                if suffix == "/phase":
                    self._send(200, "<uws:phase>COMPLETED</uws:phase>")
                elif suffix == "/results":
                    job_rows = server.jobs[jid]["rows"]
                    cols = list(job_rows[0].keys()) if job_rows else server.cols
                    self._send(200, _votable(cols, job_rows))
                else:
                    base = f"http://127.0.0.1:{server.port}"
                    self._send(200, _JOB_XML.format(jid=jid, base=base))

            def do_POST(self) -> None:  # noqa: N802 - http.server API
                path = urllib.parse.urlparse(self.path).path
                length = int(self.headers.get("Content-Length", 0))
                form = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8", "replace"))
                if path == "/tap/async":
                    query = form.get("QUERY", [""])[0]
                    server.queries.append(query)
                    jid = uuid.uuid4().hex
                    server.jobs[jid] = {"query": query, "rows": server._run(query)}
                    base = f"http://127.0.0.1:{server.port}"
                    self._send(303, "", headers=[("Location", f"{base}/tap/async/{jid}")])
                elif re.fullmatch(r"/tap/async/[0-9a-f]+/phase", path):
                    self._send(200, "")
                else:
                    self._send(404, "<oops/>")

        return Handler
