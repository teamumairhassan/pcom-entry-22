"""Imitates the competition server: serves tests/round1_input.csv at a download URL, receives the
uploaded prediction at an upload URL, and checks it.  Run:  python test_local.py
(This tests the same function the /predict endpoint runs in the background.)"""
import csv
import email
import io
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from features import TARGETS
from predictor import get_bundle, start_prediction

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = open(os.path.join(HERE, "tests", "round1_input.csv"), "rb").read()
received = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type", "text/csv"); self.end_headers()
        self.wfile.write(INPUT)

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        msg = email.message_from_bytes(b"Content-Type: " + self.headers["Content-Type"].encode() + b"\r\n\r\n" + body)
        for part in msg.walk():
            if part.get_param("name", header="content-disposition") == "predictions":
                received["csv"] = part.get_payload(decode=True).decode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(b'{"ok": true}')


server = HTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
base = f"http://127.0.0.1:{server.server_port}"

get_bundle()                                   # same as the server does at startup
t0 = time.time()
start_prediction(f"{base}/input.csv", f"{base}/upload")
elapsed = time.time() - t0

rows = list(csv.reader(io.StringIO(received["csv"])))
print("\nuploaded CSV:\n" + received["csv"])
assert rows[0] == TARGETS, f"wrong header: {rows[0]}"
assert len(rows) == 2, "must contain exactly one record"
vals = [float(x) for x in rows[1]]             # must parse as floats (no NaN / empty)
assert all(v == v and v >= 0 for v in vals)
assert elapsed < 10, f"too slow: {elapsed:.1f}s"
print(f"OK: header correct, 1 record, floats valid, {elapsed:.2f}s (limit 10s)")
