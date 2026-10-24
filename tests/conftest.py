import http.server
import json
import threading

import pytest


class ScoringService:
    """A local HTTP server standing in for the platform and a scoring service: canned answers per
    path (a callable answers from the request), every request recorded."""

    def __init__(self):
        self.routes = {}
        self.seen = []
        service = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                length = int(self.headers.get("Content-Length") or 0)
                request = {"method": self.command, "path": self.path,
                           "body": self.rfile.read(length) if length else b""}
                service.seen.append(request)
                status, payload = service.routes.get(self.path.split("?")[0], (404, {"detail": "no route"}))
                if callable(payload):
                    payload = payload(request)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_POST = do_GET

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.host = f"127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def route(self, path, status, payload):
        self.routes[path] = (status, payload)


@pytest.fixture
def scoring_service():
    service = ScoringService()
    yield service
    service.server.shutdown()
