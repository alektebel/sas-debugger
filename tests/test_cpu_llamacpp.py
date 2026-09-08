"""CPU / llama.cpp path tests — no torch, no GPU, no model download.

A stub HTTP server speaks the subset of the ``llama-server`` API that
``slm.backends.LlamaCppBackend`` uses (``/health`` and
``/v1/chat/completions``), so the whole chain

    episode -> backend -> generated JSON -> oracle grading -> metrics

is exercised end to end with nothing but the stdlib.  Run:

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slm import evaluate
from slm.backends import LlamaCppBackend, LlamaServerError, make_backend
from slm.episodes import generate_episodes
from vendor.defect_catalog import DEFECTS


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class StubLlamaServer:
    """Minimal llama-server stand-in.

    ``replies`` maps a substring of the user message to the assistant content
    to return; ``sequence`` (when given) is consumed one entry per request,
    which is how the deterministic eval order is driven; ``default`` is used
    when neither matches.  Every request is recorded so tests can assert on the
    payload the backend actually sends.
    """

    def __init__(self, replies: dict[str, str] | None = None,
                 default: str = "", healthy: bool = True,
                 sequence: list[str] | None = None):
        self.replies = replies or {}
        self.sequence = list(sequence) if sequence else None
        self.default = default
        self.healthy = healthy
        self.requests: list[dict] = []
        self.port = free_port()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence the test output
                pass

            def _json(self, code: int, body: dict) -> None:
                blob = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)

            def do_GET(self):
                if self.path == "/health":
                    if stub.healthy:
                        self._json(200, {"status": "ok"})
                    else:
                        self._json(503, {"error": "loading model"})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(n) or b"{}")
                stub.requests.append(payload)
                user = ""
                for m in payload.get("messages", []):
                    if m.get("role") == "user":
                        user = m.get("content", "")
                content = stub.default
                if stub.sequence:
                    content = stub.sequence.pop(0)
                else:
                    for key, val in stub.replies.items():
                        if key in user:
                            content = val
                            break
                self._json(200, {"choices": [{"message": {"role": "assistant",
                                                          "content": content}}]})

        self.httpd = HTTPServer(("127.0.0.1", self.port), Handler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


class TestLlamaCppBackend(unittest.TestCase):
    def test_health_and_failure_message(self):
        with StubLlamaServer() as stub:
            ok, detail = LlamaCppBackend(stub.url).health()
            self.assertTrue(ok, detail)

        # server gone: health must be False and never raise
        ok, detail = LlamaCppBackend(stub.url).health()
        self.assertFalse(ok)

        # the factory turns that into an actionable error, not a stack trace
        with self.assertRaises(RuntimeError) as cm:
            make_backend("llamacpp", llama_url=stub.url)
        self.assertIn("scripts/llama_server.sh", str(cm.exception))

    def test_unhealthy_server_is_not_used(self):
        with StubLlamaServer(healthy=False) as stub:
            ok, _ = LlamaCppBackend(stub.url).health()
            self.assertFalse(ok)

    def test_greedy_payload_is_deterministic(self):
        with StubLlamaServer(default="hello") as stub:
            be = LlamaCppBackend(stub.url)
            self.assertEqual(be.generate("sys", "usr"), "hello")
            self.assertEqual(be.generate("sys", "usr", do_sample=True,
                                         temperature=0.7), "hello")
        greedy, sampled = stub.requests
        self.assertEqual(greedy["temperature"], 0.0)   # greedy == temperature 0
        self.assertEqual(greedy["seed"], 42)
        self.assertEqual(sampled["temperature"], 0.7)
        self.assertNotIn("seed", sampled)              # sampling must vary
        roles = [m["role"] for m in greedy["messages"]]
        self.assertEqual(roles, ["system", "user"])
        self.assertFalse(greedy["stream"])


class TestServerErrorTolerance(unittest.TestCase):
    """A 500 from llama-server must not abort a long evaluation.

    Reproduced against a real llama-server: its chat-template parser answers
    500 ("does not match the expected peg-native format") on some sampled
    outputs.  One of those in a 268-episode run must score as a failure, not
    kill the run.
    """

    def _failing_server(self):
        stub = StubLlamaServer()
        # patch the handler by serving a 500 for POSTs
        original = stub.httpd.RequestHandlerClass.do_POST

        def do_POST(self):
            self.send_response(500)
            body = b'{"error":{"code":500,"message":"peg-native format"}}'
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        stub.httpd.RequestHandlerClass.do_POST = do_POST
        stub._restore = original
        return stub

    def test_500_is_counted_not_raised(self):
        stub = self._failing_server()
        with stub:
            be = LlamaCppBackend(stub.url)
            self.assertEqual(be.generate("s", "u"), "")
            self.assertEqual(be.generate("s", "u"), "")
            self.assertEqual(be.server_errors, 2)
        stub.httpd.RequestHandlerClass.do_POST = stub._restore

    def test_strict_mode_still_raises(self):
        stub = self._failing_server()
        with stub:
            be = LlamaCppBackend(stub.url, tolerate_server_errors=False)
            with self.assertRaises(LlamaServerError) as cm:
                be.generate("s", "u")
            self.assertEqual(cm.exception.code, 500)
        stub.httpd.RequestHandlerClass.do_POST = stub._restore


class TestEvaluateOnLlamaCpp(unittest.TestCase):
    """The full eval chain on the CPU backend, graded by the real oracle."""

    @classmethod
    def setUpClass(cls):
        cls.eps = generate_episodes(DEFECTS[:3], n_per_defect=2, n_rows=200,
                                    test_frac=0.5)
        assert cls.eps, "episode generation produced nothing"

    def test_gold_traces_score_perfect(self):
        # run_split walks the episodes in order, so the stub replays the gold
        # traces in that same order (prompts are not unique enough to key on).
        golds = [ep["assistant"] for ep in self.eps]
        with StubLlamaServer(sequence=golds, default="{}") as stub:
            be = make_backend("llamacpp", llama_url=stub.url)
            res = evaluate.run_split(self.eps, be)
        summary = evaluate.summarize(res)
        self.assertEqual(summary["overall_success"], 1.0, summary)
        self.assertEqual(summary["false_positive_rate"], 0.0, summary)

    def test_garbage_output_is_scored_as_failure(self):
        with StubLlamaServer(default="I am not JSON.") as stub:
            be = make_backend("llamacpp", llama_url=stub.url)
            res = evaluate.run_split(self.eps, be)
        summary = evaluate.summarize(res)
        self.assertEqual(summary["overall_success"], 0.0, summary)
        self.assertEqual(summary["valid_query_rate"], 0.0, summary)

    def test_best_of_n_picks_the_gold_trace(self):
        """One good sample among junk must be the one selected."""
        ep = self.eps[0]
        gold = ep["assistant"]
        seq = ["not json", "not json", gold, "not json"]
        state = {"i": 0}

        with StubLlamaServer(default="") as stub:
            be = make_backend("llamacpp", llama_url=stub.url)
            original = be.generate

            def rotating(system, user, **kw):
                out = seq[state["i"] % len(seq)]
                state["i"] += 1
                original(system, user, **kw)  # keep the HTTP path exercised
                return out

            be.generate = rotating
            bon = evaluate.run_best_of_n([ep], be, n=len(seq), temperature=0.7)
        self.assertTrue(bon["results"][0]["success"], bon["results"][0])


class TestNoTorchOnTheCpuPath(unittest.TestCase):
    def test_cpu_modules_do_not_import_torch(self):
        """Importing the CPU path must not pull in torch."""
        import subprocess
        code = ("import sys; import slm.backends, slm.evaluate, slm.oracle, "
                "slm.episodes, slm.reward; "
                "print('torch' in sys.modules)")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run([sys.executable, "-c", code], cwd=root,
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.strip(), "False", out.stdout)


class TestDemoAgentDispatch(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "examples", "app"))
        import agent
        self.agent = agent
        self._sleep = agent._sleep
        agent._sleep = lambda *a, **k: None  # the demo paces itself for humans

    def tearDown(self):
        self.agent._sleep = self._sleep

    def _assets(self) -> tuple[str, str, str]:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        a = os.path.join(root, "examples", "app_assets")
        return (os.path.join(a, "src_basilea.egp"),
                os.path.join(a, "rep_lgd.egp"),
                os.path.join(a, "ciclos_sospechosos.xlsx"))

    def test_auto_falls_back_to_scripted_without_a_server(self):
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            self.skipTest("openpyxl not installed (requirements-cpu.txt)")
        src, rep, xlsx = self._assets()
        dead = f"http://127.0.0.1:{free_port()}"
        events = list(self.agent.run_agent(src, rep, xlsx, mode="auto",
                                           llama_url=dead))
        kinds = [e["kind"] for e in events]
        self.assertEqual(kinds[0], "backend")
        self.assertEqual(events[0]["backend"], "scripted")
        self.assertIn("done", kinds)
        localize = next(e for e in events if e["kind"] == "localize")
        self.assertEqual(localize["root_cause"]["table"], "t7_ead_cal")

    def test_model_mode_reports_a_dead_server_instead_of_crashing(self):
        src, rep, xlsx = self._assets()
        dead = f"http://127.0.0.1:{free_port()}"
        events = list(self.agent.run_agent(src, rep, xlsx, mode="model",
                                           llama_url=dead))
        self.assertEqual(events[0]["kind"], "backend")
        self.assertFalse(events[0]["healthy"])
        self.assertEqual(events[1]["kind"], "error")
        self.assertIn("llama_server.sh", events[1]["text"])


if __name__ == "__main__":
    unittest.main()
