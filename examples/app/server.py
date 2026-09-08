"""Local demo web app — watch the agent debug a deep discrepancy in real time.

Run:
    .venv/bin/python examples/app/server.py
Then open http://127.0.0.1:8000

The page lets you load the two corresponding `.egp` projects and the `.xlsx` of
suspect cycles (or use the bundled sample assets), then streams the agent's
debug trace — reasoning, key-table extractions, diagnostic SQL, oracle verdicts
and the final localisation — as Server-Sent Events.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import os
import sys

from fastapi import FastAPI, File, UploadFile, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "app_assets"

# sample assets the server auto-loads if the user uploads nothing
SAMPLE_SRC = ASSETS / "src_basilea.egp"
SAMPLE_REP = ASSETS / "rep_lgd.egp"
SAMPLE_XLS = ASSETS / "ciclos_sospechosos.xlsx"

app = FastAPI(title="SAS Reconcile SLM — debug demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

# in-memory session store: session_id -> dict(file paths, summary)
sessions: dict[str, dict] = {}

INDEX_HTML = (HERE / "index.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/api/assets")
def assets() -> dict:
    return {"src": SAMPLE_SRC.name, "rep": SAMPLE_REP.name,
            "xlsx": SAMPLE_XLS.name}


@app.get("/api/backend")
def backend_status() -> dict:
    """Is a llama.cpp server reachable?  Drives the mode toggle in the UI."""
    sys.path.insert(0, str(HERE))
    from agent import llama_server_healthy
    url = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
    ok, detail = llama_server_healthy(url)
    return {"llama_url": url, "healthy": ok, "detail": detail,
            "default_mode": "model" if ok else "scripted"}


@app.post("/api/session")
async def create_session(
    src: UploadFile | None = File(None),
    rep: UploadFile | None = File(None),
    xlsx: UploadFile | None = File(None),
) -> dict:
    sid = uuid.uuid4().hex
    tmp = Path("/tmp/sas-slm-demo") / sid
    tmp.mkdir(parents=True, exist_ok=True)

    def save(upload: UploadFile | None, fallback: Path, default: Path) -> Path:
        if upload is not None:
            p = tmp / (upload.filename or fallback.name)
            p.write_bytes(upload.file.read())
            return p
        return default  # use bundled sample when nothing uploaded

    src_p = save(src, SAMPLE_SRC, SAMPLE_SRC)
    rep_p = save(rep, SAMPLE_REP, SAMPLE_REP)
    xlsx_p = save(xlsx, SAMPLE_XLS, SAMPLE_XLS)
    sessions[sid] = {"src": str(src_p), "rep": str(rep_p), "xlsx": str(xlsx_p)}
    return {"session_id": sid, "src": src_p.name, "rep": rep_p.name,
            "xlsx": xlsx_p.name}


@app.get("/api/debug/{sid}")
def debug_stream(sid: str, mode: str = Query("scripted",
                 pattern="^(scripted|model|auto)$")) -> StreamingResponse:
    sys.path.insert(0, str(HERE))
    from agent import run_agent

    sess = sessions[sid]
    llama_url = os.environ.get("LLAMA_SERVER_URL")

    def gen():
        for ev in run_agent(sess["src"], sess["rep"], sess["xlsx"],
                            mode=mode, llama_url=llama_url):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no",
                                      "Connection": "keep-alive"})


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8010"))
    print(f"Open http://127.0.0.1:{port}  (sample assets: {SAMPLE_SRC.name}, "
          f"{SAMPLE_REP.name}, {SAMPLE_XLS.name})")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
