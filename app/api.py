# app/api.py
from __future__ import annotations
import asyncio, time
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .rag import answer_with_docs_async
from .ingest import run_ingest_async

app = FastAPI(title="Company Knowledge Assistant")

# Static frontend
static_dir = Path(__file__).with_name("static")
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Ingestion status
_ingest_lock = asyncio.Lock()
_ingest_task: asyncio.Task | None = None
_ingest_last = {#has all the status information
    "status": "idle",      # idle | running | succeeded | failed
    "started_at": None,
    "finished_at": None,
    "stats": None,         # {"documents":..., "chunks":..., "collection":...}
    "error": None,
}

class Ask(BaseModel):
    question: str

@app.get("/")
async def root_page():
    return FileResponse(static_dir / "index.html")

async def _ingest_job():
    _ingest_last.update({"status": "running", "started_at": time.time(), "finished_at": None, "stats": None, "error": None})
    try:
        #TODO: RUN INGESTION
        stats = await run_ingest_async() #invokes from the ingest.py

        _ingest_last.update({"status": "succeeded", "finished_at": time.time(), "stats": stats})
    except Exception as e:
        _ingest_last.update({"status": "failed", "finished_at": time.time(), "error": str(e)})

@app.post("/ingest")#this is the api method within which we launch an asynchronous task, this is also called fast API and it handles: listening to the http request, routing the URL /ingest to the right python function, ad=nd autimatically converts the python dictionary return value into a JSON response for the browser
async def kick_off_ingest():
    global _ingest_task
    async with _ingest_lock:
        if _ingest_task and not _ingest_task.done():
            return JSONResponse({"ok": False, "message": "Ingestion already running"}, status_code=409)
        #TODO: Create Ingestion Task
        _ingest_task = asyncio.create_task(_ingest_job())#this launches the task
    return {"ok": True, "message": "Ingestion started"}#returned to the UI

@app.get("/ingest/status")
async def ingest_status():
    return {"ok": True, **_ingest_last}

@app.post("/ask")
async def ask(q: Ask):
    start = time.perf_counter()
    
    #TODO: Call RAG
    category = None
   
    answer, sources , contexts = await answer_with_docs_async(q.question,category)

    elapsed = time.perf_counter() - start
    print(f"⏱️ /ask execution took {elapsed:.2f} seconds")
    

    return {
        "answer": answer,         
        "sources": sources,
        "contexts":contexts
    }