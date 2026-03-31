"""
FastAPI application — REST endpoints + WebSocket for live crawl streaming.

Endpoints:
  POST   /api/projects          → Create a new project
  GET    /api/projects          → List all projects
  GET    /api/projects/{id}     → Get project details + results
  POST   /api/projects/{id}/start → Start crawling
  POST   /api/projects/{id}/stop  → Stop crawling
  GET    /api/projects/{id}/results → Get extracted data
  GET    /api/projects/{id}/export/csv → Download results as CSV
  WS     /ws/{project_id}       → Live event stream
"""
import asyncio
import json
import csv
import io
import time
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pathlib

from .core.database import init_db, get_session_factory, Project, Result, EventLog, create_db_engine
from .core.orchestrator import Orchestrator
from .utils.event_bus import EventBus, Event, EventType

# ─── App setup ────────────────────────────────────────────────────
app = FastAPI(title="AI Scraper", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database
engine = create_db_engine()
init_db(engine)
SessionFactory = get_session_factory(engine)

# Active orchestrators (project_id → Orchestrator)
_active: dict[int, Orchestrator] = {}

# ─── Frontend serving (embedded — no external file needed) ───
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    # Try external file first, fall back to embedded
    for candidate in [
        pathlib.Path(__file__).parent.parent / "frontend" / "index.html",
        pathlib.Path.cwd() / "frontend" / "index.html",
        pathlib.Path.cwd() / "ai_scraper" / "frontend" / "index.html",
    ]:
        if candidate.exists():
            return HTMLResponse(content=candidate.read_text(encoding="utf-8"))
    # Fallback: serve embedded frontend
    return HTMLResponse(content=EMBEDDED_FRONTEND)


# ─── Request/Response models ─────────────────────────────────────
class ProjectCreate(BaseModel):
    name: str
    query: str
    mode: str = "TEXT_ONLY"
    schema_def: dict  # {"field_name": "type", ...}
    max_depth: int = 3


class ProjectResponse(BaseModel):
    id: int
    name: str
    query: str
    mode: str
    status: str
    total_items: int
    created_at: float


# ─── REST Endpoints ──────────────────────────────────────────────

@app.post("/api/projects", response_model=ProjectResponse)
async def create_project(req: ProjectCreate):
    session = SessionFactory()
    try:
        project = Project(
            name=req.name,
            query=req.query,
            mode=req.mode.upper(),
            schema_json=json.dumps(req.schema_def),
            max_depth=req.max_depth,
        )
        session.add(project)
        session.commit()
        session.refresh(project)
        return ProjectResponse(
            id=project.id, name=project.name, query=project.query,
            mode=project.mode, status=project.status,
            total_items=0, created_at=project.created_at,
        )
    finally:
        session.close()


@app.get("/api/projects")
async def list_projects():
    session = SessionFactory()
    try:
        projects = session.query(Project).order_by(Project.created_at.desc()).all()
        return [
            {
                "id": p.id, "name": p.name, "query": p.query,
                "mode": p.mode, "status": p.status,
                "total_items": p.total_items, "created_at": p.created_at,
            }
            for p in projects
        ]
    finally:
        session.close()


@app.get("/api/projects/{project_id}")
async def get_project(project_id: int):
    session = SessionFactory()
    try:
        project = session.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(404, "Project not found")
        results = session.query(Result).filter(Result.project_id == project_id).all()
        return {
            "id": project.id, "name": project.name, "query": project.query,
            "mode": project.mode, "status": project.status,
            "total_items": project.total_items,
            "schema": project.schema_dict,
            "results": [{"id": r.id, "data": r.data, "url": r.source_url} for r in results],
        }
    finally:
        session.close()


@app.post("/api/projects/{project_id}/start")
async def start_project(project_id: int):
    session = SessionFactory()
    try:
        project = session.query(Project).filter(Project.id == project_id).first()
        if not project:
            raise HTTPException(404, "Project not found")

        if project_id in _active and _active[project_id].is_running:
            raise HTTPException(400, "Project is already running")

        # Create orchestrator
        orch = Orchestrator(
            project_id=project.id,
            mode=project.mode,
            query=project.query,
            json_schema_dict=project.schema_dict,
            max_depth=project.max_depth,
        )

        # Subscribe to data events → save to DB
        async def on_event(event: Event):
            if event.event_type == EventType.DATA:
                s = SessionFactory()
                try:
                    result = Result(
                        project_id=project_id,
                        data_json=json.dumps(event.payload),
                        source_url=event.url,
                    )
                    s.add(result)
                    proj = s.query(Project).filter(Project.id == project_id).first()
                    if proj:
                        proj.total_items = (proj.total_items or 0) + 1
                    s.commit()
                except Exception as e:
                    s.rollback()
                    print(f"[DB] Save error: {e}")
                finally:
                    s.close()

            elif event.event_type == EventType.STATUS:
                s = SessionFactory()
                try:
                    proj = s.query(Project).filter(Project.id == project_id).first()
                    if proj:
                        proj.status = event.payload
                        s.commit()
                except Exception:
                    s.rollback()
                finally:
                    s.close()

        orch.event_bus.subscribe(on_event)
        _active[project_id] = orch

        # Start in background
        asyncio.create_task(orch.start())

        project.status = "starting"
        session.commit()

        return {"status": "started", "project_id": project_id}
    finally:
        session.close()


@app.post("/api/projects/{project_id}/stop")
async def stop_project(project_id: int):
    if project_id not in _active:
        raise HTTPException(404, "No active crawl for this project")

    await _active[project_id].stop()
    del _active[project_id]
    return {"status": "stopped"}


@app.get("/api/projects/{project_id}/results")
async def get_results(project_id: int):
    session = SessionFactory()
    try:
        results = session.query(Result).filter(
            Result.project_id == project_id
        ).order_by(Result.created_at.asc()).all()
        return [
            {"id": r.id, "data": r.data, "url": r.source_url, "created_at": r.created_at}
            for r in results
        ]
    finally:
        session.close()


@app.get("/api/projects/{project_id}/export/csv")
async def export_csv(project_id: int):
    session = SessionFactory()
    try:
        results = session.query(Result).filter(
            Result.project_id == project_id
        ).order_by(Result.created_at.asc()).all()

        if not results:
            raise HTTPException(404, "No results to export")

        # Build CSV in memory
        output = io.StringIO()
        all_data = [r.data for r in results]

        # Get all unique keys across all results
        all_keys = set()
        for d in all_data:
            all_keys.update(d.keys())
        all_keys = sorted(all_keys)

        writer = csv.DictWriter(output, fieldnames=["source_url"] + list(all_keys))
        writer.writeheader()
        for r, d in zip(results, all_data):
            row = {"source_url": r.source_url}
            row.update(d)
            writer.writerow(row)

        output.seek(0)
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=project_{project_id}_results.csv"},
        )
    finally:
        session.close()


@app.get("/api/projects/{project_id}/stats")
async def get_stats(project_id: int):
    if project_id in _active:
        return _active[project_id].stats
    return {"running": False, "message": "No active crawl"}


# ─── WebSocket — live event stream ───────────────────────────────

@app.websocket("/ws/{project_id}")
async def websocket_endpoint(websocket: WebSocket, project_id: int):
    await websocket.accept()

    # Wait for orchestrator to be available (might be starting up)
    orch = None
    for _ in range(30):  # Wait up to 30 seconds
        orch = _active.get(project_id)
        if orch:
            break
        await asyncio.sleep(1)

    if not orch:
        await websocket.send_json({"type": "error", "payload": "No active crawl found"})
        await websocket.close()
        return

    # Check if client wants to replay missed events
    last_event_id = 0
    try:
        init_msg = await asyncio.wait_for(websocket.receive_json(), timeout=2)
        last_event_id = init_msg.get("last_event_id", 0)
    except Exception:
        pass  # No replay request, start from now

    # Replay missed events
    if last_event_id > 0:
        for event in orch.event_bus.replay_since(last_event_id):
            await websocket.send_json({
                "event_id": event.event_id,
                "type": event.event_type.value,
                "payload": event.payload,
                "url": event.url,
                "timestamp": event.timestamp,
            })

    # Subscribe to live events
    async def forward_event(event: Event):
        try:
            await websocket.send_json({
                "event_id": event.event_id,
                "type": event.event_type.value,
                "payload": event.payload,
                "url": event.url,
                "timestamp": event.timestamp,
            })
        except Exception:
            pass

    orch.event_bus.subscribe(forward_event)

    try:
        while True:
            # Keep connection alive, listen for client messages
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                if msg == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat", "timestamp": time.time()})
            except WebSocketDisconnect:
                break
    except Exception:
        pass
    finally:
        orch.event_bus.unsubscribe(forward_event)


# ─── Startup / Shutdown ──────────────────────────────────────────

@app.on_event("startup")
async def startup():
    print("AI Scraper v2.0 — LangGraph Architecture")
    print(f"Frontend: http://localhost:8000")
    print(f"API docs: http://localhost:8000/docs")


@app.on_event("shutdown")
async def shutdown():
    for orch in _active.values():
        await orch.stop()
    _active.clear()
