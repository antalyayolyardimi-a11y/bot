"""FastAPI tabanlı web arayüz + websocket canlı sinyal yayın.

Çalıştırma:
  uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import asyncio
import json
from datetime import datetime
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi import Request

from engine import get_settings, SignalEngine
from engine.performance import PerformanceTracker, set_global_tracker, _GLOBAL_TRACKER
from engine.learner import OnlineLearner, set_global_learner, get_global_learner, get_global_learner as _get_l
from engine.storage import save_state, load_state

settings = get_settings()
ASSET_VERSION = "v5"  # tasarım cache bust
app = FastAPI(title="Crypto Signal Engine", version="0.1.0")

# CORS (frontend ayrı origin'de koşarsa)
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

engine = SignalEngine()

# Öncelikli (pinned) semboller – dashboard'ta sabit kartlar
PINNED_SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT"]

@app.on_event("startup")
async def _startup():
    await engine.start()
    # Performans tracker
    tracker = PerformanceTracker(engine)
    set_global_tracker(tracker)
    await tracker.start()
    learner = OnlineLearner()
    set_global_learner(learner)
    # Persistten yükle
    st = load_state()
    if st.get("learner"):
        ld = st["learner"]
        learner.bias = ld.get("bias", 0.0)
        learner.seen = ld.get("seen", 0)
        for k,v in ld.items():
            if k.startswith("w_"):
                learner.weights[k[2:]] = v
    if st.get("performance"):
        # eski kapanmış kayıtları sadece görüntüleme amaçlı ekle
        for rec in st["performance"].get("records", []):
            try:
                from engine.performance import SignalRecord
                sr = SignalRecord(**rec)
                tracker.records.append(sr)
            except Exception:
                pass
    # Arka plan periyodik persist
    async def _persist_loop():
        while True:
            await asyncio.sleep(60)
            save_state(learner=learner, performance=tracker)
    asyncio.create_task(_persist_loop())
    # Engine listener -> tracker'a sinyal akışı
    def _perf_listener(sig):
        tracker.add_signal(sig)
    engine.add_listener(_perf_listener)


@app.on_event("shutdown")
async def _shutdown():
    await engine.stop()
    save_state(learner=get_global_learner(), performance=_GLOBAL_TRACKER)


@app.get("/")
async def index():
    # Kullanıcı direkt ana sayfaya gelirse dashboard'a yönlendir.
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Kart bazlı dashboard (websocket + fallback polling)."""
    return templates.TemplateResponse("dashboard.html", {"request": request, "asset_version": ASSET_VERSION})


@app.get("/api/signals")
async def api_signals():
    return {"count": len(engine.signals_live), "signals": engine.get_live()[-100:]}


@app.get("/api/stats")
async def api_stats():
    from engine.performance import _GLOBAL_TRACKER
    perf = _GLOBAL_TRACKER.get_stats() if _GLOBAL_TRACKER else {}
    base = {
        "symbols_scanned": len(engine.current_symbols),
        "cooldown_sec": engine.cooldown_sec,
        "mode": settings.MODE,
        "sleep": settings.SLEEP_SECONDS,
        "min_score": engine.min_score,
    }
    base.update({f"perf_{k}": v for k,v in perf.items()})
    learner = get_global_learner()
    if learner:
        base["learner"] = learner.serialize()
    return base


@app.get("/api/pinned")
async def api_pinned():
    data = []
    for sym in PINNED_SYMBOLS:
        try:
            # KuCoin ticker verisi
            t = engine.client.get_ticker(sym)
            data.append({
                "symbol": sym,
                "price": float(t.get("price") or 0.0),
                "changeRate": float(t.get("changeRate") or 0.0),
                "volValue": float(t.get("volValue") or 0.0),
            })
        except Exception as e:
            data.append({"symbol": sym, "error": str(e)})
    return {"data": data}


@app.get("/healthz")
async def healthz():
	return {"ok": True, "time": datetime.utcnow().isoformat()}


class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for d in dead:
            self.disconnect(d)


manager = ConnectionManager()


def _on_signal(sig):
    # Websocket broadcast
    payload = json.dumps({"type": "signal", "data": sig})
    asyncio.create_task(manager.broadcast(payload))


engine.add_listener(_on_signal)


@app.websocket("/ws/signals")
async def ws_signals(ws: WebSocket):
    await manager.connect(ws)
    # İlk açılışta son sinyalleri gönder
    await ws.send_text(json.dumps({"type": "bootstrap", "data": engine.get_live()[-50:]}))
    try:
        while True:
            # ping/pong veya future client mesajları için bekle
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)

# Telegram entegrasyonu kaldırıldı.
