"""FastAPI tabanlı web arayüz + websocket canlı sinyal yayın.

Çalıştırma:
  uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import asyncio
import json
import time
from datetime import datetime
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi import Request

from engine import get_settings, SignalEngine
from engine import indicators
from engine.performance import PerformanceTracker, set_global_tracker, _GLOBAL_TRACKER
from engine.learner import OnlineLearner, set_global_learner, get_global_learner, get_global_learner as _get_l
from engine.storage import save_state, load_state

settings = get_settings()
ASSET_VERSION = "v6"  # tasarım cache bust
app = FastAPI(title="CryptoSignal Pro", version="1.0.0", description="AI Destekli Kripto Sinyal Platformu")

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
    engine.start_time = time.time()  # Track uptime
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
    
    print(f"🚀 CryptoSignal Pro başlatıldı - {settings.MODE} modu")


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
    """Get current live signals with enhanced formatting"""
    signals = engine.get_live()
    formatted_signals = []
    
    for signal in signals[-100:]:  # Last 100 signals
        formatted_signal = {
            "symbol": signal.get("symbol"),
            "side": signal.get("side"),
            "entry": signal.get("entry"),
            "sl": signal.get("sl"),
            "tp1": signal.get("tp1"),
            "tp2": signal.get("tp2"),
            "tp3": signal.get("tp3"),
            "score": signal.get("score"),
            "regime": signal.get("regime"),
            "reason": signal.get("reason"),
            "prob": signal.get("prob"),
            "created_ts": signal.get("created_ts"),
        }
        formatted_signals.append(formatted_signal)
    
    return {
        "count": len(formatted_signals),
        "signals": formatted_signals,
        "last_update": time.time()
    }


@app.get("/api/stats")
async def api_stats():
    """Get comprehensive stats including performance and AI learning data"""
    from engine.performance import _GLOBAL_TRACKER
    perf = _GLOBAL_TRACKER.get_stats() if _GLOBAL_TRACKER else {}
    
    base = {
        "symbols_scanned": len(engine.current_symbols),
        "cooldown_sec": engine.cooldown_sec,
        "mode": settings.MODE,
        "sleep": settings.SLEEP_SECONDS,
        "min_score": engine.min_score,
        "current_time": time.time(),
        "engine_uptime": time.time() - engine.start_time if hasattr(engine, 'start_time') else 0,
    }
    
    # Add performance data with perf_ prefix
    base.update({f"perf_{k}": v for k, v in perf.items()})
    
    # Add AI learner data
    learner = get_global_learner()
    if learner:
        base["learner"] = {
            "seen": learner.seen,
            "bias": round(learner.bias, 6),
            "weights_count": len(learner.weights),
            "learning_rate": learner.lr,
        }
    
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


@app.get("/api/ta/{symbol}")
async def api_ta(symbol: str):
    """Belirli sembol için çoklu timeframe teknik analiz verisi.

    Döner:
      {
        "symbol": "BTC-USDT",
        "timeframes": {
            "1m": { lastClose, rsi, atr, adx, ema20, ema50, bbPos, vol, changePct },
            ...
        }
      }
    """
    # KuCoin klines interval eşleştirme
    tf_map = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1h": "1hour",
        "4h": "4hour",
    }
    out = {"symbol": symbol.upper(), "timeframes": {}}
    for tf, kc_tf in tf_map.items():
        try:
            raw = engine.client.get_kline(symbol.upper(), kc_tf, limit=150)
            df = indicators.to_df_klines(raw)
            if df is None or df.empty:
                continue
            # Hesaplamalar
            close = df["c"]; high = df["h"]; low = df["l"]; vol = df["v"]
            rsi14 = float(indicators.rsi(close, 14).iloc[-1]) if len(close) >= 15 else None
            atr14 = float(indicators.atr_wilder(high, low, close, 14).iloc[-1]) if len(close) >= 15 else None
            adx14 = float(indicators.adx(high, low, close, 14).iloc[-1]) if len(close) >= 20 else None
            ema20 = float(indicators.ema(close, 20).iloc[-1]) if len(close) >= 20 else None
            ema50 = float(indicators.ema(close, 50).iloc[-1]) if len(close) >= 50 else None
            ma, upper, lower, bwidth, std = indicators.bollinger(close, 20)
            bb_pos = None
            if not upper.isna().iloc[-1] and not lower.isna().iloc[-1]:
                rng = (upper.iloc[-1] - lower.iloc[-1]) or 1e-12
                bb_pos = float((close.iloc[-1] - lower.iloc[-1]) / rng)
            change_pct = None
            if len(close) > 1:
                prev = close.iloc[-2]
                if prev:
                    change_pct = float((close.iloc[-1] - prev) / prev * 100)
            out["timeframes"][tf] = {
                "close": float(close.iloc[-1]),
                "rsi14": rsi14,
                "atr14": atr14,
                "adx14": adx14,
                "ema20": ema20,
                "ema50": ema50,
                "bb_pos": bb_pos,
                "bb_width": float(bwidth.iloc[-1]) if not bwidth.isna().iloc[-1] else None,
                "vol": float(vol.iloc[-1]),
                "change_pct": change_pct,
                "ts": int(df["time"].iloc[-1].timestamp()),
            }
        except Exception as e:
            out["timeframes"][tf] = {"error": str(e)}
    return out


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
