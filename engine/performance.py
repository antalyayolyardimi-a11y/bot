"""Basit performans izleyici ve adaptif parametre ayarlayıcı.

Amaç: Üretilen sinyallerin TP/SL sonuçlarını izleyip
 - win rate (TP1'e ulaşan) / stop rate'e göre MIN_SCORE eşiğini ayarlamak
 - stop oranı yüksekse ATR çarpanını genişletmek, düşükse daraltmak

MVP: Bellek içi (persist yok). Gelecekte JSON veya DB kalıcılığı eklenebilir.
"""
from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from .learner import get_global_learner


@dataclass
class SignalRecord:
    symbol: str
    side: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    created_ts: float
    score: float
    regime: str
    reason: str
    status: str = "OPEN"  # OPEN / STOP / TP1 / TP2 / TP3 / EXPIRED
    closed_ts: Optional[float] = None
    realized_r: Optional[float] = None


class PerformanceTracker:
    def __init__(self, engine):
        self.engine = engine
        self.records: List[SignalRecord] = []
        self._task: Optional[asyncio.Task] = None
        # Adaptif parametreler
        self.dynamic_min_score: Optional[int] = None
        self._atr_mult: float = 1.2
        self._lock = asyncio.Lock()
        # Değerlendirme aralığı
        self.eval_interval = 60  # saniye
        self.signal_ttl = 60 * 90  # 90 dakika sonra kapat (EXPIRED) (MVP)

    # --------- Public ---------
    def get_atr_mult(self) -> float:
        return self._atr_mult

    async def start(self):
        if self._task:
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass

    def add_signal(self, sig: Dict[str, float]):
        r = SignalRecord(
            symbol=sig["symbol"], side=sig["side"], entry=sig["entry"], sl=sig["sl"],
            tp1=sig["tp1"], tp2=sig["tp2"], tp3=sig["tp3"], created_ts=sig["ts"],
            score=sig["score"], regime=sig["regime"], reason=sig.get("reason", "")
        )
        self.records.append(r)

    def get_stats(self) -> Dict[str, float]:
        closed = [r for r in self.records if r.status != "OPEN"]
        win = [r for r in closed if r.status in ("TP1", "TP2", "TP3")]
        stop = [r for r in closed if r.status == "STOP"]
        avg_r = None
        rs = [r.realized_r for r in closed if r.realized_r is not None]
        if rs:
            avg_r = sum(rs) / len(rs)
        return {
            "total": len(self.records),
            "open": len([r for r in self.records if r.status == "OPEN"]),
            "closed": len(closed),
            "wins": len(win),
            "stops": len(stop),
            "win_rate": round(len(win)/len(closed), 4) if closed else None,
            "stop_rate": round(len(stop)/len(closed), 4) if closed else None,
            "avg_r": round(avg_r, 4) if avg_r is not None else None,
            "dyn_min_score": self.dynamic_min_score,
            "atr_mult": round(self._atr_mult, 3),
        }

    # --------- Internal Loop ---------
    async def _loop(self):
        while True:
            try:
                await self._evaluate()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print("[PERF] loop error", e)
            await asyncio.sleep(self.eval_interval)

    async def _evaluate(self):
        if not self.records:
            return
        now = time.time()
        # Fiyatları sembole göre grupla (tekli ticker çağrıları)
        symbols = {r.symbol for r in self.records if r.status == "OPEN"}
        prices: Dict[str, float] = {}
        for sym in symbols:
            try:
                t = self.engine.client.get_ticker(sym)
                prices[sym] = float(t.get("price") or 0.0)
            except Exception:
                continue
        changed = False
        for r in self.records:
            if r.status != "OPEN":
                continue
            price = prices.get(r.symbol)
            if not price:
                # TTL kontrolü yine de yapılır
                if now - r.created_ts > self.signal_ttl:
                    r.status = "EXPIRED"
                    r.closed_ts = now
                    r.realized_r = 0.0
                    changed = True
                continue
            if r.side == "LONG":
                if price <= r.sl:
                    r.status = "STOP"; r.closed_ts = now; r.realized_r = -1.0; changed = True; continue
                if price >= r.tp3:
                    r.status = "TP3"; r.closed_ts = now; r.realized_r = 2.2; changed = True; continue
                if price >= r.tp2:
                    r.status = "TP2"; r.closed_ts = now; r.realized_r = 1.6; changed = True; continue
                if price >= r.tp1:
                    r.status = "TP1"; r.closed_ts = now; r.realized_r = 1.0; changed = True; continue
            else:  # SHORT
                if price >= r.sl:
                    r.status = "STOP"; r.closed_ts = now; r.realized_r = -1.0; changed = True; continue
                if price <= r.tp3:
                    r.status = "TP3"; r.closed_ts = now; r.realized_r = 2.2; changed = True; continue
                if price <= r.tp2:
                    r.status = "TP2"; r.closed_ts = now; r.realized_r = 1.6; changed = True; continue
                if price <= r.tp1:
                    r.status = "TP1"; r.closed_ts = now; r.realized_r = 1.0; changed = True; continue
            # TTL kapatma
            if now - r.created_ts > self.signal_ttl:
                r.status = "EXPIRED"
                r.closed_ts = now
                r.realized_r = 0.0
                changed = True

        if changed:
            await self._adaptive_update()
            await self._feed_learner()

    async def _adaptive_update(self):
        closed = [r for r in self.records if r.status != "OPEN"]
        if len(closed) < 8:  # yeterli veri yok
            return
        recent = closed[-30:]
        wins = [r for r in recent if r.status in ("TP1", "TP2", "TP3")]
        stops = [r for r in recent if r.status == "STOP"]
        win_rate = len(wins)/len(recent) if recent else 0.0
        stop_rate = len(stops)/len(recent) if recent else 0.0

        # MIN_SCORE adaptasyonu
        base_min = self.engine.min_score
        new_min = base_min
        if win_rate < 0.45 and base_min < 75:
            new_min = base_min + 2
        elif win_rate > 0.60 and base_min > 40:
            new_min = base_min - 1
        if new_min != base_min:
            self.engine.min_score = new_min
            self.dynamic_min_score = new_min
            print(f"[ADAPT] MIN_SCORE {base_min} -> {new_min} (win_rate={win_rate:.2f})")

        # ATR multiplier adaptasyonu
        atr_old = self._atr_mult
        atr_new = atr_old
        if stop_rate > 0.55 and atr_old < 1.8:
            atr_new = round(atr_old + 0.05, 2)
        elif stop_rate < 0.30 and atr_old > 0.8:
            atr_new = round(atr_old - 0.05, 2)
        if atr_new != atr_old:
            self._atr_mult = atr_new
            print(f"[ADAPT] ATR_MULT {atr_old} -> {atr_new} (stop_rate={stop_rate:.2f})")

    async def _feed_learner(self):
        learner = get_global_learner()
        if not learner:
            return
        # En yeni kapananlardan son 15 tanesine bak
        closed = [r for r in self.records if r.status not in ("OPEN",)]
        if not closed:
            return
        batch = closed[-15:]
        for r in batch:
            if r.realized_r is None:
                continue
            # Label: STOP ->0, TP1 ->0.6, TP2 ->0.8, TP3 ->1.0, EXPIRED ->0.3
            mapping = {"STOP":0.0, "TP1":0.6, "TP2":0.8, "TP3":1.0, "EXPIRED":0.3}
            label = mapping.get(r.status)
            if label is None:
                continue
            feats = self._extract_features(r)
            learner.update(feats, label)

    def _extract_features(self, r: SignalRecord) -> Dict[str, float]:
        # Basit özellikler: skor, R genişliği, side, regime
        width_r = (abs(r.tp1 - r.entry) / max(1e-9, abs(r.entry - r.sl))) if r.sl else 1.0
        feats = {
            "score": r.score/100.0,
            "width_r": width_r,
            "side_long": 1.0 if r.side == "LONG" else 0.0,
            "side_short": 1.0 if r.side == "SHORT" else 0.0,
        }
        feats[f"reg_{r.regime.lower()}"] = 1.0
        return feats


# Global yardımcı (stratejiden erişim)
_GLOBAL_TRACKER: Optional[PerformanceTracker] = None


def set_global_tracker(t: PerformanceTracker):
    global _GLOBAL_TRACKER
    _GLOBAL_TRACKER = t


def get_current_atr_mult() -> float:
    if _GLOBAL_TRACKER:
        return _GLOBAL_TRACKER.get_atr_mult()
    return 1.2
