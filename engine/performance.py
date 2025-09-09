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
        # Adaptif parametreler - daha konservatif başlangıç
        self.dynamic_min_score: Optional[int] = None
        self._atr_mult: float = 1.5  # 1.2'den 1.5'e çıkardım, daha geniş SL
        self._lock = asyncio.Lock()
        # Değerlendirme aralığı
        self.eval_interval = 30  # 60'dan 30'a düşürdüm, daha sık kontrol
        self.signal_ttl = 60 * 120  # 90'dan 120 dakikaya çıkardım, daha uzun süre bekle

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
        closed = [r for r in self.records if r.status not in ("OPEN",)]
        if not closed:
            return {"total": 0, "open": len([r for r in self.records if r.status == "OPEN"])}
            
        # Genel istatistikler
        total = len(closed)
        wins = [r for r in closed if r.status in ("TP1", "TP2", "TP3")]
        stops = [r for r in closed if r.status == "STOP"]
        expired = [r for r in closed if r.status == "EXPIRED"]
        
        win_rate = len(wins) / total if total > 0 else 0
        stop_rate = len(stops) / total if total > 0 else 0
        expire_rate = len(expired) / total if total > 0 else 0
        
        # Ortalama R değerleri
        avg_win_r = sum([r.realized_r for r in wins]) / len(wins) if wins else 0
        avg_loss_r = sum([abs(r.realized_r) for r in stops]) / len(stops) if stops else 0
        
        # Son 20 sinyal performansı
        recent = closed[-20:] if len(closed) >= 20 else closed
        recent_wins = [r for r in recent if r.status in ("TP1", "TP2", "TP3")]
        recent_win_rate = len(recent_wins) / len(recent) if recent else 0
        
        return {
            "total": total,
            "open": len([r for r in self.records if r.status == "OPEN"]),
            "win_rate": round(win_rate * 100, 1),
            "stop_rate": round(stop_rate * 100, 1),
            "expire_rate": round(expire_rate * 100, 1),
            "avg_win_r": round(avg_win_r, 2),
            "avg_loss_r": round(avg_loss_r, 2),
            "recent_win_rate": round(recent_win_rate * 100, 1),
            "atr_mult": self._atr_mult,
            "dynamic_min_score": self.dynamic_min_score or self.engine.min_score
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
        if len(closed) < 5:  # daha az veri ile de çalışsın
            return
        recent = closed[-20:]  # son 20 sinyal
        wins = [r for r in recent if r.status in ("TP1", "TP2", "TP3")]
        stops = [r for r in recent if r.status == "STOP"]
        win_rate = len(wins)/len(recent) if recent else 0.0
        stop_rate = len(stops)/len(recent) if recent else 0.0

        print(f"[ADAPT] Analyzing {len(recent)} recent signals: wins={len(wins)}, stops={len(stops)}, win_rate={win_rate:.2f}, stop_rate={stop_rate:.2f}")

        # MIN_SCORE adaptasyonu (daha agresif)
        base_min = self.engine.min_score
        new_min = base_min
        
        # Stop rate yüksekse minimum skoru artır (daha seçici ol)
        if stop_rate > 0.60 and base_min < 80:
            new_min = base_min + 3
        elif stop_rate > 0.45 and base_min < 70:
            new_min = base_min + 2
        # Win rate yüksekse minimum skoru azalt (daha fazla sinyal al)
        elif win_rate > 0.65 and base_min > 35:
            new_min = base_min - 2
        elif win_rate > 0.55 and base_min > 40:
            new_min = base_min - 1
            
        if new_min != base_min:
            self.engine.min_score = new_min
            self.dynamic_min_score = new_min
            print(f"[ADAPT] MIN_SCORE {base_min} -> {new_min} (win_rate={win_rate:.2f}, stop_rate={stop_rate:.2f})")

        # ATR multiplier adaptasyonu (daha agresif)
        atr_old = self._atr_mult
        atr_new = atr_old
        
        # Stop rate çok yüksekse SL'ları genişlet
        if stop_rate > 0.65 and atr_old < 2.0:
            atr_new = round(atr_old + 0.1, 2)
        elif stop_rate > 0.50 and atr_old < 1.8:
            atr_new = round(atr_old + 0.05, 2)
        # Stop rate düşükse SL'ları daralt 
        elif stop_rate < 0.25 and atr_old > 0.9:
            atr_new = round(atr_old - 0.05, 2)
        elif stop_rate < 0.35 and atr_old > 1.0:
            atr_new = round(atr_old - 0.03, 2)
            
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
        
        # Öğrenmemiş kayıtları bul (yeni kapanan sinyaller)
        unlearned = [r for r in closed if not hasattr(r, '_learned')]
        if not unlearned:
            return
            
        print(f"[LEARNER] Processing {len(unlearned)} new closed signals for learning...")
        
        for r in unlearned:
            if r.realized_r is None:
                continue
            # Label: STOP ->0, TP1 ->0.6, TP2 ->0.8, TP3 ->1.0, EXPIRED ->0.3
            mapping = {"STOP":0.0, "TP1":0.6, "TP2":0.8, "TP3":1.0, "EXPIRED":0.3}
            label = mapping.get(r.status)
            if label is None:
                continue
            feats = self._extract_features(r)
            old_prob = learner.predict_proba(feats)
            new_prob = learner.update(feats, label)
            
            # Öğrenme işlemini logla
            print(f"[LEARNER] {r.symbol} {r.side} {r.status}: prob {old_prob:.3f}->{new_prob:.3f}, features: {feats}")
            r._learned = True  # Bu kayıt öğrenildi olarak işaretle

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
