from __future__ import annotations
import asyncio
import math
import random
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Callable, Any

from kucoin.client import Market

from .config import get_settings
from .indicators import to_df_klines
from .strategies import pick_candidate, Candidate


class SignalEngine:
    """Arka planda çalışan tarama motoru.

    KuCoin sembollerini periyodik tarar, en iyi adayı seçer ve
    signal_emit callback'i ile yeni sinyal yayınlar.
    """

    def __init__(self):
        self.settings = get_settings()
        self.client = Market(url=self.settings.EXCHANGE_BASE_URL)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.last_signal_ts: Dict[str, float] = {}
        self.cooldown_sec = 1800
        self.min_score = self.settings.MIN_SCORE
        # canlı sinyaller listesi
        self.signals_live = []
        self.max_store = 200
        self._listeners = []
        self.current_symbols = []

    # ----------------- Public API -----------------
    def add_listener(self, fn: Callable[[Dict[str, Any]], None]):
        self._listeners.append(fn)

    def get_live(self) -> List[Dict[str, Any]]:
        return list(self.signals_live)

    async def start(self):
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        self._running = False
        if self._task:
            await self._task

    # ----------------- Internal -----------------
    def log(self, *a):
        print(self.settings.LOG_PREFIX, *a, flush=True)

    def _push_signal(self, sig: Dict[str, Any]):
        self.signals_live.append(sig)
        if len(self.signals_live) > self.max_store:
            self.signals_live = self.signals_live[-self.max_store :]
        for fn in self._listeners:
            try:
                fn(sig)
            except Exception as e:
                self.log("listener hata", e)

    def _get_ohlcv(self, symbol: str, interval: str, limit: int):
        try:
            raw = self.client.get_kline(symbol, interval, limit=limit)
            return to_df_klines(raw)
        except Exception as e:
            self.log("Veri hatası", symbol, interval, e)
            return None

    async def _run_loop(self):
        s = self.settings
        while self._running:
            try:
                await self._scan_symbols()
            except Exception as e:
                self.log("scan döngü hata", e)
            await asyncio.sleep(s.SLEEP_SECONDS)

    async def _scan_symbols(self):
        s = self.settings
        syms_resp = self.client.get_symbol_list()
        all_pairs = [r["symbol"] for r in syms_resp if r.get("quoteCurrency") == s.QUOTE_FILTER]
        tickers = self.client.get_all_tickers().get("ticker", [])
        volmap = {t.get("symbol"): float(t.get("volValue", 0.0)) for t in tickers}
        filt = [sym for sym in all_pairs if volmap.get(sym, 0.0) >= s.MIN_VOLVALUE_USDT]
        if not filt:
            filt = all_pairs
        random.shuffle(filt)
        filt = filt[: s.SCAN_LIMIT]
        self.current_symbols = filt
        self.log(f"Tarama: {len(filt)} sembol")

        sem = asyncio.Semaphore(s.SYMBOL_CONCURRENCY)

        async def scan_one(sym: str):
            async with sem:
                now = time.time()
                if sym in self.last_signal_ts and now - self.last_signal_ts[sym] < self.cooldown_sec:
                    return None
                df15 = self._get_ohlcv(sym, s.TF_LTF, s.LOOKBACK_LTF)
                df1h = self._get_ohlcv(sym, s.TF_HTF, s.LOOKBACK_HTF)
                if df15 is None or df1h is None:
                    return None
                cand = pick_candidate(sym, df15, df1h)
                if not cand:
                    return None
                # Basit dinamik skor filtresi
                if cand.score < self.min_score:
                    if sent == 0:  # İlk birkaç düşük skorluda log
                        self.log(f"{sym}: score {cand.score:.1f} < {self.min_score} (red)")
                    return None
                payload = asdict(cand)
                payload["ts"] = time.time()
                self.last_signal_ts[sym] = now
                return payload

        tasks = [scan_one(sym) for sym in filt]
        results = await asyncio.gather(*tasks)
        sent = 0
        for r in results:
            if not r:
                continue
            self._push_signal(r)
            sent += 1
        self.log(f"Tarama tamam, yayınlanan sinyal: {sent}")
