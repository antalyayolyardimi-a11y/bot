"""bot_core.py

README içindeki monolitik Telegram bot kodunu modüler hale getiren çekirdek.
Burada:
 - KuCoin veri erişimi
 - Sinyal aday üretimi (basitleştirilmiş demo versiyon)
 - Periodik tarayıcı coroutine'i
 - Telegram gönderim callback arayüzü (enjekte edilebilir)

Not: Orijinal README çok büyük ve tam kopyayı burada tutmak bakım maliyeti yaratır.
Bu yüzden minimal, çalışabilir ve genişletilebilir bir çekirdek koyduk.
Gerektiğinde ileri fonksiyonları README'den bölüp ekleyebilirsin.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Awaitable, Dict, List, Optional

import pandas as pd
import numpy as np
from kucoin.client import Market

# ================== PARAMETRELER ==================
TF_LTF = "15min"
TF_HTF = "1hour"
LOOKBACK_LTF = 160
LOOKBACK_HTF = 120
SLEEP_SECONDS = 300
MIN_VOLVALUE_USDT = 2_000_000
SCAN_LIMIT = 120
SYMBOL_CONCURRENCY = 8

# ================== BASİT GÖSTERGE ARAÇLARI ==================
def to_df_klines(raw):
	if not raw:
		return None
	df = pd.DataFrame(raw, columns=["time","o","c","h","l","v","turnover"])
	for col in df.columns:
		df[col] = pd.to_numeric(df[col], errors="coerce")
	df.dropna(inplace=True)
	df["time"] = pd.to_datetime(df["time"].astype(np.int64), unit="ms", utc=True)
	df.sort_values("time", inplace=True)
	df.reset_index(drop=True, inplace=True)
	return df

def ema(s: pd.Series, n: int):
	return s.ewm(span=n, adjust=False).mean()

def atr_wilder(h, l, c, n=14):
	pc = c.shift(1)
	tr = pd.concat([
		(h - l).abs(),
		(h - pc).abs(),
		(l - pc).abs()
	], axis=1).max(axis=1)
	return tr.ewm(alpha=1 / n, adjust=False).mean()

def bollinger(close, n=20, k=2.0):
	ma = close.rolling(n).mean(); std = close.rolling(n).std(ddof=0)
	upper = ma + k * std; lower = ma - k * std
	bwidth = (upper - lower) / (ma + 1e-12)
	return ma, upper, lower, bwidth

def adx(h, l, c, n=14):
	up = h.diff(); dn = -l.diff()
	plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
	minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
	atr = atr_wilder(h, l, c, n)
	pdi = 100 * pd.Series(plus_dm, index=c.index).ewm(alpha=1/n, adjust=False).mean() / (atr + 1e-12)
	ndi = 100 * pd.Series(minus_dm, index=c.index).ewm(alpha=1/n, adjust=False).mean() / (atr + 1e-12)
	dx = (np.abs(pdi - ndi) / ((pdi + ndi) + 1e-12)) * 100
	return dx.ewm(alpha=1/n, adjust=False).mean()

# ================== VERİ KATMANI ==================
class DataClient:
	def __init__(self):
		self.market = Market(url="https://api.kucoin.com")

	def list_symbols(self) -> List[str]:
		syms = self.market.get_symbol_list()
		return [s["symbol"] for s in syms if s.get("quoteCurrency") == "USDT"]

	def volumes(self) -> Dict[str, float]:
		tickers = self.market.get_all_tickers().get("ticker", [])
		return {t.get("symbol"): float(t.get("volValue", 0.0)) for t in tickers}

	def ohlcv(self, symbol: str, interval: str, limit: int):
		try:
			raw = self.market.get_kline(symbol, interval, limit=limit)
			return to_df_klines(raw)
		except Exception:
			return None

# ================== SİNYAL MODELİ ==================
@dataclass
class Signal:
	symbol: str
	side: str
	entry: float
	sl: float
	tp1: float
	tp2: float
	tp3: float
	score: float
	reason: str
	meta: Dict = field(default_factory=dict)

# ================== ADAY ÜRETİCİ ==================
def build_candidate(symbol: str, df15: pd.DataFrame, df1h: pd.DataFrame) -> Optional[Signal]:
	if df15 is None or len(df15) < 80 or df1h is None or len(df1h) < 60:
		return None
	c = df15["c"]; h=df15["h"]; l=df15["l"]
	close = float(c.iloc[-1])
	# Basit trend yönü: 1H EMA50 eğimi
	e50 = ema(df1h["c"], 50)
	bias = None
	if len(e50) >= 2 and not math.isnan(e50.iloc[-1]) and not math.isnan(e50.iloc[-2]):
		bias = "LONG" if e50.iloc[-1] > e50.iloc[-2] else ("SHORT" if e50.iloc[-1] < e50.iloc[-2] else None)
	# Bollinger genişliği + ADX basit skorlama
	_,_,_, bw = bollinger(c)
	adx15 = float(adx(h,l,c).iloc[-1])
	bw_last = float(bw.iloc[-1]) if not math.isnan(bw.iloc[-1]) else 0.1
	atrv = float(atr_wilder(h,l,c).iloc[-1])
	# LONG adayı: bias LONG ve bw dar + adx artış
	if bias == "LONG" and bw_last < 0.07 and adx15 > 18:
		sl = close - 1.2 * atrv
		tp1 = close + 1.0 * atrv
		tp2 = close + 1.6 * atrv
		tp3 = close + 2.2 * atrv
		score = 60 + (18 - 18 + (20 if bw_last < 0.05 else 10))
		return Signal(symbol, "LONG", close, sl, tp1, tp2, tp3, score, "Dar bant + trend yukarı + momentum")
	if bias == "SHORT" and bw_last < 0.07 and adx15 > 18:
		sl = close + 1.2 * atrv
		tp1 = close - 1.0 * atrv
		tp2 = close - 1.6 * atrv
		tp3 = close - 2.2 * atrv
		score = 60 + (18 - 18 + (20 if bw_last < 0.05 else 10))
		return Signal(symbol, "SHORT", close, sl, tp1, tp2, tp3, score, "Dar bant + trend aşağı + momentum")
	return None

# ================== TARAMA ==================
class Scanner:
	def __init__(self, data: DataClient, send_callback: Callable[[Signal], Awaitable[None]] | None = None):
		self.data = data
		self.send_callback = send_callback
		self._last_sent: Dict[str, float] = {}
		self.cooldown = 1800

	async def scan_symbol(self, symbol: str, sem: asyncio.Semaphore):
		async with sem:
			now = time.time()
			if symbol in self._last_sent and now - self._last_sent[symbol] < self.cooldown:
				return None
			df15 = self.data.ohlcv(symbol, TF_LTF, LOOKBACK_LTF)
			df1h = self.data.ohlcv(symbol, TF_HTF, LOOKBACK_HTF)
			cand = build_candidate(symbol, df15, df1h)
			if cand:
				self._last_sent[symbol] = now
			return cand

	async def run_once(self) -> List[Signal]:
		symbols = self.data.list_symbols()
		vols = self.data.volumes()
		filt = [s for s in symbols if vols.get(s,0.0) >= MIN_VOLVALUE_USDT]
		random.shuffle(filt)
		filt = filt[:SCAN_LIMIT]
		sem = asyncio.Semaphore(SYMBOL_CONCURRENCY)
		tasks = [self.scan_symbol(s, sem) for s in filt]
		res = await asyncio.gather(*tasks)
		signals = [r for r in res if r]
		return signals

	async def loop_forever(self):
		while True:
			signals = await self.run_once()
			if self.send_callback:
				for sig in signals:
					try:
						await self.send_callback(sig)
					except Exception:
						pass
			await asyncio.sleep(SLEEP_SECONDS)

# ================== YARDIMCI ==================
async def demo_print_sender(sig: Signal):  # CLI test
	print(f"SIGNAL {sig.symbol} {sig.side} entry={sig.entry:.6f} tp1={sig.tp1:.6f} sl={sig.sl:.6f} score={sig.score}")

async def _demo():
	sc = Scanner(DataClient(), demo_print_sender)
	await sc.loop_forever()

if __name__ == "__main__":
	asyncio.run(_demo())
