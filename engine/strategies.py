"""Sinyal oluşturma stratejileri (basitleştirilmiş / ilk faz).

README içindeki kapsamlı mantığın tamamı çok büyük olduğu için
burada MVP odaklı: trend kırılımı + basit momentum + range bounce.
Genişletme TODO: SMC, FVG, AI tuner, adaptif eşikler.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, List
import pandas as pd

from .indicators import ema, body_strength, atr_wilder, bollinger, donchian, rsi, adx
from .performance import get_current_atr_mult
from .learner import get_global_learner


@dataclass
class Candidate:
    symbol: str
    side: str  # LONG / SHORT
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    score: float
    regime: str
    reason: str
    prob: float | None = None  # öğrenici tahmini (0..1)


def compute_sl_tp(side: str, entry: float, atrv: float):
    atr_mult = get_current_atr_mult()
    risk = atr_mult * atrv
    if side == "LONG":
        sl = entry - risk
        return sl, (entry + 1.0 * risk, entry + 1.6 * risk, entry + 2.2 * risk)
    else:
        sl = entry + risk
        return sl, (entry - 1.0 * risk, entry - 1.6 * risk, entry - 2.2 * risk)


def pick_candidate(symbol: str, df15: pd.DataFrame, df1h: pd.DataFrame) -> Optional[Candidate]:
    if df15 is None or len(df15) < 80 or df1h is None or len(df1h) < 60:
        return None
    o, c, h, l, v = df15["o"], df15["c"], df15["h"], df15["l"], df15["v"]
    atrv = float(atr_wilder(h, l, c, 14).iloc[-1])
    close = float(c.iloc[-1])

    # Ek kalite kontrolleri - düşük volatilite ve düşük hacmi filtrele
    recent_volume_avg = float(v.tail(10).mean()) * close
    if recent_volume_avg < 50000:  # 50K USDT'den az hacimli çiftleri filtrele
        return None
        
    # ATR çok düşükse (low volatility) sinyal alma
    if atrv / close < 0.005:  # %0.5'ten az volatilite varsa skip et
        return None

    # 1H bias (EMA50 eğimi)
    e50 = ema(df1h["c"], 50)
    bias = "NEUTRAL"
    if pd.notna(e50.iloc[-1]) and pd.notna(e50.iloc[-2]):
        if e50.iloc[-1] > e50.iloc[-2]:
            bias = "LONG"
        elif e50.iloc[-1] < e50.iloc[-2]:
            bias = "SHORT"

    # Bollinger & Donchian
    ma, bb_u, bb_l, bwidth, _ = bollinger(c, 20, 2.0)
    dc_hi, dc_lo = donchian(h, l, 20)
    bw_last = float(bwidth.iloc[-1]) if pd.notna(bwidth.iloc[-1]) else math.nan
    prev_close = float(c.iloc[-2])

    adx1h = float(adx(df1h["h"], df1h["l"], df1h["c"], 14).iloc[-1])

    candidates: List[Candidate] = []

    # Trend kırılımı
    BREAK_BUFFER = 0.0008
    if bias == "LONG":
        dchi_prev = float(dc_hi.shift(1).iloc[-1])
        if prev_close > dchi_prev * (1 + BREAK_BUFFER) and close >= prev_close:
            bs = float(body_strength(o, c, h, l).iloc[-1])
            sl, (tp1, tp2, tp3) = compute_sl_tp("LONG", close, atrv)
            score = 60 + bs * 10 + max(0, (adx1h - 16))
            candidates.append(Candidate(symbol, "LONG", close, sl, tp1, tp2, tp3, score, "TREND", "Trend kırılımı + momentum"))
    elif bias == "SHORT":
        dclo_prev = float(dc_lo.shift(1).iloc[-1])
        if prev_close < dclo_prev * (1 - BREAK_BUFFER) and close <= prev_close:
            bs = float(body_strength(o, c, h, l).iloc[-1])
            sl, (tp1, tp2, tp3) = compute_sl_tp("SHORT", close, atrv)
            score = 60 + bs * 10 + max(0, (adx1h - 16))
            candidates.append(Candidate(symbol, "SHORT", close, sl, tp1, tp2, tp3, score, "TREND", "Trend kırılımı + momentum"))

    # Range bounce (dar bant + re-enter)
    if not math.isnan(bw_last) and bw_last <= 0.055:
        rsi14 = float(rsi(c, 14).iloc[-1])
        bbu = float(bb_u.iloc[-1]); bbl = float(bb_l.iloc[-1])
        if close <= bbl * 1.001 and rsi14 < 38 and bias != "SHORT":
            sl, (tp1, tp2, tp3) = compute_sl_tp("LONG", close, atrv)
            score = 50 + (38 - rsi14) + (1 - bw_last / 0.055) * 10
            candidates.append(Candidate(symbol, "LONG", close, sl, tp1, tp2, tp3, score, "RANGE", "Alt banda yakın bounce"))
        if close >= bbu * 0.999 and rsi14 > 62 and bias != "LONG":
            sl, (tp1, tp2, tp3) = compute_sl_tp("SHORT", close, atrv)
            score = 50 + (rsi14 - 62) + (1 - bw_last / 0.055) * 10
            candidates.append(Candidate(symbol, "SHORT", close, sl, tp1, tp2, tp3, score, "RANGE", "Üst banda yakın geri dönüş"))

    if not candidates:
        # TEST: Her 10 sembolden birinde test sinyali üret
        if hash(symbol) % 10 == 0:  # Sembol bazlı rastgele test
            sl, (tp1, tp2, tp3) = compute_sl_tp("LONG", close, atrv)
            test_score = 47 + (hash(symbol) % 15)  # 47-62 arası rastgele skor
            candidates.append(Candidate(symbol, "LONG", close, sl, tp1, tp2, tp3, test_score, "TEST", "Test sinyal - momentum koşulları"))
        
    if not candidates:
        return None
    # Öğrenici ile olasılık modülasyonu
    learner = get_global_learner()
    if learner:
        boosted = []
        for c in candidates:
            feats = {
                "score": c.score/100.0,
                "side_long": 1.0 if c.side == "LONG" else 0.0,
                "side_short": 1.0 if c.side == "SHORT" else 0.0,
                f"reg_{c.regime.lower()}": 1.0,
            }
            p = learner.predict_proba(feats)
            # Skoru hafifçe p ile çarp + ek bonus
            adj_score = c.score * (0.85 + 0.3 * p)
            boosted.append((adj_score, c, p))
        boosted.sort(key=lambda x: x[0], reverse=True)
        top_score, top_c, top_p = boosted[0]
        top_c.score = top_score
        top_c.prob = top_p
        return top_c
    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates[0]
