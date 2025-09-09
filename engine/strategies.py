"""Gelişmiş sinyal oluşturma stratejileri.

Akıllı multi-timeframe analiz, momentum confirmation, 
volatility filtering ve risk yönetimi.
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


def analyze_market_structure(df: pd.DataFrame) -> dict:
    """Piyasa yapısını analiz et (trend, support/resistance)."""
    close = df["c"]
    high = df["h"] 
    low = df["l"]
    
    # Trend analizi (EMA slopes)
    ema9 = ema(close, 9)
    ema21 = ema(close, 21)
    ema50 = ema(close, 50)
    
    trend_strength = 0
    if len(ema9) > 5:
        ema9_slope = (ema9.iloc[-1] - ema9.iloc[-5]) / 5
        ema21_slope = (ema21.iloc[-1] - ema21.iloc[-5]) / 5
        
        if ema9_slope > 0 and ema21_slope > 0 and ema9.iloc[-1] > ema21.iloc[-1]:
            trend_strength = 1  # Bullish
        elif ema9_slope < 0 and ema21_slope < 0 and ema9.iloc[-1] < ema21.iloc[-1]:
            trend_strength = -1  # Bearish
    
    # Support/Resistance levels
    recent_highs = high.tail(20).max()
    recent_lows = low.tail(20).min()
    
    return {
        "trend": trend_strength,
        "resistance": recent_highs,
        "support": recent_lows,
        "ema9_slope": ema9_slope if len(ema9) > 5 else 0,
        "ema21_slope": ema21_slope if len(ema9) > 5 else 0
    }


def calculate_momentum_score(df15: pd.DataFrame, df1h: pd.DataFrame) -> float:
    """Multi-timeframe momentum skoru."""
    score = 50
    
    # 15m momentum
    rsi15 = rsi(df15["c"], 14).iloc[-1] if len(df15) > 14 else 50
    adx15 = adx(df15["h"], df15["l"], df15["c"], 14).iloc[-1] if len(df15) > 20 else 20
    
    # 1h momentum  
    rsi1h = rsi(df1h["c"], 14).iloc[-1] if len(df1h) > 14 else 50
    adx1h = adx(df1h["h"], df1h["l"], df1h["c"], 14).iloc[-1] if len(df1h) > 20 else 20
    
    # RSI alignment bonus
    if 30 < rsi15 < 70 and 30 < rsi1h < 70:  # Healthy range
        score += 10
    elif (rsi15 < 30 and rsi1h < 40) or (rsi15 > 70 and rsi1h > 60):  # Oversold/Overbought alignment
        score += 5
    
    # ADX strength bonus
    if adx15 > 25 and adx1h > 20:  # Strong trend
        score += 15
    elif adx15 > 20 or adx1h > 15:  # Moderate trend
        score += 8
    
    # Volume confirmation
    recent_vol = df15["v"].tail(5).mean()
    avg_vol = df15["v"].tail(50).mean()
    if recent_vol > avg_vol * 1.2:  # Volume spike
        score += 12
    elif recent_vol > avg_vol:
        score += 6
    
    return min(100, max(0, score))


def detect_smart_money_concepts(df: pd.DataFrame) -> dict:
    """Smart Money Concepts benzeri analiz."""
    close = df["c"]
    high = df["h"]
    low = df["l"]
    
    # Order Block detection (basit versiyon)
    recent_candles = df.tail(10)
    order_blocks = []
    
    for i in range(2, len(recent_candles)-1):
        curr_high = recent_candles.iloc[i]["h"]
        curr_low = recent_candles.iloc[i]["l"]
        prev_close = recent_candles.iloc[i-1]["c"]
        next_close = recent_candles.iloc[i+1]["c"]
        
        # Bullish Order Block
        if (curr_low < prev_close and next_close > curr_high):
            order_blocks.append({"type": "bullish", "low": curr_low, "high": curr_high})
        
        # Bearish Order Block  
        elif (curr_high > prev_close and next_close < curr_low):
            order_blocks.append({"type": "bearish", "low": curr_low, "high": curr_high})
    
    # Liquidity zones (recent highs/lows)
    liquidity_high = high.tail(20).max()
    liquidity_low = low.tail(20).min()
    
    return {
        "order_blocks": order_blocks,
        "liquidity_high": liquidity_high,
        "liquidity_low": liquidity_low,
        "current_price": close.iloc[-1]
    }


def pick_candidate(symbol: str, df15: pd.DataFrame, df1h: pd.DataFrame) -> Optional[Candidate]:
    """Gelişmiş multi-timeframe sinyal analizi."""
    if df15 is None or len(df15) < 80 or df1h is None or len(df1h) < 60:
        return None
    
    o, c, h, l, v = df15["o"], df15["c"], df15["h"], df15["l"], df15["v"]
    atrv = float(atr_wilder(h, l, c, 14).iloc[-1])
    close = float(c.iloc[-1])

    # Kalite kontrolleri - daha sıkı
    recent_volume_avg = float(v.tail(10).mean()) * close
    if recent_volume_avg < 100000:  # 100K USDT minimum
        return None
        
    if atrv / close < 0.008:  # %0.8'den az volatilite varsa skip
        return None

    # Market structure analysis
    structure15 = analyze_market_structure(df15)
    structure1h = analyze_market_structure(df1h)
    
    # Momentum scoring
    momentum_score = calculate_momentum_score(df15, df1h)
    
    # Smart money concepts
    smc_data = detect_smart_money_concepts(df15)
    
    # Teknik indikatörler
    rsi_15 = rsi(c, 14).iloc[-1] if len(c) > 14 else 50
    rsi_1h = rsi(df1h["c"], 14).iloc[-1] if len(df1h) > 14 else 50
    
    # Bollinger bands
    ma, bb_u, bb_l, bwidth, std = bollinger(c, 20)
    bb_position = (close - bb_l.iloc[-1]) / (bb_u.iloc[-1] - bb_l.iloc[-1]) if not pd.isna(bb_u.iloc[-1]) else 0.5
    
    candidates: List[Candidate] = []
    base_score = momentum_score

    # LONG Sinyalleri
    if structure15["trend"] >= 0 and structure1h["trend"] >= 0:  # Bullish alignment
        long_reasons = []
        long_score = base_score
        
        # RSI oversold bounce
        if 25 < rsi_15 < 35 and rsi_1h < 50:
            long_reasons.append("RSI oversold bounce")
            long_score += 15
        
        # Bollinger lower band bounce
        if bb_position < 0.2:
            long_reasons.append("BB lower band bounce") 
            long_score += 12
        
        # Order block support
        for ob in smc_data["order_blocks"]:
            if ob["type"] == "bullish" and ob["low"] <= close <= ob["high"] * 1.02:
                long_reasons.append("Bullish order block")
                long_score += 20
                break
        
        # EMA alignment bonus
        if structure15["ema9_slope"] > 0 and structure1h["ema9_slope"] > 0:
            long_reasons.append("EMA alignment")
            long_score += 10
        
        # Volume confirmation
        if v.iloc[-1] > v.tail(10).mean() * 1.3:
            long_reasons.append("Volume spike")
            long_score += 8
        
        if long_reasons and long_score >= 65:
            sl, tps = compute_sl_tp("LONG", close, atrv)
            candidates.append(Candidate(
                symbol=symbol, side="LONG", entry=close, sl=sl,
                tp1=tps[0], tp2=tps[1], tp3=tps[2],
                score=long_score, regime="BULLISH",
                reason=" + ".join(long_reasons)
            ))

    # SHORT Sinyalleri  
    if structure15["trend"] <= 0 and structure1h["trend"] <= 0:  # Bearish alignment
        short_reasons = []
        short_score = base_score
        
        # RSI overbought rejection
        if 65 < rsi_15 < 75 and rsi_1h > 50:
            short_reasons.append("RSI overbought rejection")
            short_score += 15
        
        # Bollinger upper band rejection
        if bb_position > 0.8:
            short_reasons.append("BB upper band rejection")
            short_score += 12
        
        # Order block resistance  
        for ob in smc_data["order_blocks"]:
            if ob["type"] == "bearish" and ob["low"] * 0.98 <= close <= ob["high"]:
                short_reasons.append("Bearish order block")
                short_score += 20
                break
        
        # EMA alignment bonus
        if structure15["ema9_slope"] < 0 and structure1h["ema9_slope"] < 0:
            short_reasons.append("EMA alignment")
            short_score += 10
            
        # Volume confirmation
        if v.iloc[-1] > v.tail(10).mean() * 1.3:
            short_reasons.append("Volume spike")
            short_score += 8
        
        if short_reasons and short_score >= 65:
            sl, tps = compute_sl_tp("SHORT", close, atrv)
            candidates.append(Candidate(
                symbol=symbol, side="SHORT", entry=close, sl=sl,
                tp1=tps[0], tp2=tps[1], tp3=tps[2],
                score=short_score, regime="BEARISH", 
                reason=" + ".join(short_reasons)
            ))

    if not candidates:
        return None
        
    # AI learner probability modulation
    learner = get_global_learner()
    if learner:
        for cand in candidates:
            features = {
                "score": cand.score / 100,
                "rsi_15": rsi_15 / 100,
                "rsi_1h": rsi_1h / 100,
                "bb_pos": bb_position,
                "trend_15": structure15["trend"],
                "trend_1h": structure1h["trend"],
                "vol_ratio": float(v.iloc[-1] / v.tail(20).mean()),
                f"side_{cand.side}": 1.0
            }
            cand.prob = learner.predict_proba(features)
            # Probability ile score'u modüle et
            cand.score = cand.score * (0.7 + 0.6 * cand.prob)

    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates[0]
