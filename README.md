# ================== INSTALL (Colab: ilk çalıştırmada açın) ==================
!pip -q install kucoin-python aiogram nest_asyncio pandas numpy

# ================== IMPORTS ==================
import os, nest_asyncio, asyncio, time, math, random, sys, warnings
import datetime as dt
import pandas as pd
import numpy as np
from kucoin.client import Market
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.filters import Command
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ================== TELEGRAM TOKEN ==================
# İSTEK ÜZERİNE SABİT (kendi riskiniz)
TELEGRAM_TOKEN = "8455231287:AAFIt52_C5VMWB6KHHeZHdFKXDlO2seXXVc"

# ================== GLOBAL PARAMS (balanced+ varsayılan) ==================
TF_LTF              = "15min"
TF_HTF              = "1hour"
LOOKBACK_LTF        = 320
LOOKBACK_HTF        = 180

SLEEP_SECONDS       = 300       # 5 dk tarama
SYMBOL_CONCURRENCY  = 8
SCAN_LIMIT          = 260

# Orta-sıkı preset (balanced+)
MIN_VOLVALUE_USDT   = 2_000_000
BASE_MIN_SCORE      = 68
FALLBACK_MIN_SCORE  = 62
TOP_N_PER_SCAN      = 2
COOLDOWN_SEC        = 1800
OPPOSITE_MIN_BARS   = 2

ADX_TREND_MIN       = 18
ONEH_DISP_BODY_MIN  = 0.55
ONEH_DISP_LOOKBACK  = 2

BB_PERIOD           = 20
BB_K                = 2.0
BWIDTH_RANGE        = 0.055
DONCHIAN_WIN        = 20
BREAK_BUFFER        = 0.0008
RETEST_TOL_ATR      = 0.25

SWING_LEFT          = 2
SWING_RIGHT         = 2
SWEEP_EPS           = 0.0005
BOS_EPS             = 0.0005
FVG_LOOKBACK        = 20
OTE_LOW, OTE_HIGH   = 0.62, 0.79
SMC_REQUIRE_FVG     = True

ATR_PERIOD          = 14
SWING_WIN           = 10
MAX_SL_ATRx         = 2.0
MIN_SL_ATRx         = 0.30
TPS_R               = (1.0, 1.6, 2.2)

# ===== ATR+R risk ayarları =====
USE_ATR_R_RISK   = True      # SL/TP swing yerine ATR+R kullan
ATR_STOP_MULT    = 1.2       # balanced için varsayılan; mode'a göre değişecek

# ===== Momentum onay ayarları =====
# Seçenekler: "off", "strict3", "2of3", "net_body", "ema_rv", "hybrid"
MOMO_CONFIRM_MODE = "hybrid"
MOMO_BODY_MIN     = 0.50     # gövde/oran tabanı
MOMO_REL_VOL      = 1.35     # relatif hacim eşiği (v[-1] > 20MA * MOMO_REL_VOL)
MOMO_NET_BODY_TH  = 1.20     # net gövde eşiği (3 mumun işaretli gövde toplamı)

FALLBACK_ENABLE     = False
FBB_EPS             = 0.0003
FBB_ATR_MIN         = 0.0010
FBB_ATR_MAX         = 0.028

EVAL_BARS_AHEAD     = 12
ADAPT_MIN_SAMPLES   = 20
ADAPT_WINDOW        = 60
ADAPT_UP_THRESH     = 0.55
ADAPT_DN_THRESH     = 0.35
ADAPT_STEP          = 2
MIN_SCORE_FLOOR     = 58
MIN_SCORE_CEIL      = 78

PRINT_PREFIX        = "📟"

# ---- LOG AYARLARI ----
VERBOSE_SCAN = True
SHOW_SYMBOL_LIST_AT_START = True
SHOW_SKIP_REASONS = True
CHUNK_PRINT = 20

# ================== SCORING CONFIG ==================
SCORING_WEIGHTS = {
    "htf_align": 18.0,
    "adx_norm": 14.0,
    "ltf_momo": 10.0,
    "rr_norm": 16.0,      # RR puanı devre dışı (aşağıda 0'a çekildi)
    "bw_adv": 5.0,
    "retest_or_fvg": 8.0,
    "atr_sweet": 3.0,
    "vol_pct": 8.0,
    "recent_penalty": -3.0,
}
SCORING_BASE   = 20.0
PROB_CALIB_A   = 0.10
PROB_CALIB_B   = -7.0

# ===== SELF-LEARN / AUTO-TUNER =====
AUTO_TUNER_ON     = True
WR_TARGET         = 0.52     # hedef başarı oranı
WIN_MIN_SAMPLES   = 20
TUNE_WINDOW       = 80
TUNE_COOLDOWN_SEC = 900
_last_tune_ts     = 0

# Sınır korumaları
BOUNDS = {
    "BASE_MIN_SCORE": (56, 80),
    "ADX_TREND_MIN":  (12, 26),
    "BWIDTH_RANGE":   (0.045, 0.090),
    "VOL_MULT_REQ":   (1.10, 1.80),
}

# RANGE hacim eşiği (tuner bunu oynatacak)
VOL_MULT_REQ_GLOBAL = 1.40

# RR puanlamasını kapat (SL'i manuel yöneteceksin)
SCORING_WEIGHTS["rr_norm"] = 0.0

# ================== INIT ==================
client = Market(url="https://api.kucoin.com")
bot    = Bot(token=TELEGRAM_TOKEN)
dp     = Dispatcher()

_cached_chat_id   = None
last_signal_ts    = {}
position_state    = {}
signals_store     = {}
_sid_counter      = 0

dyn_MIN_SCORE     = BASE_MIN_SCORE
signals_history   = []

# Hacim persentilleri cache
VOL_PCT_CACHE     = {}

# Takip özelliği
FOLLOWED = {}          # chat_id -> set(symbols)
FOLLOW_STATE = {}      # (chat,symbol) -> state dict
FOLLOW_LAST_TS = {}    # (chat,symbol) -> last push ts

# Adaptif gevşetme (sinyal çıkmayan turlarda)
_empty_scans = 0
_relax_acc   = 0
EMPTY_LIMIT  = 3
RELAX_STEP   = 2
RELAX_MAX    = 6

# ================== HELPERS ==================
def log(*a): print(PRINT_PREFIX, *a); sys.stdout.flush()
def now_utc(): return dt.datetime.now(dt.timezone.utc)

def to_df_klines(raw):
    # KuCoin: [time, open, close, high, low, volume, turnover]
    if not raw: return None
    df = pd.DataFrame(raw, columns=["time","o","c","h","l","v","turnover"])
    for col in ["time","o","c","h","l","v","turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    df["time"] = pd.to_datetime(df["time"].astype(np.int64), unit="ms", utc=True)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ===== KuCoin sembol normalizasyonu =====
KNOWN_QUOTES = ["USDT", "USDC", "BTC", "ETH", "TUSD", "EUR", "KCS"]
_SYMBOLS_SET = None

def _load_symbols_set():
    global _SYMBOLS_SET
    try:
        syms = client.get_symbol_list()
        _SYMBOLS_SET = set(s["symbol"].upper() for s in syms)
    except Exception as e:
        log("Sembol listesi alınamadı:", e)
        _SYMBOLS_SET = set()

def normalize_symbol_to_kucoin(user_sym: str):
    """WIFUSDT / wif-usdt / WIF/USDT → WIF-USDT, ve gerçekten var mı kontrol eder."""
    if not user_sym:
        return None
    s = user_sym.strip().upper().replace(" ", "").replace("_", "-").replace("/", "-")
    if "-" in s:
        cand = s
    else:
        cand = None
        for q in KNOWN_QUOTES:
            if s.endswith(q):
                base = s[: -len(q)]
                if base:
                    cand = f"{base}-{q}"
                    break
        if cand is None:
            cand = s
    global _SYMBOLS_SET
    if _SYMBOLS_SET is None:
        _load_symbols_set()
    if cand in _SYMBOLS_SET:
        return cand
    alts = [cand.replace("--", "-")]
    if "-" not in s:
        for q in KNOWN_QUOTES:
            if s.endswith(q):
                base = s[: -len(q)]
                if base:
                    alts.append(f"{base}-{q}")
    if "-" not in cand and all(not cand.endswith(q) for q in KNOWN_QUOTES):
        alts.append(f"{cand}-USDT")
    for a in alts:
        if a in _SYMBOLS_SET:
            return a
    return None

def get_ohlcv(symbol, interval, limit):
    try:
        raw = client.get_kline(symbol, interval, limit=limit)
        return to_df_klines(raw)
    except Exception as e:
        msg = str(e)
        if "Unsupported trading pair" in msg or '"code":"400100"' in msg:
            log(f"❗ Desteklenmeyen parite: {symbol} (KuCoin formatı 'BASE-QUOTE' olmalı, örn. WIF-USDT)")
        else:
            log(f"{symbol} {interval} veri hatası:", e)
        return None

def ema(s, n): return pd.Series(s).ewm(span=n, adjust=False).mean()

def rsi(series, period=14):
    d = series.diff()
    up = d.clip(lower=0.0); dn = (-d).clip(lower=0.0)
    ru = up.rolling(period).mean(); rd = dn.rolling(period).mean()
    rs = ru/(rd+1e-12); return 100 - (100/(1+rs))

def _series_like(x, idx): return x if isinstance(x, pd.Series) else pd.Series(x, index=idx)

def body_strength(o, c, h, l):
    o = _series_like(o, c.index); h = _series_like(h, c.index); l = _series_like(l, c.index)
    body = np.abs(c.to_numpy() - o.to_numpy())
    rng  = np.abs(h.to_numpy() - l.to_numpy())
    rng[rng == 0] = np.nan
    val = body / rng
    return pd.Series(np.nan_to_num(val, nan=0.0), index=c.index)

def atr_wilder(h, l, c, n=14):
    pc = c.shift(1)
    tr1 = np.abs((h - l).to_numpy())
    tr2 = np.abs((h - pc).to_numpy())
    tr3 = np.abs((l - pc).to_numpy())
    tr  = pd.Series(np.maximum.reduce([tr1, tr2, tr3]), index=c.index)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def adx(h, l, c, n=14):
    up = h.diff(); dn = -l.diff()
    plus_dm  = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    atr = atr_wilder(h, l, c, n)
    pdi = 100 * pd.Series(plus_dm,  index=c.index).ewm(alpha=1/n, adjust=False).mean()/(atr+1e-12)
    ndi = 100 * pd.Series(minus_dm, index=c.index).ewm(alpha=1/n, adjust=False).mean()/(atr+1e-12)
    dx  = (np.abs(pdi - ndi) / ((pdi + ndi) + 1e-12)) * 100
    return pd.Series(dx, index=c.index).ewm(alpha=1/n, adjust=False).mean()

def bollinger(close, n=20, k=2.0):
    ma = close.rolling(n).mean(); std = close.rolling(n).std(ddof=0)
    upper = ma + k*std; lower = ma - k*std
    bwidth = (upper - lower) / (ma + 1e-12)
    return ma, upper, lower, bwidth, std

def donchian(h, l, win=20):
    return h.rolling(win).max(), l.rolling(win).min()

def swing_low(lows, win=10):  return lows.iloc[-win:].min()
def swing_high(highs, win=10): return highs.iloc[-win:].max()
def fmt(x): return f"{x:.6f}"
def sigmoid(x): return 1/(1+math.exp(-x))

# ===== Scoring helpers =====
def normalize_adx(adx_val):
    return max(0.0, min(1.0, (adx_val - ADX_TREND_MIN) / 20.0))

def normalize_rr(rr1):
    return max(0.0, min(1.0, (rr1 - 0.8) / 1.6))

def bw_advantage(bw):
    if math.isnan(bw): return 0.0
    return max(0.0, 1.0 - (bw / max(1e-6, BWIDTH_RANGE)))

def atr_in_sweet(atr_pct):
    return 1.0 if FBB_ATR_MIN <= atr_pct <= FBB_ATR_MAX else 0.0

_recent_penalty = {}
PENALTY_DECAY = 2

def mark_symbol_outcome(symbol, res):
    if res == "SL":
        _recent_penalty[symbol] = PENALTY_DECAY

def use_recent_penalty(symbol):
    p = _recent_penalty.get(symbol, 0)
    if p > 0:
        _recent_penalty[symbol] = p - 1
        return 1.0
    return 0.0

def score_to_prob(score):
    return 1.0 / (1.0 + math.exp(-(PROB_CALIB_A*score + PROB_CALIB_B)))

# ================== SMC & GATE ==================
def find_swings(h, l, left=2, right=2):
    sh_idx, sl_idx = [], []
    for i in range(left, len(h)-right):
        wh = h.iloc[i-left:i+right+1]; wl = l.iloc[i-left:i+right+1]
        if h.iloc[i] == wh.max() and (wh.idxmax()==h.index[i]): sh_idx.append(i)
        if l.iloc[i] == wl.min() and (wl.idxmin()==l.index[i]): sl_idx.append(i)
    return sh_idx, sl_idx

def find_fvgs(h, l, lookback=20):
    bulls, bears = [], []
    start = max(2, len(h)-lookback)
    for i in range(start, len(h)):
        try:
            if l.iloc[i] > h.iloc[i-2]: bulls.append((h.iloc[i-2], l.iloc[i]))
            if h.iloc[i] < l.iloc[i-2]: bears.append((h.iloc[i], l.iloc[i-2]))
        except: pass
    return bulls[-1] if bulls else None, bears[-1] if bears else None

def htf_gate_and_bias(df1h):
    c,h,l,o = df1h["c"], df1h["h"], df1h["l"], df1h["o"]
    e50 = ema(c, 50)
    bias = "NEUTRAL"
    if pd.notna(e50.iloc[-1]) and pd.notna(e50.iloc[-2]):
        if e50.iloc[-1] > e50.iloc[-2]: bias="LONG"
        elif e50.iloc[-1] < e50.iloc[-2]: bias="SHORT"
    disp_ok = False
    for i in range(1, ONEH_DISP_LOOKBACK+1):
        rng   = float(h.iloc[-i] - l.iloc[-i])
        body  = abs(float(c.iloc[-i] - o.iloc[-i]))
        if rng>0 and (body/rng) >= ONEH_DISP_BODY_MIN:
            disp_ok = True; break
    adx1h = float(adx(h,l,c,14).iloc[-1])
    trend_ok = adx1h >= ADX_TREND_MIN
    return bias, disp_ok, adx1h, trend_ok

def htf_bias_only(df1h):
    c = df1h["c"]; e50 = ema(c, 50)
    if pd.isna(e50.iloc[-1]) or pd.isna(e50.iloc[-2]):
        return "NEUTRAL"
    return "LONG" if e50.iloc[-1] > e50.iloc[-2] else ("SHORT" if e50.iloc[-1] < e50.iloc[-2] else "NEUTRAL")

# ===== ATR tabanlı SL/TP ve esnek momentum onayı =====
def compute_sl_tp(side, entry, atrv):
    """ATR tabanlı SL ve R-multiples TP'ler."""
    risk = ATR_STOP_MULT * atrv
    if side == "LONG":
        sl  = entry - risk
        tps = (entry + TPS_R[0]*risk, entry + TPS_R[1]*risk, entry + TPS_R[2]*risk)
    else:
        sl  = entry + risk
        tps = (entry - TPS_R[0]*risk, entry - TPS_R[1]*risk, entry - TPS_R[2]*risk)
    return sl, tps

def _bar_body_ratio(o, c, h, l, i):
    rng = float(h.iloc[i]-l.iloc[i]);
    return 0.0 if rng<=0 else abs(float(c.iloc[i]-o.iloc[i]))/rng

def confirm_momentum(df15, side, mode=None):
    """3-mum kuralını esnekleştir: farklı onay modları (default: 'hybrid')."""
    mode = (mode or MOMO_CONFIRM_MODE).lower()
    if mode == "off":
        return True
    o,c,h,l,v = df15["o"], df15["c"], df15["h"], df15["l"], df15["v"]
    br = [_bar_body_ratio(o,c,h,l,i) for i in [-1,-2,-3]]  # gövde/menzil oranları
    dirflags = [1 if c.iloc[i]>o.iloc[i] else (-1 if c.iloc[i]<o.iloc[i] else 0) for i in [-1,-2,-3]]
    up_count   = sum(1 for d,b in zip(dirflags,br) if d==1  and b>=MOMO_BODY_MIN)
    down_count = sum(1 for d,b in zip(dirflags,br) if d==-1 and b>=MOMO_BODY_MIN)
    net_body   = sum(d*b for d,b in zip(dirflags, br))
    relvol     = float(v.iloc[-1]) > float(v.rolling(20).mean().iloc[-1]) * MOMO_REL_VOL
    e21        = ema(c,21)
    above_e21  = float(c.iloc[-1]) > float(e21.iloc[-1])

    if mode == "strict3":
        all_green = all(c.iloc[i]>o.iloc[i] for i in [-1,-2,-3]) and ((br[0]>=0.55 and br[1]>=0.55) or (br[0]>=0.50 and br[1]>=0.50 and br[2]>=0.50))
        all_red   = all(c.iloc[i]<o.iloc[i] for i in [-1,-2,-3]) and ((br[0]>=0.55 and br[1]>=0.55) or (br[0]>=0.50 and br[1]>=0.50 and br[2]>=0.50))
        return all_green if side=="LONG" else all_red
    if mode == "2of3":
        return (up_count>=2) if side=="LONG" else (down_count>=2)
    if mode == "net_body":
        return (net_body >=  MOMO_NET_BODY_TH) if side=="LONG" else (net_body <= -MOMO_NET_BODY_TH)
    if mode == "ema_rv":
        return (above_e21 and relvol) if side=="LONG" else ((not above_e21) and relvol)
    if mode == "hybrid":
        return ((up_count>=2) if side=="LONG" else (down_count>=2)) or ((above_e21 and relvol) if side=="LONG" else ((not above_e21) and relvol))
    return True

def build_smc_candidate(symbol, df15, df1h):
    bias = htf_bias_only(df1h)
    o,c,h,l,v = df15["o"],df15["c"],df15["h"],df15["l"],df15["v"]
    atrv = float(atr_wilder(h,l,c, ATR_PERIOD).iloc[-1])
    sh, sl_idx = find_swings(h, l, SWING_LEFT, SWING_RIGHT)
    if len(sh)<2 and len(sl_idx)<2:
        return None
    last_close = c.iloc[-1]

    # ==== LONG ====
    if len(sl_idx)>=2 and bias=="LONG":
        ref_low = l.iloc[sl_idx[-2]]
        swept_low = (l.iloc[sl_idx[-1]] < ref_low*(1-SWEEP_EPS)) and (c.iloc[sl_idx[-1]] > ref_low*(1-SWEEP_EPS))
        minor_sh = next((idx for idx in reversed(sh) if idx>=sl_idx[-1]), sh[-1] if sh else None)
        choch_up = (minor_sh is not None) and (last_close > h.iloc[minor_sh]*(1+BOS_EPS))
        if swept_low and choch_up:
            bull_fvg,_ = find_fvgs(h,l,FVG_LOOKBACK)
            if SMC_REQUIRE_FVG and not bull_fvg:
                return None
            leg_low  = l.iloc[sl_idx[-1]]
            leg_high = max(last_close, h.iloc[minor_sh]) if minor_sh is not None else last_close
            leg = abs(leg_high - leg_low)
            if leg / (last_close + 1e-12) < 0.004:  # mikro dalga ele
                return None
            ote_a = leg_low + (leg_high-leg_low)*OTE_LOW
            ote_b = leg_low + (leg_high-leg_low)*OTE_HIGH
            entry_a, entry_b = (bull_fvg[0], bull_fvg[1]) if bull_fvg else (ote_a, ote_b)
            entry_mid = (entry_a+entry_b)/2
            sl, (tp1,tp2,tp3) = compute_sl_tp("LONG", entry_mid, atrv)
            rr1 = (tp1-entry_mid)/max(1e-9, entry_mid-sl)
            score = 45 + min(15, rr1*10)
            return {"symbol":symbol,"side":"LONG","entry":entry_mid,"tps":(tp1,tp2,tp3),
                    "sl":sl,"score":score,"p":sigmoid((score-65)/7),"regime":"SMC",
                    "reason":"SMC: likidite süpürme → CHOCH (+FVG/OTE)"}

    # ==== SHORT ====
    if len(sh)>=2 and bias=="SHORT":
        ref_high = h.iloc[sh[-2]]
        swept_high = (h.iloc[sh[-1]] > ref_high*(1+SWEEP_EPS)) and (c.iloc[sh[-1]] < ref_high*(1+SWEEP_EPS))
        minor_sl = next((idx for idx in reversed(sl_idx) if idx>=sh[-1]), sl_idx[-1] if sl_idx else None)
        choch_dn = (minor_sl is not None) and (last_close < l.iloc[minor_sl]*(1-BOS_EPS))
        if swept_high and choch_dn:
            _,bear_fvg = find_fvgs(h,l, FVG_LOOKBACK)
            if SMC_REQUIRE_FVG and not bear_fvg:
                return None
            leg_high = h.iloc[sh[-1]]
            leg_low  = min(last_close, l.iloc[minor_sl]) if minor_sl is not None else last_close
            leg = abs(leg_high - leg_low)
            if leg / (last_close + 1e-12) < 0.004:
                return None
            ote_a = leg_high - (leg_high-leg_low)*OTE_LOW
            ote_b = leg_high - (leg_high-leg_low)*OTE_HIGH
            entry_a, entry_b = (bear_fvg[0], bear_fvg[1]) if bear_fvg else (ote_a, ote_b)
            entry_mid = (entry_a+entry_b)/2
            sl, (tp1,tp2,tp3) = compute_sl_tp("SHORT", entry_mid, atrv)
            rr1 = (entry_mid-tp1)/max(1e-9, sl-entry_mid)
            score = 45 + min(15, rr1*10)
            return {"symbol":symbol,"side":"SHORT","entry":entry_mid,"tps":(tp1,tp2,tp3),
                    "sl":sl,"score":score,"p":sigmoid((score-65)/7),"regime":"SMC",
                    "reason":"SMC: likidite süpürme → CHOCH (+FVG/OTE)"}
    return None

# ================== FEATURE/SCORING ==================
def extract_features_for_scoring(symbol, dfL, df1h, candidate, extra_ctx=None):
    c,h,l,o = dfL["c"], dfL["h"], dfL["l"], dfL["o"]
    close = float(c.iloc[-1])
    adxv  = float(adx(h,l,c,14).iloc[-1])

    entry, tp1, sl = float(candidate["entry"]), float(candidate["tps"][0]), float(candidate["sl"])
    rr1 = (tp1-entry)/max(1e-9, entry-sl) if candidate["side"]=="LONG" else (entry-tp1)/max(1e-9, sl-entry)

    _,_,_, bwidth, _ = bollinger(c, BB_PERIOD, BB_K)
    bw_last = float(bwidth.iloc[-1]) if pd.notna(bwidth.iloc[-1]) else float("nan")

    b1h, _, _, _ = htf_gate_and_bias(df1h)
    htf_align = (b1h == candidate["side"])

    ltf_is_ok = momentum_ok(dfL, candidate["side"])
    has_retest_or_fvg = ("Retest" in candidate.get("reason","")) or (candidate.get("regime")=="SMC")

    atrv = float(atr_wilder(h,l,c,ATR_PERIOD).iloc[-1])
    atr_pct = atrv/(close+1e-12)

    vol_pct = (extra_ctx or {}).get("vol_pct", 0.5)

    feats = {
        "htf_align": 1.0 if htf_align else 0.0,
        "adx_norm": normalize_adx(adxv),
        "ltf_momo": 1.0 if ltf_is_ok else 0.0,
        "rr_norm": normalize_rr(rr1),      # ağırlığı 0 olduğundan puana etkisi yok
        "bw_adv": bw_advantage(bw_last),
        "retest_or_fvg": 1.0 if has_retest_or_fvg else 0.0,
        "atr_sweet": atr_in_sweet(atr_pct),
        "vol_pct": max(0.0, min(1.0, vol_pct)),
        "recent_penalty": use_recent_penalty(symbol),
    }
    explain = {"rr1": rr1, "adx": adxv, "bw": bw_last, "atr_pct": atr_pct, "b1h": b1h}
    return feats, explain

def composite_score_from_feats(feats):
    s = SCORING_BASE
    for k,w in SCORING_WEIGHTS.items():
        s += w * float(feats.get(k,0.0))
    return max(0.0, s)

def apply_scoring(symbol, dfL, df1h, cand, extra_ctx=None):
    feats, explain = extract_features_for_scoring(symbol, dfL, df1h, cand, extra_ctx)
    score = composite_score_from_feats(feats)

    # --- TA tabanlı sert kurallar ---
    htf_align = float(feats.get("htf_align", 0.0)) >= 1.0
    adx_norm  = float(feats.get("adx_norm", 0.0))
    bw_adv    = float(feats.get("bw_adv", 0.0))

    if not htf_align:
        score -= 10.0  # 1H bias ≠ sinyal yönü

    # Trend çok zayıfsa eleriz.
    if adx_norm < 0.10:
        score = 0.0

    if cand.get("regime") == "RANGE" and bw_adv < 0.20:
        score -= 6.0

    score = max(0.0, score)

    prob  = score_to_prob(score)
    cand["score"] = score
    cand["p"]     = prob
    cand["_feats"]   = feats
    cand["_explain"] = explain
    return cand

# ================== MINI AI (ONLINE LOGIT) ==================
AI_ENABLED   = True
AI_LR        = 0.02
AI_L2        = 1e-4
AI_INIT_BIAS = -2.0

_ai_w = {k: 0.0 for k in SCORING_WEIGHTS.keys()}
_ai_b = AI_INIT_BIAS
_ai_seen = 0

def _sigm(x): return 1.0/(1.0+math.exp(-x))
def ai_predict_proba(feats: dict):
    z = _ai_b
    for k,v in feats.items():
        z += _ai_w.get(k,0.0) * float(v)
    return _sigm(z)

def ai_update_online(feats: dict, y: int):
    global _ai_b, _ai_w, _ai_seen
    p = ai_predict_proba(feats)
    err = (p - y)
    _ai_b -= AI_LR * (err + AI_L2*_ai_b)
    for k,v in feats.items():
        g = err*float(v) + AI_L2*_ai_w[k]
        _ai_w[k] -= AI_LR * g
    _ai_seen += 1

def enrich_with_ai(cand):
    if not AI_ENABLED:
        cand["p_final"] = cand.get("p",0.5); return cand
    feats = cand.get("_feats", {})
    ai_p  = ai_predict_proba(feats)
    cand["ai_p"]   = ai_p
    cand["p_final"]= (cand.get("p",0.5) + ai_p) / 2.0
    return cand

# ================== MOD / MODE KOMUTU ==================
MODE = "balanced"

def apply_mode(mode):
    global MODE, MIN_VOLVALUE_USDT, BASE_MIN_SCORE, FALLBACK_MIN_SCORE, TOP_N_PER_SCAN
    global COOLDOWN_SEC, ADX_TREND_MIN, ONEH_DISP_BODY_MIN, BWIDTH_RANGE, BREAK_BUFFER
    global RETEST_TOL_ATR, SMC_REQUIRE_FVG, FBB_ATR_MIN, FBB_ATR_MAX, FALLBACK_ENABLE
    global ATR_STOP_MULT
    MODE = mode.lower()

    if MODE == "aggressive":
        MIN_VOLVALUE_USDT = 700_000
        BASE_MIN_SCORE    = 45
        FALLBACK_MIN_SCORE= 55
        TOP_N_PER_SCAN    = 5
        COOLDOWN_SEC      = 900
        ADX_TREND_MIN     = 14
        ONEH_DISP_BODY_MIN= 0.45
        BWIDTH_RANGE      = 0.080
        BREAK_BUFFER      = 0.0006
        RETEST_TOL_ATR    = 0.50
        SMC_REQUIRE_FVG   = False
        FBB_ATR_MIN       = 0.0007
        FBB_ATR_MAX       = 0.030
        FALLBACK_ENABLE   = False
        ATR_STOP_MULT     = 1.0

    elif MODE == "conservative":
        MIN_VOLVALUE_USDT = 3_000_000
        BASE_MIN_SCORE    = 72
        FALLBACK_MIN_SCORE= 65
        TOP_N_PER_SCAN    = 2
        COOLDOWN_SEC      = 2400
        ADX_TREND_MIN     = 20
        ONEH_DISP_BODY_MIN= 0.60
        BWIDTH_RANGE      = 0.045
        BREAK_BUFFER      = 0.0012
        RETEST_TOL_ATR    = 0.20
        SMC_REQUIRE_FVG   = True
        FBB_ATR_MIN       = 0.0012
        FBB_ATR_MAX       = 0.020
        FALLBACK_ENABLE   = False
        ATR_STOP_MULT     = 1.5

    else:  # balanced (balanced+ taban)
        MIN_VOLVALUE_USDT = 2_000_000
        BASE_MIN_SCORE    = 58
        FALLBACK_MIN_SCORE= 62
        TOP_N_PER_SCAN    = 2
        COOLDOWN_SEC      = 1800
        ADX_TREND_MIN     = 16
        ONEH_DISP_BODY_MIN= 0.55
        BWIDTH_RANGE      = 0.055
        BREAK_BUFFER      = 0.0008
        RETEST_TOL_ATR    = 0.25
        SMC_REQUIRE_FVG   = True
        FBB_ATR_MIN       = 0.0010
        FBB_ATR_MAX       = 0.028
        FALLBACK_ENABLE   = False
        ATR_STOP_MULT     = 1.2

apply_mode(MODE)
dyn_MIN_SCORE = BASE_MIN_SCORE

# ================== RETEST & MOMENTUM ==================
def retest_ok_long(dc_break_level, df15, atrv):
    low  = float(df15["l"].iloc[-1]); close=float(df15["c"].iloc[-1]); open_=float(df15["o"].iloc[-1])
    tol  = RETEST_TOL_ATR*atrv
    touched = (low <= dc_break_level + tol)
    strong  = (close > open_) and ((close-open_)/max(1e-12, df15["h"].iloc[-1]-df15["l"].iloc[-1]) > 0.55)
    return touched and strong

def retest_ok_short(dc_break_level, df15, atrv):
    high = float(df15["h"].iloc[-1]); close=float(df15["c"].iloc[-1]); open_=float(df15["o"].iloc[-1])
    tol  = RETEST_TOL_ATR*atrv
    touched = (high >= dc_break_level - tol)
    strong  = (close < open_) and ((open_-close)/max(1e-12, df15["h"].iloc[-1]-df15["l"].iloc[-1]) > 0.55)
    return touched and strong

def momentum_ok(df15, side):
    c,o = df15["c"], df15["o"]
    e9, e21 = ema(c, 9), ema(c, 21)
    bs = body_strength(o,c,df15["h"],df15["l"]).iloc[-1]
    if side=="LONG":
        return (e9.iloc[-1] > e21.iloc[-1]) and (float(c.iloc[-1]) >= float(c.iloc[-2])) and (bs >= 0.60)
    else:
        return (e9.iloc[-1] < e21.iloc[-1]) and (float(c.iloc[-1]) <= float(c.iloc[-2])) and (bs >= 0.60)

# ================== TREND/RANGE CANDIDATE ==================
def evaluate_trend_range(symbol, df15, df1h):
    o,c,h,l,v = df15["o"],df15["c"],df15["h"],df15["l"],df15["v"]
    atr = atr_wilder(h,l,c,ATR_PERIOD)
    ma, bb_u, bb_l, bwidth, _ = bollinger(c, BB_PERIOD, BB_K)
    dc_hi, dc_lo = donchian(h,l,DONCHIAN_WIN)
    bs = body_strength(o,c,h,l)

    close = float(c.iloc[-1]); prev_close=float(c.iloc[-2])
    atrv  = float(atr.iloc[-1]); bw = float(bwidth.iloc[-1])
    dchi  = float(dc_hi.shift(1).iloc[-1]); dclo=float(dc_lo.shift(1).iloc[-1])

    bias, disp_ok, adx1h, trend_ok = htf_gate_and_bias(df1h)
    candidates = []

    # --- TREND BREAK + (RETEST or MOMENTUM) ---
    if trend_ok and disp_ok:
        if bias=="LONG":
            long_break = (prev_close > dchi*(1+BREAK_BUFFER)) and (close >= prev_close)
            if long_break and (retest_ok_long(dchi, df15, atrv) or momentum_ok(df15,"LONG")):
                sl, (tp1,tp2,tp3) = compute_sl_tp("LONG", close, atrv)
                rr1 = (tp1-close)/max(1e-9, close-sl)
                score = 40 + min(20, (adx1h-ADX_TREND_MIN)*1.2) + (bs.iloc[-1]*10)
                if rr1 < 1.0: score -= 4
                candidates.append({"symbol":symbol,"side":"LONG","entry":close,"tps":(tp1,tp2,tp3),
                                   "sl":sl,"score":score,"p":sigmoid((score-65)/7),
                                   "regime":"TREND","reason":f"Trend kırılımı + {'Retest' if retest_ok_long(dchi,df15,atrv) else 'Momentum'} | 1H ADX={adx1h:.1f}, BW={bw:.4f}"})
        elif bias=="SHORT":
            short_break= (prev_close < dclo*(1-BREAK_BUFFER)) and (close <= prev_close)
            if short_break and (retest_ok_short(dclo, df15, atrv) or momentum_ok(df15,"SHORT")):
                sl, (tp1,tp2,tp3) = compute_sl_tp("SHORT", close, atrv)
                rr1 = (close-tp1)/max(1e-9, sl-close)
                score = 40 + min(20, (adx1h-ADX_TREND_MIN)*1.2) + (bs.iloc[-1]*10)
                if rr1 < 1.0: score -= 4
                candidates.append({"symbol":symbol,"side":"SHORT","entry":close,"tps":(tp1,tp2,tp3),
                                   "sl":sl,"score":score,"p":sigmoid((score-65)/7),
                                   "regime":"TREND","reason":f"Trend kırılımı + {'Retest' if retest_ok_short(dclo, df15, atrv) else 'Momentum'} | 1H ADX={adx1h:.1f}, BW={bw:.4f}"})

    # --- RANGE MEAN-REVERT (SMART BOUNCE) ---
    if (not trend_ok) and (not math.isnan(bw)) and bw <= BWIDTH_RANGE:
        rsi14 = float(rsi(c,14).iloc[-1])
        ma_v, bbu_v, bbl_v = float(ma.iloc[-1]), float(bb_u.iloc[-1]), float(bb_l.iloc[-1])

        near_lower = close <= bbl_v*(1+0.0010)
        near_upper = close >= bbu_v*(1-0.0010)

        re_enter_long  = (float(c.iloc[-2]) < bbl_v) and (float(c.iloc[-1]) > bbl_v)
        re_enter_short = (float(c.iloc[-2]) > bbu_v) and (float(c.iloc[-1]) < bbu_v)

        bs_last = float(body_strength(o, c, h, l).iloc[-1])

        VOL_MULT_REQ = VOL_MULT_REQ_GLOBAL
        vol_ok = float(v.iloc[-1]) > float(v.rolling(20).mean().iloc[-1]) * VOL_MULT_REQ
        RSI_LONG_TH  = 36
        RSI_SHORT_TH = 64
        bs_last_req  = 0.62

        if near_lower and rsi14 < RSI_LONG_TH and re_enter_long and bs_last >= bs_last_req and vol_ok and bias != "SHORT":
            sl, (tp1,tp2,tp3) = compute_sl_tp("LONG", close, atrv)
            score = 30 + (max(0, 38-rsi14)) + (1 - bw/max(1e-12,BWIDTH_RANGE))*10
            candidates.append({
                "symbol":symbol,"side":"LONG","entry":close,"tps":(tp1,tp2,tp3),
                "sl":sl,"score":score,"p":sigmoid((score-65)/7),
                "regime":"RANGE",
                "reason":f"Bant içi bounce (false breakout→re-enter + güçlü mum + hacim) | RSI={rsi14:.1f}, BW={bw:.4f}"
            })

        if near_upper and rsi14 > RSI_SHORT_TH and re_enter_short and bs_last >= bs_last_req and vol_ok and bias != "LONG":
            sl, (tp1,tp2,tp3) = compute_sl_tp("SHORT", close, atrv)
            score = 30 + (max(0, rsi14-62)) + (1 - bw/max(1e-12,BWIDTH_RANGE))*10
            candidates.append({
                "symbol":symbol,"side":"SHORT","entry":close,"tps":(tp1,tp2,tp3),
                "sl":sl,"score":score,"p":sigmoid((score-65)/7),
                "regime":"RANGE",
                "reason":f"Bant içi bounce (false breakout→re-enter + güçlü mum + hacim) | RSI={rsi14:.1f}, BW={bw:.4f}"
            })

    if not candidates:
        return None

    # Ortak ATR% filtresi: aşırı oynak günleri ele
    atrv_last = float(atr_wilder(df15["h"], df15["l"], df15["c"], ATR_PERIOD).iloc[-1])
    close_last = float(df15["c"].iloc[-1])
    atr_pct = atrv_last / (close_last + 1e-12)
    if atr_pct > 0.035:
        return None

    return sorted(candidates, key=lambda x: x["score"], reverse=True)[0]

# ================== MOMENTUM-BREAKOUT (erken yakalama) ==================
def momentum_breakout_candidate(symbol, df15, df1h):
    if df15 is None or len(df15)<50 or df1h is None or len(df1h)<50:
        return None
    o,c,h,l,v = df15["o"],df15["c"],df15["h"],df15["l"],df15["v"]
    close = float(c.iloc[-1])

    # Donchian/EMA kapısı
    dc_hi, dc_lo = donchian(h,l, DONCHIAN_WIN)
    e21 = ema(c, 21)
    long_gate  = (float(c.iloc[-1]) > float(dc_hi.shift(1).iloc[-1]) * (1+BREAK_BUFFER)) and (float(c.iloc[-1]) > float(e21.iloc[-1]))
    short_gate = (float(c.iloc[-1]) < float(dc_lo.shift(1).iloc[-1]) * (1-BREAK_BUFFER)) and (float(c.iloc[-1]) < float(e21.iloc[-1]))

    # 1H bias
    e50_1h = ema(df1h["c"], 50)
    bias_1h = "NEUTRAL"
    if pd.notna(e50_1h.iloc[-1]) and pd.notna(e50_1h.iloc[-2]):
        bias_1h = "LONG" if e50_1h.iloc[-1]>e50_1h.iloc[-2] else ("SHORT" if e50_1h.iloc[-1]<e50_1h.iloc[-2] else "NEUTRAL")

    # ATR koruma + aşırı uzama freni
    atrv  = float(atr_wilder(h,l,c, ATR_PERIOD).iloc[-1]);
    atr_pct = atrv/(close+1e-12)
    if atr_pct > 0.05:
        return None
    if abs(close - float(e21.iloc[-1])) > 1.8*atrv:
        return None

    # Esnek momentum onayı (3 mum şartı yerine)
    if long_gate and bias_1h!="SHORT" and confirm_momentum(df15, "LONG"):
        sl, (tp1,tp2,tp3) = compute_sl_tp("LONG", close, atrv)
        return {"symbol":symbol,"side":"LONG","entry":close,"tps":(tp1,tp2,tp3),"sl":sl,
                "regime":"MO","reason":f"Momentum onay ({MOMO_CONFIRM_MODE}) + DC/EMA breakout"}

    if short_gate and bias_1h!="LONG" and confirm_momentum(df15, "SHORT"):
        sl, (tp1,tp2,tp3) = compute_sl_tp("SHORT", close, atrv)
        return {"symbol":symbol,"side":"SHORT","entry":close,"tps":(tp1,tp2,tp3),"sl":sl,
                "regime":"MO","reason":f"Momentum onay ({MOMO_CONFIRM_MODE}) + DC/EMA breakdown"}

    return None

def pick_best_candidate(symbol, df15, df1h):
    best = None
    # legacy (SMC + TREND/RANGE)
    smc = tr = None
    try: smc = build_smc_candidate(symbol, df15, df1h)
    except: pass
    try: tr  = evaluate_trend_range(symbol, df15, df1h)
    except: pass
    for c in [smc, tr]:
        if c:
            c = apply_scoring(symbol, df15, df1h, c, {"vol_pct": VOL_PCT_CACHE.get(symbol,0.5)})
            if (best is None) or (c["score"] > best["score"]): best = c
    # momentum-breakout (erken fırsata öncelik)
    mo = momentum_breakout_candidate(symbol, df15, df1h)
    if mo:
        mo = apply_scoring(symbol, df15, df1h, mo, {"vol_pct": VOL_PCT_CACHE.get(symbol,0.5)})
        mo["score"] = max(mo["score"], BASE_MIN_SCORE - 4)
        if (best is None) or (mo["score"] > best["score"]): best = mo
    return best
# ================== EXPLANATION & ANALIZ ==================
def human_reason_text(sig):
    r = sig.get("reason","")
    regime = sig.get("regime","-")
    if regime == "TREND":
        return "1H trend yönünde kırılım + " + ("retest" if "Retest" in r else "momentum teyidi")
    if regime == "RANGE":
        return "Bant içinde (false breakout→re-enter + güçlü mum + hacim) bounce"
    if regime == "SMC":
        return "Likidite süpürme → CHOCH, FVG/OTE bölgesinden dönüş"
    if regime == "MO":
        return "Momentum onay + DC/EMA breakout/breakdown"
    if regime == "FALLBACK":
        return "Bollinger bandı dışına taşma sonrası dönüş (FBB)"
    return r or "-"

def build_explanation(symbol, sig):
    df15 = get_ohlcv(symbol, TF_LTF, LOOKBACK_LTF)
    df1h = get_ohlcv(symbol, TF_HTF, LOOKBACK_HTF)
    if df15 is None or df1h is None: return "Veri alınamadı."
    o,c,h,l,v = df15["o"], df15["c"], df15["h"], df15["l"], df15["v"]
    close = float(c.iloc[-1])
    ma, bb_u, bb_l, bwidth_series, _ = bollinger(c, BB_PERIOD, BB_K)
    bw_last = float(bwidth_series.iloc[-1]) if pd.notna(bwidth_series.iloc[-1]) else float("nan")
    adxv  = float(adx(h,l,c,14).iloc[-1])
    rsi14 = float(rsi(c,14).iloc[-1])
    atrv  = float(atr_wilder(h,l,c,ATR_PERIOD).iloc[-1])
    atr_pct = atrv/(close+1e-12)
    e50 = ema(df1h["c"], 50)
    bias = "NEUTRAL"
    if pd.notna(e50.iloc[-1]) and pd.notna(e50.iloc[-2]):
        bias = "LONG" if e50.iloc[-1] > e50.iloc[-2] else ("SHORT" if e50.iloc[-1] < e50.iloc[-2] else "NEUTRAL")
    entry=float(sig["entry"]); tp1,tp2,tp3=sig["tps"]; sl=float(sig["sl"])
    rr = (tp1-entry)/max(1e-9, (entry-sl)) if sig["side"]=="LONG" else (entry-tp1)/max(1e-9, (sl-entry))

    pros, cons = [], []
    if sig.get("regime")=="SMC": pros.append("SMC (süpürme→CHOCH + FVG/OTE)")
    if adxv >= ADX_TREND_MIN: pros.append(f"1H trend gücü yeterli (ADX≈{adxv:.1f})")
    else: cons.append(f"1H ADX düşük (≈{adxv:.1f})")
    if not math.isnan(bw_last) and bw_last <= BWIDTH_RANGE: pros.append(f"Bant dar (BW≈{bw_last:.4f})")
    if atr_pct < 0.001: cons.append(f"ATR% çok düşük (≈{atr_pct:.4f})")
    if atr_pct > 0.03: cons.append(f"ATR% yüksek (≈{atr_pct:.4f})")
    if bias != sig.get("side","NEUTRAL"): cons.append(f"Üst zaman dilimi bias {bias} ≠ sinyal {sig['side']}")

    reason = human_reason_text(sig)
    text = (
        f"🧠 *{symbol}* — 15m\n"
        f"• Yön: *{sig['side']}* | Düzen: *{sig.get('regime','-')}*\n"
        f"• Neden: {reason}\n"
        f"• Üst ZD (1H) Bias: *{bias}* | ADX≈{adxv:.1f} | BW≈{bw_last:.4f}\n"
        f"• Fiyat: {close:.6f} | Entry: {entry:.6f} | SL: {sl:.6f}\n"
        f"• TP1/2/3: {tp1:.6f} / {tp2:.6f} / {tp3:.6f}\n"
        f"• Risk-Ödül (TP1’e, R): ≈{rr:.2f} | Mod: *{MODE}*\n\n"
        f"✅ Artılar: " + (", ".join(pros) if pros else "-") + "\n"
        f"⚠️ Riskler: " + (", ".join(cons) if cons else "-")
    )
    return text

def _analyze_symbol_text(symbol: str) -> str:
    df15 = get_ohlcv(symbol, TF_LTF, LOOKBACK_LTF)
    df1h = get_ohlcv(symbol, TF_HTF, LOOKBACK_HTF)
    if df15 is None or len(df15)<80 or df1h is None or len(df1h)<60:
        return "Veri alınamadı ya da yetersiz."

    o,c,h,l,v = df15["o"], df15["c"], df15["h"], df15["l"], df15["v"]
    close = float(c.iloc[-1])

    rsi14 = float(rsi(c,14).iloc[-1])
    adx15 = float(adx(h,l,c,14).iloc[-1])
    atrv  = float(atr_wilder(h,l,c,ATR_PERIOD).iloc[-1]); atr_pct = atrv/(close+1e-12)
    ma, bb_u, bb_l, bwidth, _ = bollinger(c, BB_PERIOD, BB_K)
    bw = float(bwidth.iloc[-1]) if pd.notna(bwidth.iloc[-1]) else float("nan")
    bbu = float(bb_u.iloc[-1]); bbl = float(bb_l.iloc[-1])
    dc_hi, dc_lo = donchian(h,l,DONCHIAN_WIN)
    dchi = float(dc_hi.iloc[-1]); dclo = float(dc_lo.iloc[-1])

    bias, disp_ok, adx1h, trend_ok = htf_gate_and_bias(df1h)

    sw_hi = float(swing_high(h, SWING_WIN))
    sw_lo = float(swing_low(l, SWING_WIN))

    regime = "TREND" if (trend_ok and disp_ok) else ("RANGE" if (not math.isnan(bw) and bw <= BWIDTH_RANGE) else "NEUTRAL")

    pos_bb = "Alt banda yakın" if close <= bbl*(1+0.001) else ("Üst banda yakın" if close >= bbu*(1-0.001) else "Band içi")
    pos_dc = "Üst kırılım yakın" if close >= dchi*(1-0.001) else ("Alt kırılım yakın" if close <= dclo*(1+0.001) else "Orta")

    # ATR+R örnek SL
    risk = ATR_STOP_MULT * atrv
    sl_long  = close - risk
    sl_short = close + risk
    rr_hint  = "ATR+R standardı: TP’ler (1.0R, 1.6R, 2.2R). R = ATR_STOP_MULT × ATR."

    txt = (
        f"📊 *{symbol}* — Teknik Analiz (15m + 1H)\n"
        f"• Fiyat: `{fmt(close)}` | ATR%≈`{atr_pct:.4f}` | BW≈`{bw:.4f}`\n"
        f"• RSI14(15m): `{rsi14:.1f}` | ADX(15m): `{adx15:.1f}`\n"
        f"• 1H Bias: *{bias}* | ADX(1H): `{adx1h:.1f}` | Trend_OK: *{str(trend_ok)}*\n"
        f"• BB Pozisyon: {pos_bb} | Donchian: {pos_dc}\n"
        f"• Donchian Üst/Alt: `{fmt(dchi)}` / `{fmt(dclo)}` | Swing H/L({SWING_WIN}): `{fmt(sw_hi)}` / `{fmt(sw_lo)}`\n"
        f"• Rejim Tahmini: *{regime}*\n\n"
        f"🎯 *Plan İpuçları*\n"
        f"- TREND gününde (ADX1H yüksek): kırılım + retest/momentum kovala.\n"
        f"- RANGE gününde: alt/üst banda sarkıp *içeri dönüş + güçlü mum + hacim* varsa bounce denenir.\n"
        f"- ATR+R Stop/TP: LONG SL `{fmt(sl_long)}` | SHORT SL `{fmt(sl_short)}`. {rr_hint}\n"
    )
    return txt

# ================== PERFORMANCE & ADAPT ==================
def evaluate_signal_outcome(sym, side, entry, sl, tp1, since_ts):
    df = get_ohlcv(sym, TF_LTF, LOOKBACK_LTF)
    if df is None: return None
    ts = pd.to_datetime(since_ts, unit="s", utc=True)
    idx = df.index[df["time"] == ts]
    if len(idx)==0:
        idx = df.index[df["time"] > ts]
        if len(idx)==0: return None
        start = idx[0]
    else:
        start = idx[0] + 1
    end = min(len(df)-1, start + EVAL_BARS_AHEAD)
    for i in range(start, end+1):
        hi = float(df["h"].iloc[i]); lo = float(df["l"].iloc[i])
        if side=="LONG":
            if lo <= sl and hi >= tp1: return "SL"
            if lo <= sl: return "SL"
            if hi >= tp1: return "TP"
        else:
            if hi >= sl and lo <= tp1: return "SL"
            if hi >= sl: return "SL"
            if lo <= tp1: return "TP"
    return None

def adapt_thresholds():
    global dyn_MIN_SCORE
    recent = [s for s in signals_history if s.get("resolved")]
    recent = recent[-ADAPT_WINDOW:]
    if len(recent) < ADAPT_MIN_SAMPLES: return
    wins = sum(1 for s in recent if s.get("result")=="TP")
    wr = wins/len(recent)
    if wr > ADAPT_UP_THRESH:
        dyn_MIN_SCORE = max(MIN_SCORE_FLOOR, dyn_MIN_SCORE - ADAPT_STEP)
    elif wr < ADAPT_DN_THRESH:
        dyn_MIN_SCORE = min(MIN_SCORE_CEIL, dyn_MIN_SCORE + ADAPT_STEP)

def schedule_signal_for_eval(sym, side, entry, sl, tp1, bar_ts, feats=None):
    signals_history.append({
        "symbol": sym, "side": side, "entry": float(entry), "sl": float(sl),
        "tp1": float(tp1), "ts_close": int(bar_ts), "resolved": False, "result": None,
        "_feats": feats or {}
    })

def resolve_open_signals():
    updated = 0
    for s in signals_history:
        if s["resolved"]: continue
        res = evaluate_signal_outcome(s["symbol"], s["side"], s["entry"], s["sl"], s["tp1"], s["ts_close"])
        if res in ("TP","SL"):
            s["resolved"] = True; s["result"] = res; updated += 1
            mark_symbol_outcome(s["symbol"], res)
            if AI_ENABLED and "_feats" in s:
                ai_update_online(s["_feats"], 1 if res=="TP" else 0)
    if updated:
        adapt_thresholds()

# ====== Auto-Tuner yardımcıları ======
def _clip(v, lo, hi): return max(lo, min(hi, v))

def _recent_wr(history, n):
    arr = [1 if s.get("resolved") and s.get("result")=="TP" else (0 if s.get("resolved") else None)
           for s in history[-n:]]
    arr = [x for x in arr if x is not None]
    if not arr: return None
    return sum(arr)/len(arr)

def _streak(history, what="SL"):
    k = 0
    for s in reversed(history):
        if not s.get("resolved"): break
        if s.get("result")==what: k += 1
        else: break
    return k

def auto_tune_now():
    global BASE_MIN_SCORE, ADX_TREND_MIN, BWIDTH_RANGE, VOL_MULT_REQ_GLOBAL, dyn_MIN_SCORE, _last_tune_ts
    if not AUTO_TUNER_ON or len(signals_history) < WIN_MIN_SAMPLES: return
    now = time.time()
    if now - _last_tune_ts < TUNE_COOLDOWN_SEC: return

    wr = _recent_wr(signals_history, TUNE_WINDOW)
    if wr is None: return
    sl_streak = _streak(signals_history, "SL")
    changed = False

    if sl_streak >= 3:
        BASE_MIN_SCORE = _clip(BASE_MIN_SCORE + 2, *BOUNDS["BASE_MIN_SCORE"]); changed = True
        ADX_TREND_MIN  = _clip(ADX_TREND_MIN  + 1, *BOUNDS["ADX_TREND_MIN"])
        VOL_MULT_REQ_GLOBAL = _clip(VOL_MULT_REQ_GLOBAL + 0.05, *BOUNDS["VOL_MULT_REQ"])
    else:
        delta = wr - WR_TARGET
        step = 2 if abs(delta) > 0.06 else 1
        if delta < -0.01:
            BASE_MIN_SCORE = _clip(BASE_MIN_SCORE - step, *BOUNDS["BASE_MIN_SCORE"]); changed = True
            ADX_TREND_MIN  = _clip(ADX_TREND_MIN  - 1, *BOUNDS["ADX_TREND_MIN"])
            BWIDTH_RANGE   = _clip(BWIDTH_RANGE   + 0.003, *BOUNDS["BWIDTH_RANGE"])
            VOL_MULT_REQ_GLOBAL = _clip(VOL_MULT_REQ_GLOBAL - 0.05, *BOUNDS["VOL_MULT_REQ"])
        elif delta > 0.04:
            BASE_MIN_SCORE = _clip(BASE_MIN_SCORE + 1, *BOUNDS["BASE_MIN_SCORE"]); changed = True
            VOL_MULT_REQ_GLOBAL = _clip(VOL_MULT_REQ_GLOBAL + 0.03, *BOUNDS["VOL_MULT_REQ"])

    if changed:
        dyn_MIN_SCORE = BASE_MIN_SCORE
        _last_tune_ts = now
        log(f"🛠️ AutoTune | WR={wr:.2f} | BASE_MIN_SCORE={BASE_MIN_SCORE} ADX_MIN={ADX_TREND_MIN} BW={BWIDTH_RANGE:.3f} VOLx={VOL_MULT_REQ_GLOBAL:.2f}")

# ================== TELEGRAM ==================
def make_detail_kb(sid, symbol, chat_id):
    is_following = symbol in FOLLOWED.get(chat_id, set())
    follow_btn = InlineKeyboardButton(text=("🚫 Takipten Çık" if is_following else "👁️ Takip Et"),
                                      callback_data=f"{'UNFOLLOW' if is_following else 'FOLLOW'}|{symbol}")
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📎 Detay", callback_data=f"DETAIL|{sid}")],
        [follow_btn]
    ])

async def send_signal(sig):
    global _sid_counter
    if _cached_chat_id is None:
        log("Chat yok → /start bekleniyor."); return
    sid = str(_sid_counter); _sid_counter += 1
    signals_store[sid] = {"sig": sig, "ts": time.time()}
    t1,t2,t3 = sig["tps"]

    rr1 = (t1-sig["entry"])/max(1e-9, sig["entry"]-sig["sl"]) if sig["side"]=="LONG" else (sig["entry"]-t1)/max(1e-9, sig["sl"]-sig["entry"])
    ex = sig.get("_explain", {})
    reason_text = human_reason_text(sig)
    htf = ex.get("b1h","-")

    text = (
        f"🔔 *{sig['symbol']}* — 15m\n"
        f"• Yön: *{sig['side']}*   • Düzen: *{sig['regime']}*   • Mod: *{MODE}*\n"
        f"• 1H Bias: *{htf}*   • Neden: {reason_text}\n\n"
        f"• Entry : `{fmt(sig['entry'])}`\n"
        f"• SL    : `{fmt(sig['sl'])}`  (Stop Loss)\n"
        f"• TP1   : `{fmt(t1)}`\n"
        f"• TP2   : `{fmt(t2)}`\n"
        f"• TP3   : `{fmt(t3)}`\n"
        f"• Risk-Ödül (TP1, *R*): ≈*{rr1:.2f}*\n\n"
        f"_Notlar:_\n"
        f"- *SL (Stop Loss)*: Zarar durdur.\n"
        f"- *TP (Take Profit)*: Kar al seviyeleri.\n"
        f"- *R (Risk)*: ATR_STOP_MULT × ATR. *1.0R* = bu mesafe kadar kar/zarar.\n"
    )
    try:
        await bot.send_message(chat_id=_cached_chat_id, text=text, parse_mode="Markdown", reply_markup=make_detail_kb(sid, sig["symbol"], _cached_chat_id))
    except (TelegramBadRequest, TelegramForbiddenError) as e:
        log("Telegram:", e)

@dp.callback_query(F.data.startswith("DETAIL|"))
async def detail_handler(q: CallbackQuery):
    try:
        _, sid = q.data.split("|", 1)
        data = signals_store.get(sid)
        if not data:
            await q.message.answer("Detay bulunamadı (sinyal süresi dolmuş olabilir)."); await q.answer(); return
        sig = data["sig"]
        text = build_explanation(sig["symbol"], sig)
        await q.message.answer(text, parse_mode="Markdown"); await q.answer()
    except Exception as e:
        await q.message.answer(f"Detay üretimi hata: {e}"); await q.answer()

@dp.callback_query(F.data.startswith("FOLLOW|"))
async def follow_cb(q: CallbackQuery):
    try:
        _, sym = q.data.split("|", 1)
        chat = q.message.chat.id
        FOLLOWED.setdefault(chat, set()).add(sym)
        await q.answer("Takibe alındı.")
        await q.message.edit_reply_markup(reply_markup=make_detail_kb("0", sym, chat))
        await q.message.answer(f"👁️ *{sym}* takibe alındı.\nGelişmeler otomatik paylaşılacak.", parse_mode="Markdown")
    except Exception as e:
        await q.answer(f"Hata: {e}", show_alert=True)

@dp.callback_query(F.data.startswith("UNFOLLOW|"))
async def unfollow_cb(q: CallbackQuery):
    try:
        _, sym = q.data.split("|", 1)
        chat = q.message.chat.id
        FOLLOWED.setdefault(chat, set()).discard(sym)
        await q.answer("Takipten çıkarıldı.")
        await q.message.edit_reply_markup(reply_markup=make_detail_kb("0", sym, chat))
        await q.message.answer(f"🚫 *{sym}* takipten çıkarıldı.", parse_mode="Markdown")
    except Exception as e:
        await q.answer(f"Hata: {e}", show_alert=True)

@dp.message(Command("follow"))
async def cmd_follow(m: Message):
    parts = m.text.strip().split()
    if len(parts) < 2:
        await m.answer("Kullanım: /follow WIFUSDT veya /follow WIF-USDT"); return
    norm = normalize_symbol_to_kucoin(parts[1])
    if not norm: await m.answer("Sembol bulunamadı."); return
    FOLLOWED.setdefault(m.chat.id, set()).add(norm)
    await m.answer(f"👁️ *{norm}* takibe alındı.", parse_mode="Markdown")

@dp.message(Command("unfollow"))
async def cmd_unfollow(m: Message):
    parts = m.text.strip().split()
    if len(parts) < 2:
        await m.answer("Kullanım: /unfollow WIFUSDT veya /unfollow WIF-USDT"); return
    norm = normalize_symbol_to_kucoin(parts[1])
    if not norm: await m.answer("Sembol bulunamadı."); return
    FOLLOWED.setdefault(m.chat.id, set()).discard(norm)
    await m.answer(f"🚫 *{norm}* takipten çıkarıldı.", parse_mode="Markdown")

@dp.message(Command("list"))
async def cmd_list(m: Message):
    lst = sorted(FOLLOWED.get(m.chat.id, set()))
    if not lst: await m.answer("Takip edilen sembol yok."); return
    await m.answer("👁️ Takip listesi:\n- " + "\n- ".join(lst))

@dp.message(Command("mode"))
async def mode_handler(m: Message):
    parts = m.text.strip().split()
    if len(parts) < 2:
        await m.answer("Kullanım: /mode aggressive | balanced | conservative")
        return
    target = parts[1].lower()
    if target not in ("aggressive","balanced","conservative"):
        await m.answer("Geçersiz mod. Seçenekler: aggressive | balanced | conservative")
        return
    apply_mode(target)
    global dyn_MIN_SCORE
    dyn_MIN_SCORE = BASE_MIN_SCORE
    await m.answer(
        f"⚙️ Mode: *{MODE}*\n"
        f"MinScore={BASE_MIN_SCORE}, ADXmin={ADX_TREND_MIN}, BWmax={BWIDTH_RANGE}, VolMin≈{MIN_VOLVALUE_USDT}, ATR_STOP_MULT={ATR_STOP_MULT}",
        parse_mode="Markdown"
    )

@dp.message(Command("aistats"))
async def ai_stats_cmd(m: Message):
    if not AI_ENABLED:
        await m.answer("AI kapalı."); return
    lines = [f"AI seen: #{_ai_seen} | bias={_ai_b:.3f}"]
    for k in sorted(_ai_w.keys()):
        lines.append(f"{k:14s}: {_ai_w[k]: .3f}")
    await m.answer("```\n" + "\n".join(lines) + "\n```", parse_mode="Markdown")

@dp.message(Command("aireset"))
async def ai_reset_cmd(m: Message):
    global _ai_w, _ai_b, _ai_seen
    _ai_w = {k: 0.0 for k in SCORING_WEIGHTS.keys()}
    _ai_b = AI_INIT_BIAS
    _ai_seen = 0
    await m.answer("AI ağırlıkları sıfırlandı.")

@dp.message(Command("analiz"))
async def analiz_cmd(m: Message):
    parts = m.text.strip().split()
    if len(parts) < 2:
        await m.answer("Kullanım: /analiz WIFUSDT veya /analiz WIF-USDT")
        return
    raw = parts[1].upper()
    norm = normalize_symbol_to_kucoin(raw)
    if not norm:
        await m.answer(f"❗ '{raw}' KuCoin'de bulunamadı. Örn: WIFUSDT → WIF-USDT")
        return
    await m.answer(f"⏳ Analiz ediliyor: {norm}")
    try:
        text = _analyze_symbol_text(norm)
        await m.answer(text, parse_mode="Markdown")
    except Exception as e:
        await m.answer(f"Analiz hatası ({norm}): {e}")

@dp.message(Command("start"))
async def start_handler(m: Message):
    global _cached_chat_id, dyn_MIN_SCORE
    _cached_chat_id = m.chat.id
    dyn_MIN_SCORE = BASE_MIN_SCORE
    await m.answer(
        "✅ Bot hazır.\n"
        f"Mode: *{MODE}* (ATR_STOP_MULT={ATR_STOP_MULT})\n"
        "• 5 dakikada bir tarar, sinyaller 15m grafiğe göre üretilir.\n"
        "• 📎 Detay ve 👁️ Takip butonlarını kullanabilirsin.\n"
        "• Komutlar: /mode | /analiz <Sembol> | /follow <Sembol> | /unfollow <Sembol> | /list | /aistats | /aireset",
        parse_mode="Markdown"
    )

# ================== PIPELINE ==================
def can_emit(symbol, side, df15):
    st = position_state.get(symbol)
    if st is None: return True
    if side == st['side']: return False
    bars_since = (len(df15)-1) - st['bar_idx']
    return bars_since >= OPPOSED_MIN_BARS if (OPPOSED_MIN_BARS:=OPPOSITE_MIN_BARS) else bars_since >= 0

async def scan_one_symbol(sym, sem):
    async with sem:
        if VERBOSE_SCAN: log(f"🔎 Taranıyor: {sym}")
        now = time.time()
        if sym in last_signal_ts and now - last_signal_ts[sym] < COOLDOWN_SEC:
            if SHOW_SKIP_REASONS: log(f"⏳ (cooldown) atlanıyor: {sym}")
            return None

        df15 = get_ohlcv(sym, TF_LTF, LOOKBACK_LTF)
        df1h = get_ohlcv(sym, TF_HTF, LOOKBACK_HTF)
        if df15 is None or len(df15)<80 or df1h is None or len(df1h)<60:
            if SHOW_SKIP_REASONS: log(f"— Veri yok/az: {sym}")
            return None

        last_bar_ts = int(df15["time"].iloc[-1].timestamp())
        if sym in position_state and position_state[sym].get("last_bar_ts")==last_bar_ts:
            if SHOW_SKIP_REASONS: log(f"— Aynı bar, atlanıyor: {sym}")
            return None

        best = None
        try:
            best = pick_best_candidate(sym, df15, df1h)
        except Exception as e:
            log(f"candidate hata ({sym}): {e}")

        if not best:
            if SHOW_SKIP_REASONS: log(f"— Aday yok: {sym}")
            return None

        if not can_emit(sym, best["side"], df15):
            if SHOW_SKIP_REASONS: log(f"— Aday blok (flip): {sym}")
            return None

        if VERBOSE_SCAN: log(f"✓ Aday: {sym} {best['side']} | Skor={int(best['score'])}")
        best["_bar_idx"]     = len(df15)-1
        best["_last_bar_ts"] = last_bar_ts
        return best

def _chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def build_vol_pct_cache(symbols, volmap):
    vals = [volmap.get(s, 0.0) for s in symbols]
    if not vals: return {}
    sorted_vals = sorted(vals)
    n = len(sorted_vals)
    cache = {}
    for s in symbols:
        v = volmap.get(s, 0.0)
        rank = sum(1 for x in sorted_vals if x <= v)
        cache[s] = rank / n
    return cache

async def run_scanner():
    syms = client.get_symbol_list()
    pairs = [s["symbol"] for s in syms if s.get("quoteCurrency")=="USDT"]
    tickers = client.get_all_tickers().get("ticker", [])
    volmap = {t.get("symbol"): float(t.get("volValue", 0.0)) for t in tickers}
    filt = [s for s in pairs if volmap.get(s,0.0) >= MIN_VOLVALUE_USDT]
    if not filt: filt = pairs
    random.shuffle(filt); filt = filt[:SCAN_LIMIT]

    global VOL_PCT_CACHE
    VOL_PCT_CACHE = build_vol_pct_cache(filt, volmap)

    log(f"Toplam {len(pairs)} USDT çifti | Likidite sonrası: {len(filt)} | Mode: {MODE}")
    if SHOW_SYMBOL_LIST_AT_START:
        for chunk in _chunked(filt, CHUNK_PRINT):
            log("Taranacak: " + "  ".join(chunk))

    sem = asyncio.Semaphore(SYMBOL_CONCURRENCY)

    while True:
        resolve_open_signals()
        auto_tune_now()

        t0 = time.time()
        tasks = [scan_one_symbol(sym, sem) for sym in filt]
        results = await asyncio.gather(*tasks)
        candidates = [r for r in results if r]
        candidates.sort(key=lambda x: x["score"], reverse=True)

        global dyn_MIN_SCORE, _empty_scans, _relax_acc
        strong_count = sum(1 for c in candidates if c["score"]>=dyn_MIN_SCORE)
        sent = 0
        for cand in candidates:
            if sent >= TOP_N_PER_SCAN: break
            if cand["score"] >= dyn_MIN_SCORE or (strong_count==0 and cand["score"]>=FALLBACK_MIN_SCORE):
                log(f"✅ {cand['symbol']} {cand['side']} | Entry={fmt(cand['entry'])} TP1={fmt(cand['tps'][0])} TP2={fmt(cand['tps'][1])} TP3={fmt(cand['tps'][2])} SL={fmt(cand['sl'])} | Skor={int(cand['score'])} | {cand['reason']}")
                await send_signal(cand)
                last_signal_ts[cand["symbol"]] = time.time()
                position_state[cand["symbol"]] = {'side': cand["side"], 'bar_idx': cand["_bar_idx"], 'last_bar_ts': cand["_last_bar_ts"]}
                schedule_signal_for_eval(cand["symbol"], cand["side"], cand["entry"], cand["sl"], cand["tps"][0], cand["_last_bar_ts"], feats=cand.get("_feats", {}))
                sent += 1

        dt_scan = time.time()-t0
        log(f"♻️ Tarama tamam ({dt_scan:.1f}s). Gönderilen: {sent}. DynMinScore={dyn_MIN_SCORE} | Mode={MODE}")

        # adaptif gevşetme (sinyal yoksa geçici yumuşatma)
        if strong_count == 0:
            _empty_scans += 1
            if _empty_scans >= EMPTY_LIMIT and _relax_acc < RELAX_MAX:
                dyn_MIN_SCORE = max(58, dyn_MIN_SCORE - RELAX_STEP)
                _relax_acc   += RELAX_STEP
                _empty_scans  = 0
        else:
            _empty_scans = 0
            dyn_MIN_SCORE = max(dyn_MIN_SCORE, BASE_MIN_SCORE)  # normale dön

        resolve_open_signals()
        await asyncio.sleep(SLEEP_SECONDS)

# ================== TAKİP ÖZELLİĞİ: Olay Tespiti ==================
def detect_events(df15, df1h, prev):
    events = []
    if df15 is None or len(df15)<30 or df1h is None or len(df1h)<30:
        return events, prev

    o,c,h,l,v = df15["o"], df15["c"], df15["h"], df15["l"], df15["v"]
    close = float(c.iloc[-1])
    ma, bb_u, bb_l, bwidth, _ = bollinger(c, BB_PERIOD, BB_K)
    dc_hi, dc_lo = donchian(h,l,DONCHIAN_WIN)
    rsi14 = float(rsi(c,14).iloc[-1])
    atrv  = float(atr_wilder(h,l,c,ATR_PERIOD).iloc[-1]); atr_pct = atrv/(close+1e-12)
    adx1h = float(adx(df1h["h"], df1h["l"], df1h["c"],14).iloc[-1])

    bbu = float(bb_u.iloc[-1]); bbl = float(bb_l.iloc[-1])
    dchi= float(dc_hi.iloc[-1]); dclo= float(dc_lo.iloc[-1])

    state = {
        "above_dc": close > dchi, "below_dc": close < dclo,
        "inside_bb": bbl < close < bbu,
        "rsi_zone": "high" if rsi14>=70 else ("low" if rsi14<=30 else "mid"),
        "vol_spike": float(v.iloc[-1]) > float(v.rolling(20).mean().iloc[-1]) * 1.5,
        "adx_trend": adx1h >= ADX_TREND_MIN,
        "atr_hot": atr_pct > 0.03,
        "near_lower": close <= bbl*(1+0.001), "near_upper": close >= bbu*(1-0.001),
        "sw_low": float(swing_low(l, SWING_WIN)), "sw_high": float(swing_high(h, SWING_WIN)),
        "close": close
    }

    prev = prev or {}

    if state["above_dc"] and not prev.get("above_dc"):
        events.append(f"📈 Donchian üstü *kırıldı* (~`{fmt(dchi)}` üstü).")
    if state["below_dc"] and not prev.get("below_dc"):
        events.append(f"📉 Donchian altı *kırıldı* (~`{fmt(dclo)}` altı).")

    was_out_lower = prev.get("inside_bb") is False and prev.get("near_lower")
    was_out_upper = prev.get("inside_bb") is False and prev.get("near_upper")
    if state["inside_bb"] and was_out_lower:
        events.append("🔁 BB altından *içeri dönüş* (re-enter).")
    if state["inside_bb"] and was_out_upper:
        events.append("🔁 BB üstünden *içeri dönüş* (re-enter).")

    if state["vol_spike"] and not prev.get("vol_spike"):
        events.append("🔥 Hacim *spike* (20MA ×1.5 üzeri).")

    if state["rsi_zone"]=="low" and prev.get("rsi_zone")!="low":
        events.append("🟢 RSI *düşük bölge* (≤30).")
    if state["rsi_zone"]=="high" and prev.get("rsi_zone")!="high":
        events.append("🔴 RSI *yüksek bölge* (≥70).")

    if state["adx_trend"] and not prev.get("adx_trend"):
        events.append(f"💪 1H ADX *trend* bölgesine geçti (≥{ADX_TREND_MIN}).")

    if state["near_lower"] and state["close"] > state["sw_low"]*(1+0.002) and not prev.get("near_lower"):
        events.append("🟩 Alt swing çevresinde *bounce* işareti.")
    if state["near_upper"] and state["close"] < state["sw_high"]*(1-0.002) and not prev.get("near_upper"):
        events.append("🟥 Üst swing çevresinde *red* işareti.")

    return events, state

async def follow_watcher():
    while True:
        try:
            if not FOLLOWED:
                await asyncio.sleep(60); continue
            for chat_id, syms in list(FOLLOWED.items()):
                for sym in list(syms):
                    df15 = get_ohlcv(sym, TF_LTF, LOOKBACK_LTF)
                    df1h = get_ohlcv(sym, TF_HTF, LOOKBACK_HTF)
                    events, state = detect_events(df15, df1h, FOLLOW_STATE.get((chat_id, sym)))
                    FOLLOW_STATE[(chat_id, sym)] = state
                    now = time.time()
                    last = FOLLOW_LAST_TS.get((chat_id, sym), 0)
                    if events and (now - last) > 180:
                        txt = f"👁️ *{sym}* — Takip güncellemesi\n" + "\n".join(f"• {e}" for e in events)
                        txt += "\n\n_Notlar:_ Donchian: kanal kırılımı. BB: Bollinger bandı. RSI/ADX: momentum/trend."
                        try:
                            await bot.send_message(chat_id=chat_id, text=txt, parse_mode="Markdown")
                            FOLLOW_LAST_TS[(chat_id, sym)] = now
                        except Exception as e:
                            log("Takip gönderim hata:", e)
            await asyncio.sleep(60)
        except Exception as e:
            log("follow_watcher hata:", e)
            await asyncio.sleep(60)

# ================== MAIN ==================
nest_asyncio.apply()
async def main():
    await asyncio.gather(
        dp.start_polling(bot),
        run_scanner(),
        follow_watcher()
    )

if __name__ == "__main__":
    asyncio.run(main())