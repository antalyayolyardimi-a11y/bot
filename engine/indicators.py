import math
import pandas as pd
import numpy as np


def to_df_klines(raw):
    """KuCoin formatını DataFrame'e çevirir.

    KuCoin: [time, open, close, high, low, volume, turnover]
    """
    if not raw:
        return None
    df = pd.DataFrame(raw, columns=["time", "o", "c", "h", "l", "v", "turnover"])
    for col in ["time", "o", "c", "h", "l", "v", "turnover"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(inplace=True)
    df["time"] = pd.to_datetime(df["time"].astype(np.int64), unit="ms", utc=True)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def ema(s, n):  # Exponential moving average
    return pd.Series(s).ewm(span=n, adjust=False).mean()


def body_strength(o, c, h, l):
    body = (c - o).abs()
    rng = (h - l).abs().replace(0, float("nan"))
    val = body / rng
    return val.fillna(0.0)


def atr_wilder(h, l, c, n=14):
    pc = c.shift(1)
    tr1 = (h - l).abs()
    tr2 = (h - pc).abs()
    tr3 = (l - pc).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(h, l, c, n=14):
    up = h.diff(); dn = -l.diff()
    plus_dm = (up > dn) & (up > 0)
    minus_dm = (dn > up) & (dn > 0)
    p_dm = up.where(plus_dm, 0.0)
    m_dm = dn.where(minus_dm, 0.0)
    atr = atr_wilder(h, l, c, n)
    pdi = 100 * p_dm.ewm(alpha=1 / n, adjust=False).mean() / (atr + 1e-12)
    ndi = 100 * m_dm.ewm(alpha=1 / n, adjust=False).mean() / (atr + 1e-12)
    dx = (pdi - ndi).abs() / ((pdi + ndi) + 1e-12) * 100
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def bollinger(close, n=20, k=2.0):
    ma = close.rolling(n).mean(); std = close.rolling(n).std(ddof=0)
    upper = ma + k * std; lower = ma - k * std
    bwidth = (upper - lower) / (ma + 1e-12)
    return ma, upper, lower, bwidth, std


def donchian(h, l, win=20):
    return h.rolling(win).max(), l.rolling(win).min()


def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100 - (100 / (1 + rs))
