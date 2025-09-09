import os
from functools import lru_cache


class Settings:
    """Runtime ayarları (ENV üzerinden override edilebilir)."""

    EXCHANGE_BASE_URL: str = os.getenv("EXCHANGE_BASE_URL", "https://api.kucoin.com")
    QUOTE_FILTER: str = os.getenv("QUOTE_FILTER", "USDT")
    MIN_VOLVALUE_USDT: float = float(os.getenv("MIN_VOLVALUE_USDT", "2000000"))
    SCAN_LIMIT: int = int(os.getenv("SCAN_LIMIT", "220"))
    SYMBOL_CONCURRENCY: int = int(os.getenv("SYMBOL_CONCURRENCY", "8"))
    SLEEP_SECONDS: int = int(os.getenv("SLEEP_SECONDS", "300"))
    LOOKBACK_LTF: int = int(os.getenv("LOOKBACK_LTF", "320"))
    LOOKBACK_HTF: int = int(os.getenv("LOOKBACK_HTF", "180"))
    TF_LTF: str = os.getenv("TF_LTF", "15min")
    TF_HTF: str = os.getenv("TF_HTF", "1hour")
    MODE: str = os.getenv("MODE", "balanced")
    LOG_PREFIX: str = os.getenv("LOG_PREFIX", "📟")
    WEBSOCKET_PUSH_BUFFER: int = int(os.getenv("WEBSOCKET_PUSH_BUFFER", "100"))
    ATR_PERIOD: int = int(os.getenv("ATR_PERIOD", "14"))
    MIN_SCORE: int = int(os.getenv("MIN_SCORE", "45"))
    # Güvenlik
    HIDE_SENSITIVE: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
