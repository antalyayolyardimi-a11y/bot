"""Trading bot engine package.

Bu paket sinyal üretim motoru, indikatörler ve strateji seçim mantığı
için modüler bir yapı sağlar. Telegram entegrasyonu kaldırıldı.
"""

from .config import Settings, get_settings
from .scanner import SignalEngine

__all__ = ["Settings", "get_settings", "SignalEngine"]
