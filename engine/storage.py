"""Basit JSON persist katmanı.

Uygulama kapanıp açıldığında learner ağırlıkları ve kapanmış sinyal
performans kayıtları saklanıp yeniden yüklenir.
"""
from __future__ import annotations
import json
import os
from typing import Any, List, Dict
from pathlib import Path

DATA_DIR = Path("data")
STATE_FILE = DATA_DIR / "state.json"


def ensure_dir():
    DATA_DIR.mkdir(exist_ok=True)


def save_state(*, learner: Any = None, performance: Any = None):
    ensure_dir()
    payload: Dict[str, Any] = {}
    if learner:
        try:
            payload["learner"] = learner.serialize()
        except Exception:
            pass
    if performance:
        try:
            # yalnızca kapanan kayıtlar (hafıza şişmesin)
            closed = [r for r in performance.records if r.status != "OPEN"]
            payload["performance"] = {
                "records": [r.__dict__ for r in closed][-400:]  # son 400
            }
        except Exception:
            pass
    tmp = STATE_FILE.with_suffix(".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        tmp.replace(STATE_FILE)
    except Exception as e:
        print("[STORAGE] save error", e)


def load_state():
    if not STATE_FILE.exists():
        return {}
    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("[STORAGE] load error", e)
        return {}
