"""Basit online lojistik öğrenici (özellik -> kazanma olasılığı).

Sinyal kapandığında (TP1+ = 1, STOP = 0 diğerleri yarım kredi olabilir)
özellik vektörüyle ağırlıkları günceller.
"""
from __future__ import annotations
import math
from typing import Dict, Iterable, Tuple


class OnlineLearner:
    def __init__(self, lr: float = 0.05, l2: float = 0.0005):
        self.lr = lr
        self.l2 = l2
        self.weights: Dict[str, float] = {}  # w0 bias dahil değil
        self.bias: float = 0.0
        self.seen = 0

    def _sigmoid(self, z: float) -> float:
        if z > 35:
            return 1.0
        if z < -35:
            return 0.0
        return 1 / (1 + math.exp(-z))

    def predict_proba(self, feats: Dict[str, float]) -> float:
        z = self.bias
        for k, v in feats.items():
            z += self.weights.get(k, 0.0) * v
        return self._sigmoid(z)

    def update(self, feats: Dict[str, float], label: float):
        # label 0..1
        p = self.predict_proba(feats)
        err = label - p
        self.bias += self.lr * err
        for k, v in feats.items():
            if k not in self.weights:
                self.weights[k] = 0.0
            # L2 + gradient
            old_w = self.weights[k]
            self.weights[k] = (self.weights[k] * (1 - self.lr * self.l2)) + self.lr * err * v
        self.seen += 1
        
        # Debug: Öğrenme bilgilerini yazdır
        if self.seen % 5 == 0:  # Her 5 örnekte bir
            print(f"[LEARNER DEBUG] Seen={self.seen}, bias={self.bias:.4f}, active_weights={len(self.weights)}")
            
        return self.predict_proba(feats)  # Güncellenmiş tahmini döndür

    def serialize(self) -> Dict[str, float]:
        d = {"bias": round(self.bias, 6), "seen": self.seen}
        for k, v in self.weights.items():
            d[f"w_{k}"] = round(v, 6)
        return d


_GLOBAL_LEARNER: OnlineLearner | None = None


def set_global_learner(l: OnlineLearner):
    global _GLOBAL_LEARNER
    _GLOBAL_LEARNER = l


def get_global_learner() -> OnlineLearner | None:
    return _GLOBAL_LEARNER
