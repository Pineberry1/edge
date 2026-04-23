"""PacketGame-style scorer operating purely on compressed-domain features.

Score = f(is_IDR, normalized size z-score, size delta from previous packet).
IDRs get a boost; large unexpected P/B packets (high z-score, large delta)
indicate motion or scene change and are ranked higher.

For v2 a lightweight MLP on feature windows can be swapped in; for v1 we use
a transparent heuristic that is easy to analyze and cheap on CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .features import PacketFeatures


@dataclass
class ScorerWeights:
    i_boost: float = 1.5
    size_weight: float = 1.0
    novelty_weight: float = 0.8
    window_alpha: float = 0.2


class PacketScorer:
    def __init__(self, weights: ScorerWeights) -> None:
        self.w = weights
        self._ema = 0.0

    def score(self, feats: PacketFeatures) -> float:
        raw = 0.0
        if feats.is_idr:
            raw += self.w.i_boost
        raw += self.w.size_weight * math.tanh(max(0.0, feats.size_zscore))
        raw += self.w.novelty_weight * min(1.0, feats.size_delta_norm)

        if feats.slice_type == "B":
            raw *= 0.85

        self._ema = (1 - self.w.window_alpha) * self._ema + self.w.window_alpha * raw
        return raw - 0.2 * self._ema
