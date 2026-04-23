"""GOP-aligned buffering with decoder-safe selection.

Rules:
  * A GOP starts on IDR and ends just before the next IDR.
  * Selection is decoder-safe: we only ever keep an IDR-aligned prefix of a
    GOP. Dropping the tail of a closed GOP does not break any kept frame's
    reference chain.
  * Three decision modes: KEEP_ALL, KEEP_PREFIX(k), IDR_ONLY. IDR is always
    kept so temporal continuity is preserved on the cloud.

ρ is read from `RhoState` at every finalize, so cloud-driven updates apply
at the next GOP boundary; per-GOP aggregated score can push the decision
above or below the current ρ.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .config import EdgeConfig
from .features import PacketFeatures
from .rho_state import RhoState
from .rtsp_source import PacketRecord

MODE_ALL = "all"
MODE_PREFIX = "prefix"
MODE_IDR_ONLY = "idr_only"


@dataclass
class GopDecision:
    mode: str
    keep_non_idr: int
    gop_score: float
    rho_used: float


@dataclass
class GopBatch:
    gop_index: int
    packets: List[PacketRecord]
    features: List[PacketFeatures]
    scores: List[float]
    decision: GopDecision
    selected_local_indices: List[int]


class GopBuffer:
    """Stateful buffer that emits one GopBatch per closed GOP.

    ρ is read fresh from `rho_state` at every `_finalize`, so cloud-side
    updates take effect on the next GOP boundary without restarting the
    pipeline.
    """

    def __init__(
        self,
        cfg: EdgeConfig,
        scorer_fn: Callable[[PacketFeatures], float],
        rho_state: RhoState,
    ) -> None:
        self.cfg = cfg
        self.scorer_fn = scorer_fn
        self.rho_state = rho_state
        self._cur_packets: List[PacketRecord] = []
        self._cur_features: List[PacketFeatures] = []
        self._cur_scores: List[float] = []
        self._cur_gop_index: int = -1

    def push(self, rec: PacketRecord, feats: PacketFeatures) -> Optional[GopBatch]:
        score = self.scorer_fn(feats)
        flush_batch: Optional[GopBatch] = None
        if feats.is_idr and self._cur_packets:
            flush_batch = self._finalize()
            self._reset()
        if feats.is_idr:
            self._cur_gop_index = feats.gop_index
        self._cur_packets.append(rec)
        self._cur_features.append(feats)
        self._cur_scores.append(score)
        return flush_batch

    def flush(self) -> Optional[GopBatch]:
        if not self._cur_packets:
            return None
        batch = self._finalize()
        self._reset()
        return batch

    def _reset(self) -> None:
        self._cur_packets = []
        self._cur_features = []
        self._cur_scores = []

    def _finalize(self) -> GopBatch:
        n = len(self._cur_packets)
        non_idr_indices = [i for i, f in enumerate(self._cur_features) if not f.is_idr]
        n_non_idr = len(non_idr_indices)

        gop_score = self._aggregate_score(self._cur_scores)
        rho = self.rho_state.current
        target_keep = int(math.ceil(rho * n_non_idr))
        target_keep = max(self.cfg.min_keep_per_gop, min(n_non_idr, target_keep))

        activity_boost = math.tanh(max(0.0, gop_score))
        keep_non_idr = min(n_non_idr, int(round(target_keep * (0.5 + activity_boost))))
        keep_non_idr = max(0, keep_non_idr)

        if keep_non_idr >= n_non_idr and n_non_idr > 0:
            mode = MODE_ALL
        elif keep_non_idr == 0:
            mode = MODE_IDR_ONLY
        else:
            mode = MODE_PREFIX

        selected: List[int] = []
        kept_non_idr = 0
        for i, f in enumerate(self._cur_features):
            if f.is_idr:
                selected.append(i)
                continue
            if kept_non_idr < keep_non_idr:
                selected.append(i)
                kept_non_idr += 1

        return GopBatch(
            gop_index=max(0, self._cur_gop_index),
            packets=list(self._cur_packets),
            features=list(self._cur_features),
            scores=list(self._cur_scores),
            decision=GopDecision(
                mode=mode,
                keep_non_idr=keep_non_idr,
                gop_score=gop_score,
                rho_used=rho,
            ),
            selected_local_indices=selected,
        )

    @staticmethod
    def _aggregate_score(scores: List[float]) -> float:
        if not scores:
            return 0.0
        arr = sorted(scores, reverse=True)
        k = max(1, len(arr) // 3)
        return sum(arr[:k]) / k
