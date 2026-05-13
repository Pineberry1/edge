"""α executor — space-domain token budget enforcement on the cloud side.

What α means in this module
---------------------------
α ∈ (0, 1] is the cloud's token-keep rate per frame. Qwen3-VL's visual
encoder produces visual tokens roughly proportional to image area:

    visual_tokens(img) ≈ (H × W) / (patch_size × pixel_shuffle)²

so resizing each axis by √α scales the number of visual tokens by α. We
stay out of vLLM internals by doing the scaling in pixel space *before*
JPEG-encoding the frame for `/v1/online_prefill/sessions/.../append`.

Implementation notes
--------------------
* The resize is aligned to the ViT patch stride (28 for Qwen3-VL: patch 14
  × pixel-shuffle 2) so no tokens are wasted on fractional patches.
* We clamp to a sane minimum side so α → 0 doesn't collapse the frame to
  a 1-pixel smudge; Qwen3-VL's processor refuses inputs smaller than a
  few patches.
* `apply_alpha_weighted` honours an optional per-frame importance score
  (typically the edge anchor's packet score): high-score frames keep a
  larger share of the window's token budget; low-score frames shrink
  further. Budget invariant: Σ(α_i · tokens_i) ≈ α · Σ(tokens_i).
* If α ≥ 0.999 we skip the resize — it's a no-op that would still cost a
  memcpy.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

log = logging.getLogger("intake.alpha")


@dataclass
class AlphaPolicy:
    alpha: float                  # cloud-side token keep rate
    min_side: int = 112           # pixel floor per axis (~4 patches on Qwen3-VL)
    max_side: int = 1568          # pixel ceiling (Qwen3-VL typical max ≈ 56 patches)
    align: int = 28               # ViT patch stride × pixel-shuffle
    skip_threshold: float = 0.999 # α ≥ this is treated as identity
    # per-frame weighting exponent: 0 = uniform, >0 emphasises high-score frames
    score_power: float = 1.0
    # minimum per-frame α (prevents a low-score frame from dropping to 0)
    per_frame_floor: float = 0.15


def _snap(n: int, align: int, lo: int, hi: int) -> int:
    """Round n down to a multiple of `align`, clamped to [lo, hi]."""
    if align <= 1:
        return max(lo, min(hi, n))
    snapped = (n // align) * align
    if snapped < lo:
        snapped = ((lo + align - 1) // align) * align
    if snapped > hi:
        snapped = (hi // align) * align
    return snapped


def _resize_by_alpha(bgr: np.ndarray, alpha: float, policy: AlphaPolicy) -> np.ndarray:
    if alpha >= policy.skip_threshold:
        return bgr
    h, w = bgr.shape[:2]
    scale = math.sqrt(max(alpha, 1e-4))
    new_w = _snap(int(round(w * scale)), policy.align, policy.min_side, policy.max_side)
    new_h = _snap(int(round(h * scale)), policy.align, policy.min_side, policy.max_side)
    if new_w == w and new_h == h:
        return bgr
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(bgr, (new_w, new_h), interpolation=interp)


def apply_alpha_uniform(
    frames_bgr: Sequence[np.ndarray], policy: AlphaPolicy
) -> List[np.ndarray]:
    """Apply the same α to every frame — simplest policy."""
    if not frames_bgr:
        return []
    return [_resize_by_alpha(f, policy.alpha, policy) for f in frames_bgr]


def apply_alpha_weighted(
    frames_bgr: Sequence[np.ndarray],
    scores: Sequence[float],
    policy: AlphaPolicy,
) -> List[np.ndarray]:
    """Per-frame α weighted by importance scores, preserving Σtokens ≈ α·N·K.

    Mechanics:
      1. Normalise scores to sum to N (so uniform scores → α_i = α for all i).
      2. α_i = clip(α · normalised_score_i, per_frame_floor, 1.0).
      3. Renormalise to hit the global α budget (within [floor, 1] constraints).
    """
    n = len(frames_bgr)
    if n == 0:
        return []
    # Fallback: no scores / all zeros → uniform
    if len(scores) != n or sum(max(0.0, s) for s in scores) <= 1e-9:
        return apply_alpha_uniform(frames_bgr, policy)

    raw = np.asarray([max(0.0, float(s)) for s in scores], dtype=np.float64)
    if policy.score_power != 1.0:
        raw = np.power(raw, policy.score_power)
    # Shift so the minimum ≥ floor-scale proportion; normalise sum to n.
    raw = raw + 1e-9
    weights = raw * (n / raw.sum())        # mean(weights) = 1
    target = np.clip(
        policy.alpha * weights,
        policy.per_frame_floor,
        1.0,
    )
    # Re-hit the budget α·n after clipping (best-effort; clipped ceil/floor may miss by a bit).
    budget = policy.alpha * n
    gap = budget - float(target.sum())
    if abs(gap) > 1e-3:
        headroom = np.where(
            (target < 1.0) if gap > 0 else (target > policy.per_frame_floor),
            1.0,
            0.0,
        )
        if headroom.sum() > 0:
            target = target + gap * (headroom / headroom.sum())
            target = np.clip(target, policy.per_frame_floor, 1.0)

    out: List[np.ndarray] = []
    for frame, a_i in zip(frames_bgr, target.tolist()):
        per_frame_policy = AlphaPolicy(
            alpha=float(a_i),
            min_side=policy.min_side,
            max_side=policy.max_side,
            align=policy.align,
            skip_threshold=policy.skip_threshold,
            score_power=policy.score_power,
            per_frame_floor=policy.per_frame_floor,
        )
        out.append(_resize_by_alpha(frame, per_frame_policy.alpha, per_frame_policy))
    return out
