from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "remote_intake_budget_allocator",
    ROOT / "budget_allocator.py",
)
budget_allocator = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = budget_allocator
SPEC.loader.exec_module(budget_allocator)


def _cap():
    return budget_allocator.GlobalCapacity(
        kv_cap_tokens=98304,
        safety_factor=0.85,
        min_windows=2,
        max_windows=32,
        block_overhead_tokens=64,
        policy="fair_share",
    )


def _meta(stream_id: str, h: int = 320, w: int = 240):
    return budget_allocator.StreamCostMeta(
        stream_id=stream_id,
        frame_height=h,
        frame_width=w,
        frames_per_window=8,
        rho=1.0,
        prompt_tokens=64,
        max_tokens=24,
    )


def test_cost_model_and_equal_share_scaling() -> None:
    cap = _cap()
    assert budget_allocator.tokens_per_frame(320, 240) == 12 * 9 == 108
    assert budget_allocator.per_window_tokens(_meta("s0")) == 864

    for n in (1, 8, 16, 32):
        allocator = budget_allocator.BudgetAllocator(cap)
        metas = [_meta(f"s{i}") for i in range(n)]
        budgets = allocator.allocate(metas)
        got = budgets["s0"].windows_per_decision
        expected = math.floor(((98304 * 0.85) / n - 64 - 24 - 64) / 864)
        expected = min(cap.max_windows, max(cap.min_windows, expected))
        assert got == expected

    assert budget_allocator.BudgetAllocator(cap).allocate([_meta("s0")])["s0"].windows_per_decision == 32

    n8_w = budget_allocator.BudgetAllocator(cap).allocate([_meta(f"s{i}") for i in range(8)])[
        "s0"
    ].windows_per_decision
    assert 8 <= n8_w <= 14

    n16_w = budget_allocator.BudgetAllocator(cap).allocate([_meta(f"s{i}") for i in range(16)])[
        "s0"
    ].windows_per_decision
    assert 3 <= n16_w <= 7

    n32_w = budget_allocator.BudgetAllocator(cap).allocate([_meta(f"s{i}") for i in range(32)])[
        "s0"
    ].windows_per_decision
    assert n32_w == 2


def test_default_ef_guard_does_not_shorten_without_ef() -> None:
    cap = budget_allocator.GlobalCapacity(
        kv_cap_tokens=98304,
        safety_factor=0.85,
        min_windows=2,
        max_windows=32,
        block_overhead_tokens=64,
    )
    allocator = budget_allocator.BudgetAllocator(cap)
    metas = [
        budget_allocator.StreamCostMeta(
            stream_id=f"s{i}",
            frame_height=448,
            frame_width=448,
            frames_per_window=10,
            rho=0.5,
            prompt_tokens=64,
            max_tokens=12,
            alpha=0.5,
        )
        for i in range(100)
    ]
    budgets = allocator.allocate(metas)
    assert {b.windows_per_decision for b in budgets.values()} == {10}


def test_default_ef_guard_shortens_only_after_ef() -> None:
    cap = budget_allocator.GlobalCapacity(
        kv_cap_tokens=98304,
        safety_factor=0.85,
        min_windows=2,
        max_windows=32,
        block_overhead_tokens=64,
        ef_windows=6,
    )
    metas = [
        budget_allocator.StreamCostMeta(
            stream_id="normal",
            frame_height=448,
            frame_width=448,
            frames_per_window=10,
            rho=0.5,
            prompt_tokens=64,
            max_tokens=12,
            alpha=0.5,
        ),
        budget_allocator.StreamCostMeta(
            stream_id="ef",
            frame_height=448,
            frame_width=448,
            frames_per_window=10,
            rho=0.5,
            prompt_tokens=64,
            max_tokens=12,
            alpha=0.5,
            early_finalized_recent=True,
        ),
    ]
    budgets = budget_allocator.BudgetAllocator(cap).allocate(metas)
    assert budgets["normal"].windows_per_decision == 10
    assert budgets["ef"].windows_per_decision == 6


def test_ef_guard_uses_observed_trigger_window_with_margin() -> None:
    cap = budget_allocator.GlobalCapacity(
        kv_cap_tokens=98304,
        safety_factor=0.85,
        min_windows=2,
        max_windows=32,
        block_overhead_tokens=64,
        target_windows=10,
        ef_windows=6,
        ef_dynamic=True,
        ef_margin_windows=1,
    )
    metas = [
        replace(
            _meta("early"),
            early_finalized_recent=True,
            last_early_finalized_windows=2,
        ),
        replace(
            _meta("mid"),
            early_finalized_recent=True,
            last_early_finalized_windows=6,
        ),
        replace(
            _meta("late"),
            early_finalized_recent=True,
            last_early_finalized_windows=9,
        ),
    ]

    budgets = budget_allocator.BudgetAllocator(cap).allocate(metas)

    assert budgets["early"].windows_per_decision == 6
    assert budgets["mid"].windows_per_decision == 7
    assert budgets["late"].windows_per_decision == 10


def test_ef_guard_can_use_fixed_floor_for_ablations() -> None:
    cap = budget_allocator.GlobalCapacity(
        kv_cap_tokens=98304,
        safety_factor=0.85,
        min_windows=2,
        max_windows=32,
        block_overhead_tokens=64,
        target_windows=10,
        ef_windows=6,
        ef_dynamic=False,
        ef_margin_windows=1,
    )

    budget = budget_allocator.BudgetAllocator(cap).allocate([
        replace(
            _meta("s0"),
            early_finalized_recent=True,
            last_early_finalized_windows=9,
        )
    ])["s0"]

    assert budget.windows_per_decision == 6


def test_heterogeneous_resolution_gets_smaller_budget() -> None:
    allocator = budget_allocator.BudgetAllocator(_cap())
    metas = [_meta("hi", h=1280, w=720)] + [_meta(f"lo{i}") for i in range(7)]
    budgets = allocator.allocate(metas)

    assert budgets["hi"].windows_per_decision < budgets["lo0"].windows_per_decision


def test_alpha_and_live_kv_affect_budget() -> None:
    allocator = budget_allocator.BudgetAllocator(_cap())
    base = _meta("s0", h=1280, w=720)
    alpha_half = replace(base, alpha=0.5)
    live_heavy = replace(base, live_kv_tokens=20000)

    base_budget = allocator.allocate([base])["s0"]
    alpha_budget = allocator.allocate([alpha_half])["s0"]
    live_budget = allocator.allocate([live_heavy])["s0"]

    assert budget_allocator.per_window_tokens(alpha_half) < budget_allocator.per_window_tokens(base)
    assert alpha_budget.windows_per_decision > base_budget.windows_per_decision
    assert live_budget.windows_per_decision < base_budget.windows_per_decision
    assert live_budget.est_kv_tokens >= 20000


def test_deterministic_windows_and_monotonic_versions() -> None:
    allocator = budget_allocator.BudgetAllocator(_cap())
    metas = [_meta(f"s{i}") for i in range(3)]

    first = allocator.allocate(metas)
    second = allocator.allocate(metas)

    assert {sid: b.windows_per_decision for sid, b in first.items()} == {
        sid: b.windows_per_decision for sid, b in second.items()
    }
    assert max(b.version for b in first.values()) < min(b.version for b in second.values())
