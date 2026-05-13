from edge.src.budget_state import BudgetState


def test_force_close_event_is_consumed_once() -> None:
    state = BudgetState(default_windows=10)

    state.request_force_close(reason="append_response", decision_id=7, stream_id="s0")

    event = state.consume_force_close()
    assert event is not None
    assert event["reason"] == "append_response"
    assert event["decision_id"] == 7
    assert event["stream_id"] == "s0"
    assert state.consume_force_close() is None


def test_budget_update_can_ride_with_force_close() -> None:
    state = BudgetState(default_windows=10)

    changed = state.set(version=1, windows_per_decision=6, reason="ef_guard")
    state.request_force_close(reason="ef_guard", decision_id=3)

    assert changed is True
    assert state.windows_per_decision == 6
    assert state.changed_since_last_check is True
    assert state.consume_force_close()["decision_id"] == 3
