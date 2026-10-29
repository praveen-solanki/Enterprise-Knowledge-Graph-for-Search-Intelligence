"""Unit tests for services/adaptive_controller.py's LatencyAdaptiveThrottle --
specifically the per-call target_latency_seconds override on record_success,
which lets one shared concurrency pool correctly judge calls of very
different natural sizes (e.g. a small confirmation call and a large
extraction call both routed through Ministral) instead of one fixed,
one-size-fits-all baseline set at construction."""
import asyncio

from services.adaptive_controller import LatencyAdaptiveThrottle


def _run(coro):
    return asyncio.run(coro)


def test_record_success_without_override_uses_the_instance_default_target():
    throttle = LatencyAdaptiveThrottle(initial=4, target_latency_seconds=15.0, min_concurrency=1)
    # 20s is > 15s (the instance default) but not > 15*3=45s -- neither fast
    # enough to grow nor slow enough to back off; concurrency holds steady.
    _run(throttle.record_success(20.0))
    assert throttle.current_concurrency == 4


def test_record_success_per_call_target_backs_off_a_call_thats_slow_for_its_own_kind():
    """A confirmation call that took 8s looks fine against extraction's 15s
    target, but is genuinely struggling against its own 2s target -- the
    per-call override must let the SAME shared instance catch this."""
    throttle = LatencyAdaptiveThrottle(initial=4, target_latency_seconds=15.0, min_concurrency=1)
    # 8s <= 15*3=45 under the instance default (no backoff at all), but
    # 8s > 2*3=6 under this call's own, smaller target -- must back off.
    _run(throttle.record_success(8.0, target_latency_seconds=2.0))
    assert throttle.current_concurrency == 2  # halved from 4


def test_record_success_per_call_target_does_not_penalize_a_naturally_large_call():
    """The same latency that would look like a backoff signal under a small
    call's target must NOT be misjudged as slow for a call whose own target
    is larger -- each call is judged only against its own baseline."""
    throttle = LatencyAdaptiveThrottle(initial=4, target_latency_seconds=15.0, min_concurrency=1)
    _run(throttle.record_success(8.0, target_latency_seconds=15.0))
    assert throttle.current_concurrency == 4  # unchanged -- 8s is fine for a 15s-target call


def test_record_success_per_call_target_still_grows_concurrency_after_enough_fast_calls():
    throttle = LatencyAdaptiveThrottle(initial=4, target_latency_seconds=15.0, min_concurrency=1)
    for _ in range(throttle.INCREASE_AFTER):
        _run(throttle.record_success(1.0, target_latency_seconds=2.0))
    assert throttle.current_concurrency == 5  # grew by 1, same shared _target as any other caller


def test_slot_concurrency_pool_is_shared_regardless_of_which_targets_are_reported():
    """Two different "call kinds" reporting two different targets must still
    draw from the exact same concurrency pool, not two independent ones."""
    throttle = LatencyAdaptiveThrottle(initial=1, target_latency_seconds=15.0)

    async def _run_scenario():
        async with throttle.slot():
            second_acquired = {"value": False}

            async def _second_caller():
                async with throttle.slot():
                    second_acquired["value"] = True

            task = asyncio.ensure_future(_second_caller())
            await asyncio.sleep(0.05)
            assert second_acquired["value"] is False
        await task  # released now that the first slot exited
        assert second_acquired["value"] is True

    _run(_run_scenario())
