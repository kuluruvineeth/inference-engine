import pytest

from ie.serve.autoscale import (
    AdmissionController,
    Autoscaler,
    CapacityPlan,
    ScalingAction,
    ScalingPolicy,
)

PLAN = CapacityPlan(num_blocks=1024, block_size=16, bytes_per_token=2048)


def test_capacity_reports_total_tokens_and_bytes():
    assert PLAN.total_tokens == 16384
    assert PLAN.total_bytes == 16384 * 2048


def test_blocks_round_up_to_whole_blocks():
    assert PLAN.blocks_for(1) == 1
    assert PLAN.blocks_for(16) == 1
    assert PLAN.blocks_for(17) == 2


def test_concurrency_falls_as_sequences_get_longer():
    short = PLAN.concurrent_sequences(prompt_tokens=128, generated_tokens=128)
    long = PLAN.concurrent_sequences(prompt_tokens=2048, generated_tokens=512)
    assert short > long
    assert long >= 1


def test_a_shared_prefix_buys_more_concurrency():
    private_only = PLAN.concurrent_sequences(prompt_tokens=1024, generated_tokens=64)
    with_sharing = PLAN.concurrent_sequences(prompt_tokens=1024, generated_tokens=64,
                                             shared_prefix_tokens=960)
    assert with_sharing > private_only


def test_blocks_needed_scales_with_concurrency():
    one = PLAN.blocks_needed_for(1, prompt_tokens=256, generated_tokens=256)
    ten = PLAN.blocks_needed_for(10, prompt_tokens=256, generated_tokens=256)
    assert ten == 10 * one


def test_a_sequence_too_large_for_the_pool_yields_zero_concurrency():
    tiny = CapacityPlan(num_blocks=2, block_size=16, bytes_per_token=1)
    assert tiny.concurrent_sequences(prompt_tokens=4096, generated_tokens=0) == 0


def test_policy_rejects_inconsistent_thresholds():
    with pytest.raises(ValueError):
        ScalingPolicy(scale_up_at=0.5, scale_down_at=0.9)
    with pytest.raises(ValueError):
        ScalingPolicy(target_per_replica=0)
    with pytest.raises(ValueError):
        ScalingPolicy(min_replicas=4, max_replicas=2)


def test_load_below_the_band_scales_down():
    scaler = Autoscaler(ScalingPolicy(target_per_replica=8, min_replicas=1), replicas=4)
    decision = scaler.observe(in_flight=2)
    assert decision.action is ScalingAction.SCALE_DOWN
    assert scaler.replicas == 3


def test_load_above_the_band_scales_up():
    scaler = Autoscaler(ScalingPolicy(target_per_replica=8, max_replicas=16), replicas=1)
    decision = scaler.observe(in_flight=40)
    assert decision.action is ScalingAction.SCALE_UP
    assert scaler.replicas == 5


def test_load_inside_the_band_holds():
    scaler = Autoscaler(ScalingPolicy(target_per_replica=8), replicas=2)
    assert scaler.observe(in_flight=12).action is ScalingAction.HOLD
    assert scaler.replicas == 2


def test_scaling_never_exceeds_the_ceiling():
    scaler = Autoscaler(ScalingPolicy(target_per_replica=4, max_replicas=3), replicas=1)
    for _ in range(10):
        scaler.observe(in_flight=500)
    assert scaler.replicas == 3


def test_scaling_never_drops_below_the_floor():
    scaler = Autoscaler(ScalingPolicy(target_per_replica=8, min_replicas=2), replicas=5)
    for _ in range(10):
        scaler.observe(in_flight=0)
    assert scaler.replicas == 2


def test_scale_to_zero_waits_for_the_idle_window():
    policy = ScalingPolicy(target_per_replica=8, min_replicas=1, scale_to_zero_after=30.0)
    scaler = Autoscaler(policy, replicas=1)

    assert scaler.observe(in_flight=0, seconds_since_last=10.0).replicas == 1
    assert scaler.observe(in_flight=0, seconds_since_last=10.0).replicas == 1
    final = scaler.observe(in_flight=0, seconds_since_last=15.0)
    assert final.action is ScalingAction.SCALE_DOWN
    assert final.replicas == 0


def test_traffic_after_zero_triggers_a_cold_start():
    policy = ScalingPolicy(target_per_replica=8, min_replicas=1, scale_to_zero_after=1.0,
                           cold_start_seconds=25.0)
    scaler = Autoscaler(policy, replicas=0)

    decision = scaler.observe(in_flight=1)
    assert decision.action is ScalingAction.SCALE_UP
    assert "cold start" in decision.reason
    assert scaler.time_to_serve(was_cold=True) == 25.0
    assert scaler.time_to_serve(was_cold=False) == 0.0


def test_activity_resets_the_idle_timer():
    policy = ScalingPolicy(min_replicas=1, scale_to_zero_after=20.0)
    scaler = Autoscaler(policy, replicas=1)
    scaler.observe(in_flight=0, seconds_since_last=15.0)
    scaler.observe(in_flight=3, seconds_since_last=1.0)
    assert scaler.idle_seconds == 0.0
    assert scaler.observe(in_flight=0, seconds_since_last=10.0).replicas == 1


def test_admission_accepts_until_the_queue_is_full():
    controller = AdmissionController(max_queue_depth=3)
    assert controller.admit(queue_depth=0, prompt_tokens=100)[0]
    assert controller.admit(queue_depth=2, prompt_tokens=100)[0]
    allowed, reason = controller.admit(queue_depth=3, prompt_tokens=100)
    assert not allowed and reason == "queue is full"


def test_admission_rejects_an_oversized_prompt():
    controller = AdmissionController(max_queue_depth=10, max_prompt_tokens=512)
    allowed, reason = controller.admit(queue_depth=0, prompt_tokens=4096)
    assert not allowed and "maximum accepted length" in reason


def test_rejection_rate_is_tracked():
    controller = AdmissionController(max_queue_depth=1)
    controller.admit(queue_depth=0, prompt_tokens=10)
    controller.admit(queue_depth=5, prompt_tokens=10)
    assert controller.rejection_rate == pytest.approx(0.5)
