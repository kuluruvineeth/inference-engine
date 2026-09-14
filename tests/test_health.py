import pytest

from ie.engine.engine import EngineConfig, build_engine
from ie.model.transformer import ModelConfig
from ie.serve.health import HealthProbe, Readiness, probe_from_engine

CONFIG = ModelConfig(vocab_size=64, hidden_size=32, num_layers=2, num_heads=4,
                     num_kv_heads=2, head_dim=8, intermediate_size=64)


def test_a_fresh_probe_is_starting_not_ready():
    probe = HealthProbe()
    assert probe.state is Readiness.STARTING
    assert probe.is_live
    assert not probe.accepts_traffic


def test_a_fully_warmed_probe_accepts_traffic():
    probe = HealthProbe(weights_loaded=True, cache_allocated=True, warmed_up=True)
    assert probe.state is Readiness.READY
    assert probe.accepts_traffic


def test_a_draining_replica_stays_live_but_refuses_new_work():
    probe = HealthProbe(weights_loaded=True, cache_allocated=True, warmed_up=True,
                        draining=True)
    assert probe.state is Readiness.DRAINING
    assert probe.is_live
    assert not probe.accepts_traffic


def test_an_error_makes_the_replica_unhealthy():
    probe = HealthProbe(weights_loaded=True, cache_allocated=True, warmed_up=True,
                        last_error="cuda out of memory")
    assert probe.state is Readiness.UNHEALTHY
    assert not probe.is_live
    assert not probe.accepts_traffic


def test_a_half_started_replica_never_reports_ready():
    assert not HealthProbe(weights_loaded=True).accepts_traffic
    assert not HealthProbe(weights_loaded=True, cache_allocated=True).accepts_traffic


def test_a_built_engine_reports_ready():
    engine = build_engine(CONFIG, EngineConfig(block_size=16, num_blocks=64))
    probe = probe_from_engine(engine)
    assert probe.accepts_traffic
    assert probe.snapshot()["state"] == "ready"


def test_a_draining_engine_is_reported_as_draining():
    engine = build_engine(CONFIG, EngineConfig(block_size=16, num_blocks=64))
    assert probe_from_engine(engine, draining=True).state is Readiness.DRAINING


def test_snapshot_carries_every_field():
    snapshot = HealthProbe(weights_loaded=True).snapshot()
    for key in ("state", "live", "ready", "weights_loaded", "cache_allocated",
                "warmed_up", "error"):
        assert key in snapshot
