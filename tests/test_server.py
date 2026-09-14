import json

import pytest

from ie.engine.engine import EngineConfig, build_engine
from ie.model.transformer import ModelConfig
from ie.serve.metrics import Histogram, ServerMetrics
from ie.serve.openai import (
    completion_chunk,
    completion_response,
    done_event,
    finish_reason_for,
    server_sent_event,
)
from ie.serve.server import InferenceServer, RequestState

CONFIG = ModelConfig(vocab_size=256, hidden_size=32, num_layers=2, num_heads=4,
                     num_kv_heads=2, head_dim=8, intermediate_size=64)
ENGINE = EngineConfig(block_size=16, num_blocks=256, max_batch_tokens=512)


def server():
    return InferenceServer(build_engine(CONFIG, ENGINE, model_seed=1))


def test_histogram_reports_quantiles():
    histogram = Histogram("h")
    for value in (1.0, 2.0, 3.0, 4.0):
        histogram.observe(value)
    assert histogram.count == 4
    assert histogram.mean == pytest.approx(2.5)
    assert histogram.quantile(0.5) == pytest.approx(2.5)


def test_metrics_snapshot_exposes_the_serving_signals():
    snapshot = ServerMetrics().snapshot()
    for key in ("requests_received", "requests_in_flight", "kv_utilization",
                "ttft_p50", "itl_p99", "e2e_p50"):
        assert key in snapshot


def test_prometheus_text_has_one_metric_per_line():
    text = ServerMetrics().prometheus_text()
    assert all(len(line.split()) == 2 for line in text.splitlines())
    assert "inference_requests_received 0" in text


def test_streaming_emits_tokens_one_at_a_time():
    api = server()
    served = api.submit([1, 2, 3], max_new_tokens=5)
    chunks = list(api.stream())

    assert [c.index for c in chunks] == [0, 1, 2, 3, 4]
    assert [c.token_id for c in chunks] == served.tokens
    assert chunks[-1].finished
    assert chunks[-1].stop_reason == "length"


def test_streamed_tokens_equal_the_batch_result():
    expected = build_engine(CONFIG, ENGINE, model_seed=1).generate(
        [[4, 5, 6]], max_new_tokens=6, temperature=0.0)[0]

    api = server()
    served = api.submit([4, 5, 6], max_new_tokens=6)
    api.drain()
    assert served.tokens == expected


def test_two_requests_interleave_while_streaming():
    api = server()
    api.submit([1, 2, 3], max_new_tokens=4, request_id="a")
    api.submit([9, 8, 7], max_new_tokens=4, request_id="b")

    order = [chunk.request_id for chunk in api.stream()]
    assert set(order) == {"a", "b"}
    assert order.count("a") == 4 and order.count("b") == 4
    assert order[:2] != ["a", "a"] or order[2:4] != ["a", "a"]


def test_cancelling_releases_the_blocks():
    api = server()
    api.submit([1, 2, 3], max_new_tokens=64, request_id="doomed")
    api.step()

    free_before = api.engine.blocks.num_free
    assert api.cancel("doomed")
    assert api.engine.blocks.num_free > free_before
    assert api.requests["doomed"].state is RequestState.CANCELLED
    assert api.metrics.requests_cancelled == 1
    assert not api.has_work
    assert api.metrics.snapshot()["requests_in_flight"] == 0


def test_cancelling_twice_is_a_no_op():
    api = server()
    api.submit([1, 2], max_new_tokens=8, request_id="x")
    assert api.cancel("x")
    assert not api.cancel("x")
    assert not api.cancel("never-existed")


def test_an_empty_prompt_is_rejected():
    api = server()
    with pytest.raises(ValueError):
        api.submit([], max_new_tokens=4)
    assert api.metrics.requests_failed == 1


def test_duplicate_request_ids_are_rejected():
    api = server()
    api.submit([1, 2], request_id="dup")
    with pytest.raises(ValueError):
        api.submit([3, 4], request_id="dup")


def test_metrics_record_latency_and_queue_depth():
    api = server()
    api.submit([1, 2, 3], max_new_tokens=5)
    api.submit([4, 5, 6], max_new_tokens=5)
    api.drain()

    snapshot = api.metrics.snapshot()
    assert snapshot["requests_received"] == 2
    assert snapshot["requests_finished"] == 2
    assert snapshot["generated_tokens"] == 10
    assert api.metrics.time_to_first_token.count == 2
    assert api.metrics.inter_token_latency.count == 8
    assert api.metrics.end_to_end.count == 2
    assert api.metrics.completion_rate == 1.0


def test_finish_reasons_map_to_the_openai_vocabulary():
    assert finish_reason_for("eos") == "stop"
    assert finish_reason_for("length") == "length"
    assert finish_reason_for(None) is None


def test_completion_response_has_the_expected_shape():
    api = server()
    served = api.submit([1, 2, 3], max_new_tokens=3)
    api.drain()

    payload = completion_response(served, model="ie-small", text="hello", created=1700000000)
    assert payload["object"] == "text_completion"
    assert payload["choices"][0]["finish_reason"] == "length"
    assert payload["usage"]["prompt_tokens"] == 3
    assert payload["usage"]["completion_tokens"] == 3
    assert payload["usage"]["total_tokens"] == 6


def test_chunks_serialize_as_server_sent_events():
    chunk = completion_chunk("req-1", "ie-small", " world", index=0, created=1700000000)
    event = server_sent_event(chunk)
    assert event.startswith("data: ") and event.endswith("\n\n")
    assert json.loads(event[6:].strip())["id"] == "req-1"
    assert done_event() == "data: [DONE]\n\n"
