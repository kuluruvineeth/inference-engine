import numpy as np
import pytest

from ie.engine.engine import Engine, EngineConfig, build_engine
from ie.model.transformer import ModelConfig, Transformer
from ie.parallel.pipeline import PipelineParallelModel

CONFIG = ModelConfig(vocab_size=128, hidden_size=32, num_layers=8, num_heads=4,
                     num_kv_heads=2, head_dim=8, intermediate_size=64)
ENGINE = EngineConfig(block_size=16, num_blocks=256, max_batch_tokens=256)
PROMPTS = [[1, 2, 3, 4], [9, 8, 7]]


def test_stages_cover_every_layer_exactly_once():
    pipeline = PipelineParallelModel(Transformer(CONFIG, seed=0), num_stages=4)
    covered = []
    for stage in pipeline.stages:
        covered.extend(range(stage.layer_slice.start, stage.layer_slice.stop))
    assert covered == list(range(CONFIG.num_layers))


def test_layers_are_split_evenly_when_they_divide():
    pipeline = PipelineParallelModel(Transformer(CONFIG, seed=0), num_stages=4)
    assert pipeline.layers_per_stage() == [2, 2, 2, 2]


def test_an_uneven_split_still_covers_everything():
    pipeline = PipelineParallelModel(Transformer(CONFIG, seed=0), num_stages=3)
    assert sum(pipeline.layers_per_stage()) == CONFIG.num_layers
    assert max(pipeline.layers_per_stage()) - min(pipeline.layers_per_stage()) <= 1


def test_more_stages_than_layers_is_rejected():
    with pytest.raises(ValueError):
        PipelineParallelModel(Transformer(CONFIG, seed=0), num_stages=CONFIG.num_layers + 1)


def test_first_and_last_stages_are_marked():
    pipeline = PipelineParallelModel(Transformer(CONFIG, seed=0), num_stages=3)
    assert pipeline.stages[0].is_first and not pipeline.stages[0].is_last
    assert pipeline.stages[-1].is_last and not pipeline.stages[-1].is_first


@pytest.mark.parametrize("num_stages", [1, 2, 4, 8])
def test_pipeline_generation_matches_the_single_stage_model(num_stages):
    expected = build_engine(CONFIG, ENGINE, model_seed=1).generate(
        PROMPTS, max_new_tokens=8, temperature=0.0)

    pipelined = Engine(PipelineParallelModel(Transformer(CONFIG, seed=1), num_stages), ENGINE)
    assert pipelined.generate(PROMPTS, max_new_tokens=8, temperature=0.0) == expected


def test_each_extra_stage_adds_one_activation_hop():
    for num_stages in (1, 2, 4):
        pipeline = PipelineParallelModel(Transformer(CONFIG, seed=1), num_stages)
        engine = Engine(pipeline, ENGINE)
        engine.generate([[1, 2, 3]], max_new_tokens=1, temperature=0.0)
        assert pipeline.stats.hops == (num_stages - 1) * pipeline.stats.stage_invocations // num_stages


def test_activation_traffic_is_reported():
    pipeline = PipelineParallelModel(Transformer(CONFIG, seed=1), num_stages=4)
    engine = Engine(pipeline, ENGINE)
    engine.generate([list(range(32))], max_new_tokens=4, temperature=0.0)
    assert pipeline.stats.bytes_sent > 0
    assert pipeline.stats.activations_sent > 0
