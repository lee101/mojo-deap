from __future__ import annotations

import copy
import random

import numpy as np
import pytest
from deap import tools as deap_tools

from mojodeap import tools


def paired_call(upstream, port, args, seed=1729):
    upstream_args = copy.deepcopy(args)
    port_args = copy.deepcopy(args)
    random.seed(seed)
    upstream_result = upstream(*upstream_args)
    upstream_next = random.random()
    random.seed(seed)
    port_result = port(*port_args)
    port_next = random.random()
    assert port_next == upstream_next
    return upstream_args, port_args, upstream_result, port_result


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("cxOnePoint", ()),
        ("cxTwoPoint", ()),
        ("cxUniform", (0.37,)),
        ("cxMessyOnePoint", ()),
    ],
)
def test_sequence_crossovers_match_deap(name, extra):
    first = list(range(30))
    second = list(range(100, 135))
    upstream, port, upstream_result, port_result = paired_call(
        getattr(deap_tools, name),
        getattr(tools, name),
        (first, second, *extra),
    )
    assert port[:2] == upstream[:2]
    assert port_result[0] is port[0] and port_result[1] is port[1]
    assert upstream_result[0] is upstream[0]


@pytest.mark.parametrize("name", ["cxTwoPoints", "cxESTwoPoints"])
def test_deprecated_aliases_match_deap(name):
    if name == "cxESTwoPoints":
        first = ESIndividual(range(10), [0.1] * 10)
        second = ESIndividual(range(10, 20), [0.2] * 10)
    else:
        first = list(range(10))
        second = list(range(10, 20))
    upstream, port, _, _ = paired_call(
        getattr(deap_tools, name),
        getattr(tools, name),
        (first, second),
    )
    assert port[0] == upstream[0]
    assert port[1] == upstream[1]
    if name == "cxESTwoPoints":
        assert port[0].strategy == upstream[0].strategy
        assert port[1].strategy == upstream[1].strategy


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("cxPartialyMatched", ()),
        ("cxUniformPartialyMatched", (0.41,)),
        ("cxOrdered", ()),
    ],
)
def test_permutation_crossovers_match_deap_for_lists(name, extra):
    first = list(range(128))
    second = list(reversed(first))
    upstream, port, _, result = paired_call(
        getattr(deap_tools, name),
        getattr(tools, name),
        (first, second, *extra),
    )
    assert port[0] == upstream[0]
    assert port[1] == upstream[1]
    assert sorted(result[0]) == first
    assert sorted(result[1]) == first


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("cxPartialyMatched", ()),
        ("cxUniformPartialyMatched", (0.63,)),
        ("cxOrdered", ()),
    ],
)
def test_permutation_crossovers_match_deap_for_numpy(name, extra):
    first = np.arange(257, dtype=np.int64)
    second = first[::-1].copy()
    upstream, port, _, _ = paired_call(
        getattr(deap_tools, name),
        getattr(tools, name),
        (first, second, *extra),
        seed=91,
    )
    assert np.array_equal(port[0], upstream[0])
    assert np.array_equal(port[1], upstream[1])


@pytest.mark.parametrize(
    "name",
    ["cxPartialyMatched", "cxUniformPartialyMatched", "cxOrdered"],
)
def test_permutation_crossovers_reject_unsafe_gene_values(name):
    extra = (0.5,) if name == "cxUniformPartialyMatched" else ()
    with pytest.raises(ValueError, match="permutation crossover requires"):
        getattr(tools, name)([0, 1, 99], [2, 1, 0], *extra)
    with pytest.raises(ValueError, match="permutation crossover requires"):
        getattr(tools, name)([0, 0, 2], [2, 1, 0], *extra)


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("cxBlend", (0.35,)),
        ("cxSimulatedBinary", (12.0,)),
        ("cxSimulatedBinaryBounded", (15.0, -2.0, 3.0)),
    ],
)
def test_real_crossovers_match_deap(name, extra):
    first = np.linspace(-1.5, 1.5, 501)
    second = np.linspace(2.0, -1.0, 501)
    upstream, port, _, result = paired_call(
        getattr(deap_tools, name),
        getattr(tools, name),
        (first, second, *extra),
        seed=712,
    )
    assert result[0] is port[0] and result[1] is port[1]
    tolerance = 1e-9 if "SimulatedBinary" in name else 2e-14
    assert port[0] == pytest.approx(
        upstream[0], rel=tolerance, abs=tolerance
    )
    assert port[1] == pytest.approx(
        upstream[1], rel=tolerance, abs=tolerance
    )


def test_bounded_crossover_vector_bounds_matches_deap():
    first = np.linspace(-1.0, 1.0, 100)
    second = first[::-1].copy()
    low = np.linspace(-3.0, -2.0, 100).tolist()
    up = np.linspace(2.0, 4.0, 100).tolist()
    upstream, port, _, _ = paired_call(
        deap_tools.cxSimulatedBinaryBounded,
        tools.cxSimulatedBinaryBounded,
        (first, second, 20.0, low, up),
    )
    assert port[0] == pytest.approx(upstream[0])
    assert port[1] == pytest.approx(upstream[1])


class ESIndividual(list):
    def __init__(self, values, strategy):
        super().__init__(values)
        self.strategy = list(strategy)


@pytest.mark.parametrize("name", ["cxESBlend", "cxESTwoPoint"])
def test_es_crossovers_match_deap(name):
    first = ESIndividual(np.linspace(-1, 1, 80), np.linspace(0.1, 0.5, 80))
    second = ESIndividual(np.linspace(2, 0, 80), np.linspace(0.7, 0.2, 80))
    extra = (0.25,) if name == "cxESBlend" else ()
    upstream, port, _, _ = paired_call(
        getattr(deap_tools, name),
        getattr(tools, name),
        (first, second, *extra),
        seed=303,
    )
    assert port[0] == pytest.approx(upstream[0])
    assert port[1] == pytest.approx(upstream[1])
    assert port[0].strategy == pytest.approx(upstream[0].strategy)
    assert port[1].strategy == pytest.approx(upstream[1].strategy)


@pytest.mark.parametrize(
    ("name", "extra"),
    [
        ("mutGaussian", (0.2, 1.3, 0.41)),
        ("mutPolynomialBounded", (18.0, -3.0, 3.0, 0.52)),
    ],
)
def test_real_mutations_match_deap(name, extra):
    individual = np.linspace(-2.0, 2.0, 503)
    upstream, port, _, result = paired_call(
        getattr(deap_tools, name),
        getattr(tools, name),
        (individual, *extra),
        seed=81,
    )
    assert result[0] is port[0]
    tolerance = 1e-9 if name == "mutPolynomialBounded" else 2e-14
    assert port[0] == pytest.approx(
        upstream[0], rel=tolerance, abs=tolerance
    )


def test_real_mutations_accept_per_gene_parameters():
    individual = np.linspace(-0.5, 0.5, 127).tolist()
    mu = np.linspace(-0.1, 0.1, 127).tolist()
    sigma = np.linspace(0.2, 0.8, 127).tolist()
    upstream, port, _, _ = paired_call(
        deap_tools.mutGaussian,
        tools.mutGaussian,
        (individual, mu, sigma, 0.6),
    )
    assert port[0] == pytest.approx(upstream[0])


@pytest.mark.parametrize("size", [1, 3, 4, 5, 7, 8, 9, 17])
def test_polynomial_simd_tail_matches_deap(size):
    individual = np.linspace(-0.8, 0.8, size)
    upstream, port, _, _ = paired_call(
        deap_tools.mutPolynomialBounded,
        tools.mutPolynomialBounded,
        (individual, 20.0, -1.0, 1.0, 1.0),
        seed=406,
    )
    assert port[0] == pytest.approx(
        upstream[0], rel=1e-9, abs=1e-9
    )


def test_polynomial_zero_active_preserves_rng_state():
    individual = np.linspace(-0.8, 0.8, 641)
    upstream, port, _, _ = paired_call(
        deap_tools.mutPolynomialBounded,
        tools.mutPolynomialBounded,
        (individual, 20.0, -1.0, 1.0, -1.0),
        seed=815,
    )
    assert np.array_equal(port[0], upstream[0])


def test_polynomial_gpu_path_matches_cpu_and_rng_state():
    original = np.linspace(-0.9, 0.9, 10_003)
    cpu = original.copy()
    gpu = original.copy()
    random.seed(517)
    tools.mutPolynomialBounded(cpu, 20.0, -1.0, 1.0, 0.73)
    cpu_next = random.random()
    random.seed(517)
    tools.mutPolynomialBounded(
        gpu, 20.0, -1.0, 1.0, 0.73, device="gpu"
    )
    gpu_next = random.random()
    assert gpu_next == cpu_next
    assert gpu == pytest.approx(cpu, rel=1e-9, abs=1e-9)


def test_polynomial_rejects_unknown_device():
    with pytest.raises(ValueError, match="device must be"):
        tools.mutPolynomialBounded(
            [0.0], 20.0, -1.0, 1.0, 0.2, device="accelerator"
        )


def test_empty_native_backed_operators_do_not_pass_null_buffers():
    assert tools.cxUniform([], [], 0.5) == ([], [])
    assert tools.cxBlend([], [], 0.5) == ([], [])
    assert tools.cxSimulatedBinary([], [], 10.0) == ([], [])
    assert tools.mutGaussian([], 0.0, 1.0, 0.5) == ([],)
    assert tools.mutPolynomialBounded([], 20.0, -1.0, 1.0, 0.5) == ([],)
    assert tools.mutShuffleIndexes([], 0.5) == ([],)
    assert tools.mutFlipBit([], 0.5) == ([],)
    assert tools.mutUniformInt([], 0, 1, 0.5) == ([],)


def test_integer_native_conversion_rejects_uint64_narrowing():
    values = np.array([0, np.iinfo(np.uint64).max], dtype=np.uint64)
    with pytest.raises(OverflowError, match="represented as int64"):
        tools.mutShuffleIndexes(values, 0.0)


@pytest.mark.parametrize(
    ("name", "individual", "extra"),
    [
        ("mutShuffleIndexes", list(range(100)), (0.28,)),
        ("mutFlipBit", [False, True] * 50, (0.32,)),
        ("mutUniformInt", [2] * 100, (-4, 7, 0.49)),
        ("mutInversion", list(range(100)), ()),
    ],
)
def test_discrete_mutations_match_deap(name, individual, extra):
    upstream, port, _, result = paired_call(
        getattr(deap_tools, name),
        getattr(tools, name),
        (individual, *extra),
        seed=1024,
    )
    assert port[0] == upstream[0]
    assert result[0] is port[0]


def test_discrete_numpy_fast_paths_match_deap():
    bools = np.resize(np.array([False, True]), 1000)
    upstream, port, _, _ = paired_call(
        deap_tools.mutFlipBit, tools.mutFlipBit, (bools, 0.31)
    )
    assert np.array_equal(port[0], upstream[0])

    integers = np.arange(1000, dtype=np.int64) % 11
    upstream, port, _, _ = paired_call(
        deap_tools.mutUniformInt,
        tools.mutUniformInt,
        (integers, -5, 17, 0.44),
    )
    assert np.array_equal(port[0], upstream[0])


def test_es_log_normal_matches_deap():
    individual = ESIndividual(np.linspace(-1, 1, 301), np.full(301, 0.25))
    upstream, port, _, result = paired_call(
        deap_tools.mutESLogNormal,
        tools.mutESLogNormal,
        (individual, 1.0, 0.57),
        seed=778,
    )
    assert result[0] is port[0]
    assert port[0] == pytest.approx(upstream[0], rel=2e-14, abs=2e-14)
    assert port[0].strategy == pytest.approx(
        upstream[0].strategy, rel=2e-14, abs=2e-14
    )


def test_parameter_length_errors_match_deap_contract():
    with pytest.raises(IndexError, match="mu must be at least"):
        tools.mutGaussian([0.0] * 5, [0.0] * 4, 1.0, 0.5)
    with pytest.raises(IndexError, match="shorter individual"):
        tools.cxSimulatedBinaryBounded(
            [0.0] * 5, [1.0] * 5, 10.0, [0.0] * 4, 1.0
        )
    random.seed(38)
    state = random.getstate()
    with pytest.raises(IndexError, match="low must be at least"):
        tools.mutPolynomialBounded(
            [0.0] * 5, 20.0, [0.0] * 4, 1.0, -1.0
        )
    assert random.getstate() == state


def test_blend_batch_matches_formula():
    first = np.linspace(-2, 1, 10_000).reshape(100, 100).copy()
    second = np.linspace(3, -1, 10_000).reshape(100, 100).copy()
    expected_first = first.copy()
    expected_second = second.copy()
    random_values = np.random.default_rng(42).random(first.size).reshape(first.shape)
    gamma = 1.5 * random_values - 0.25
    expected_first[:] = (1.0 - gamma) * first + gamma * second
    expected_second[:] = gamma * first + (1.0 - gamma) * second
    result = tools.cxBlendBatch(
        first, second, 0.25, rng=np.random.default_rng(42)
    )
    assert result[0] is first and result[1] is second
    assert first == pytest.approx(expected_first)
    assert second == pytest.approx(expected_second)


def test_sbx_batch_matches_formula():
    first = np.linspace(-2, 1, 5000).copy()
    second = np.linspace(3, -1, 5000).copy()
    original_first = first.copy()
    original_second = second.copy()
    random_values = np.random.default_rng(7).random(first.size)
    beta = np.where(
        random_values <= 0.5,
        2.0 * random_values,
        1.0 / (2.0 * (1.0 - random_values)),
    ) ** (1.0 / 16.0)
    expected_first = 0.5 * (
        (1.0 + beta) * original_first + (1.0 - beta) * original_second
    )
    expected_second = 0.5 * (
        (1.0 - beta) * original_first + (1.0 + beta) * original_second
    )
    tools.cxSimulatedBinaryBatch(
        first, second, 15.0, rng=np.random.default_rng(7)
    )
    assert first == pytest.approx(expected_first)
    assert second == pytest.approx(expected_second)


def test_gaussian_batch_matches_generated_mask_and_noise():
    population = np.zeros((120, 90), dtype=np.float64)
    generator = np.random.default_rng(99)
    active = generator.random(population.size) < 0.4
    delta = generator.normal(0.3, 1.2, population.size)
    expected = np.where(active, delta, 0.0).reshape(population.shape)
    result = tools.mutGaussianBatch(
        population, 0.3, 1.2, 0.4, rng=np.random.default_rng(99)
    )
    assert result[0] is population
    assert np.array_equal(population, expected)
