from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import qnn_common as common

logger = logging.getLogger(__name__)

_CONFIG = None
CONFIG = None
CONFIG_PATH = None
APPROACH = None
RANDOM_SEED = None
DATASET_NAME = None
TEST_SPLIT = None
VAL_SPLIT = None
ENABLE_INPUT_PARAMS = None
PARAM_BLOCK_PROB = None
MANUAL_INPUT_VALUES = None
NUM_PARAMS = None
SIMULATOR_DEVICE = None
_ENCODING_MODE = None
_RANDOM_GENERATOR = None
_STATEVECTOR_THREADS = None

q_strategy = None
EvolutionaryIndividual = None
build_quantum_circuit = None
get_param_indices = None
generate_guided_individual = None
mut_quantum_circuit = None
serialize_individual = None
deserialize_individual = None

cx_quantum_circuit = common.cx_quantum_circuit


def _resolve_device(requested_device: str) -> str:
    if requested_device == "GPU":
        logger.warning(
            "El cálculo ahora es exacto vía Statevector/StabilizerState (numpy, solo CPU); "
            "se ignora la solicitud de GPU."
        )
    return "CPU"


def init_config(dataset_name: str, approach: str | None = None) -> None:
    global _CONFIG, CONFIG, CONFIG_PATH, APPROACH, RANDOM_SEED, DATASET_NAME
    global TEST_SPLIT, VAL_SPLIT, ENABLE_INPUT_PARAMS, PARAM_BLOCK_PROB
    global MANUAL_INPUT_VALUES, NUM_PARAMS, SIMULATOR_DEVICE
    global _ENCODING_MODE, _RANDOM_GENERATOR, _STATEVECTOR_THREADS
    global q_strategy, EvolutionaryIndividual, build_quantum_circuit
    global get_param_indices, generate_guided_individual, mut_quantum_circuit
    global serialize_individual, deserialize_individual

    config_dir = Path(__file__).parent / "configs"
    config_path = config_dir / f"{dataset_name}.json"
    if not config_path.exists():
        config_path = Path(__file__).with_name("config.json")
    CONFIG_PATH = config_path

    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG = json.load(f)
    CONFIG = _CONFIG

    APPROACH = _CONFIG["approach"]
    if approach is not None:
        APPROACH = approach

    raw_random_seed = _CONFIG.get("random_seed")
    RANDOM_SEED = None if raw_random_seed is None else int(raw_random_seed)
    _RANDOM_GENERATOR = np.random.default_rng(RANDOM_SEED)

    _qnn_cfg = _CONFIG.get("qnn", {})
    DATASET_NAME = _qnn_cfg.get("dataset", dataset_name)
    TEST_SPLIT = float(_qnn_cfg.get("test_split", 0.2))
    VAL_SPLIT = float(_qnn_cfg.get("val_split", 0.2))

    _encoding_cfg = _CONFIG.get("encoding", {})
    ENABLE_INPUT_PARAMS = _encoding_cfg.get("enable_input_params", False)
    PARAM_BLOCK_PROB = _encoding_cfg.get("param_block_prob", 0.15)

    _manual_input_values = _encoding_cfg.get("manual_input_values", [1.0])
    if not isinstance(_manual_input_values, list) or len(_manual_input_values) == 0:
        raise ValueError(
            "encoding.manual_input_values must be a non-empty list of numbers"
        )
    MANUAL_INPUT_VALUES = [float(v) for v in _manual_input_values]
    NUM_PARAMS = len(MANUAL_INPUT_VALUES)

    q_strategy = importlib.import_module(f".qnn_{APPROACH}", package=__package__)
    EvolutionaryIndividual = q_strategy.EvolutionaryIndividual
    build_quantum_circuit = q_strategy.build_quantum_circuit
    get_param_indices = q_strategy.get_param_indices
    generate_guided_individual = q_strategy.generate_guided_individual
    mut_quantum_circuit = q_strategy.mut_quantum_circuit
    serialize_individual = q_strategy.serialize_individual
    deserialize_individual = q_strategy.deserialize_individual

    _execution_cfg = _CONFIG.get("execution", {})
    _STATEVECTOR_THREADS = _execution_cfg.get("statevector_max_parallel_threads", 0)
    _REQUESTED_DEVICE = _execution_cfg.get("device", "auto")
    SIMULATOR_DEVICE = _resolve_device(_REQUESTED_DEVICE)

    _ENCODING_MODE = "clifford_angle" if APPROACH == "clifford" else "amplitude"
