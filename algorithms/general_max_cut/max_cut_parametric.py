from __future__ import annotations

import functools
import random
from typing import Dict, List, Set, Tuple, Union

import networkx as nx
from qiskit import QuantumCircuit

from . import max_cut_common as common

QuantumGen = Union[
    Tuple[str, int],
    Tuple[str, int, int],
    Tuple[str, str, int, str, int],
]
EvolutionaryIndividual = List[QuantumGen]

CLIFFORD_GATES: List[str] = ["H", "S", "CX"]
PARAMETRIC_GATES: List[str] = ["RX", "RY", "RZ"]

load_external_maxcut_instance = common.load_external_maxcut_instance


def generate_random_gate(
    num_qubits: int,
    graph_instance: nx.Graph,
    max_qubits: int,
    max_params: int = 3,
    enable_input_params: bool = False,
    param_block_prob: float = 0.15,
) -> QuantumGen:
    if random.random() < param_block_prob:
        gate = random.choice(PARAMETRIC_GATES)
        p_type = random.choice(["INPUT", "WEIGHT"]) if enable_input_params else "WEIGHT"

        if p_type == "INPUT":
            pairs = common.enumerate_qubit_pairs(max_qubits)
            if pairs and random.random() < 0.8:
                i, j = random.choice(pairs)
                p_idx = common.pair_index(i, j, max_qubits)
                qubit = random.choice((i, j))
            else:
                p_idx = random.randint(0, max_params - 1) if max_params > 0 else 0
                qubit = random.randint(0, num_qubits - 1)
        else:
            p_idx = random.randint(0, max_params - 1) if max_params > 0 else 0
            qubit = random.randint(0, num_qubits - 1)

        return ("PARAM_BLOCK", p_type, p_idx, gate, qubit)

    gate = random.choice(CLIFFORD_GATES)
    if gate == "CX":
        i, j = random.choice(common.enumerate_qubit_pairs(max_qubits))
        return ("CX", i, j) if random.random() > 0.5 else ("CX", j, i)

    return (gate, random.randint(0, num_qubits - 1))


def generate_guided_individual(
    num_qubits: int,
    length: int,
    graph_instance: nx.Graph,
    max_qubits: int,
    max_params: int = 3,
    enable_input_params: bool = False,
    param_block_prob: float = 0.15,
    **_ignored,
) -> EvolutionaryIndividual:
    individual = [("H", i) for i in range(num_qubits)]
    remaining_length = max(0, length - num_qubits)
    for _ in range(remaining_length):
        individual.append(
            generate_random_gate(
                num_qubits,
                graph_instance,
                max_qubits,
                max_params=max_params,
                enable_input_params=enable_input_params,
                param_block_prob=param_block_prob,
            )
        )
    return individual


def get_param_indices(
    individual: EvolutionaryIndividual,
) -> Tuple[Set[int], Set[int]]:
    input_idx: Set[int] = set()
    weight_idx: Set[int] = set()

    for gen in individual:
        if gen[0] == "PARAM_BLOCK":
            _, p_type, p_idx, _, _ = gen
            (input_idx if p_type == "INPUT" else weight_idx).add(p_idx)

    return input_idx, weight_idx


def mut_quantum_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    graph_instance: nx.Graph,
    indpb: float,
    max_qubits: int,
    max_params: int = 3,
    enable_input_params: bool = False,
    param_block_prob: float = 0.15,
    **_ignored,
) -> Tuple[EvolutionaryIndividual]:
    gate_generator = functools.partial(
        generate_random_gate,
        max_qubits=max_qubits,
        max_params=max_params,
        enable_input_params=enable_input_params,
        param_block_prob=param_block_prob,
    )
    return common.mut_quantum_circuit(
        individual, num_qubits, graph_instance, indpb, gate_generator=gate_generator
    )


def describe_param_genes(
    individual: EvolutionaryIndividual,
    max_qubits: int,
) -> List[dict]:
    genes = []
    for position, gen in enumerate(individual):
        if gen[0] == "PARAM_BLOCK":
            _, p_type, p_idx, rot_gate, qubit = gen
            entry = {
                "position": position,
                "type": p_type,
                "param_idx": p_idx,
                "gate": rot_gate,
                "qubit": qubit,
            }
            if p_type == "INPUT":
                i, j = common.pair_from_index(p_idx, max_qubits)
                entry["linked_pair"] = [i, j]
            genes.append(entry)
    return genes


def build_quantum_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    input_values: List[float] = [0.0],
    weight_values: Dict[int, float] = {},
    measure: bool = False,
) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)

    for gen in individual:
        gate_type = gen[0]
        if gate_type == "PARAM_BLOCK":
            _, p_type, p_idx, rot_gate, qubit = gen
            if p_type == "INPUT":
                theta = input_values[p_idx % len(input_values)] if input_values else 0.0
            else:
                theta = weight_values.get(p_idx, 0.0)

            if rot_gate == "RX":
                qc.rx(theta, qubit)
            elif rot_gate == "RY":
                qc.ry(theta, qubit)
            elif rot_gate == "RZ":
                qc.rz(theta, qubit)
        else:
            if gate_type == "H":
                qc.h(gen[1])
            elif gate_type == "S":
                qc.s(gen[1])
            elif gate_type == "CX":
                qc.cx(gen[1], gen[2])

    if measure:
        qc.measure_all()
    return qc
