from __future__ import annotations

import functools
import operator
import random
from typing import List, Set, Tuple, Union

import networkx as nx
import numpy as np
from deap import gp
from qiskit import QuantumCircuit

from . import max_cut_common as common

BLOCK_HOF: List[List[Tuple]] = []

pset = gp.PrimitiveSet("MAIN", 2)
pset.renameArguments(ARG0="in_val", ARG1="w_val")

pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)


def safe_mod(a, b):
    return a % b if abs(b) > 0.001 else a


pset.addPrimitive(safe_mod, 2)
pset.addPrimitive(np.sin, 1)
pset.addPrimitive(np.cos, 1)


def generate_rand_const():
    return random.randint(-3, 3)


pset.addEphemeralConstant("rand_const", generate_rand_const)

QuantumGen = Union[
    Tuple[str, int],
    Tuple[str, int, int],
    Tuple[str, int, List[Tuple], gp.PrimitiveTree],
]
EvolutionaryIndividual = List[QuantumGen]

CLIFFORD_GATES = ["H", "S", "CX"]

load_external_maxcut_instance = common.load_external_maxcut_instance


def generate_random_gate(num_qubits: int, graph_instance: nx.Graph) -> Tuple:
    gate = random.choice(CLIFFORD_GATES)
    if gate == "CX" and graph_instance.number_of_edges() > 0:
        edge = random.choice(list(graph_instance.edges()))
        return (
            ("CX", edge[0], edge[1])
            if random.random() > 0.5
            else ("CX", edge[1], edge[0])
        )
    return (gate, random.randint(0, num_qubits - 1))


def generate_random_block(num_qubits: int, graph_instance: nx.Graph) -> List[Tuple]:
    if len(BLOCK_HOF) > 0 and random.random() < 0.3:
        return list(random.choice(BLOCK_HOF))

    length = random.randint(2, 6)
    block = []
    for _ in range(length):
        gate = random.choice(CLIFFORD_GATES)
        if gate == "CX" and graph_instance.number_of_edges() > 0:
            u, v = random.choice(list(graph_instance.edges()))
            block.append(("CX", u, v))
        else:
            block.append((gate, random.randint(0, num_qubits - 1)))
    return block


def generate_random_param_block(
    num_qubits: int,
    graph_instance: nx.Graph,
    param_idx: int,
) -> QuantumGen:
    block_gates = generate_random_block(num_qubits, graph_instance)
    expr = gp.genHalfAndHalf(pset, min_=1, max_=4)
    tree = gp.PrimitiveTree(expr)
    return ("PARAM_BLOCK", param_idx, block_gates, tree)


def mutate_block_structure(
    block: List[Tuple], num_qubits: int, graph_instance: nx.Graph
) -> List[Tuple]:
    action = random.choice(["INSERT", "DELETE", "REPLACE"])
    if action == "DELETE" and len(block) > 1:
        block.pop(random.randrange(len(block)))
    elif action == "INSERT":
        new_gate = generate_random_gate(num_qubits, graph_instance)
        block.insert(random.randint(0, len(block)), new_gate)
    elif action == "REPLACE":
        block[random.randrange(len(block))] = generate_random_gate(
            num_qubits, graph_instance
        )
    return block


def generate_guided_individual(
    num_qubits: int,
    length: int,
    graph_instance: nx.Graph,
    max_params: int = 3,
    param_block_prob: float = 0.15,
    **_ignored,
) -> EvolutionaryIndividual:
    individual: EvolutionaryIndividual = [("H", q) for q in range(num_qubits)]
    remaining = max(0, length - num_qubits)

    for _ in range(remaining):
        if random.random() < param_block_prob:
            individual.append(
                generate_random_param_block(
                    num_qubits, graph_instance, random.randint(0, max_params - 1)
                )
            )
        else:
            individual.append(generate_random_gate(num_qubits, graph_instance))

    return individual


def get_param_indices(
    individual: EvolutionaryIndividual,
) -> Tuple[Set[int], Set[int]]:
    idx = {gen[1] for gen in individual if gen[0] == "PARAM_BLOCK"}
    return idx, idx


def mut_quantum_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    graph_instance: nx.Graph,
    indpb: float,
    max_params: int = 3,
    **_ignored,
) -> Tuple[EvolutionaryIndividual]:
    i = num_qubits

    while i < len(individual):
        if random.random() < indpb:
            gen = individual[i]

            if gen[0] == "PARAM_BLOCK":
                param_idx, block_gates, tree = gen[1], gen[2], gen[3]

                if random.random() < 0.5:
                    new_block = mutate_block_structure(
                        list(block_gates), num_qubits, graph_instance
                    )
                    individual[i] = ("PARAM_BLOCK", param_idx, new_block, tree)
                else:
                    new_tree = gp.mutUniform(
                        tree,
                        expr=functools.partial(gp.genHalfAndHalf, min_=1, max_=2),
                        pset=pset,
                    )[0]
                    individual[i] = (
                        "PARAM_BLOCK",
                        param_idx,
                        block_gates,
                        gp.PrimitiveTree(new_tree),
                    )
                i += 1
                continue

            action = random.choice(["INSERT", "DELETE", "REPLACE"])
            if action == "DELETE" and len(individual) > num_qubits + 1:
                individual.pop(i)
                continue
            elif action == "REPLACE":
                individual[i] = (
                    generate_random_param_block(
                        num_qubits, graph_instance, random.randint(0, max_params - 1)
                    )
                    if random.random() < 0.2
                    else generate_random_gate(num_qubits, graph_instance)
                )
            elif action == "INSERT":
                item = (
                    generate_random_param_block(
                        num_qubits, graph_instance, random.randint(0, max_params - 1)
                    )
                    if random.random() < 0.2
                    else generate_random_gate(num_qubits, graph_instance)
                )
                individual.insert(i, item)
                i += 1
        i += 1

    return (individual,)


def build_quantum_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    input_values: List[float] = [0.0],
    weight_values: List[float] = [0.0],
    measure: bool = False,
) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)

    for gen in individual:
        if gen[0] == "PARAM_BLOCK":
            param_idx, block_gates, tree = gen[1], gen[2], gen[3]
            func = gp.compile(tree, pset)

            in_val = (
                input_values[param_idx % len(input_values)] if input_values else 0.0
            )
            w_val = (
                weight_values[param_idx % len(weight_values)] if weight_values else 0.0
            )

            try:
                raw = func(in_val, w_val)
                reps = max(1, min(int(abs(raw)), 8))
            except Exception:
                reps = 1

            for _ in range(reps):
                for b_gate in block_gates:
                    if b_gate[0] == "H":
                        qc.h(b_gate[1])
                    elif b_gate[0] == "S":
                        qc.s(b_gate[1])
                    elif b_gate[0] == "CX":
                        qc.cx(b_gate[1], b_gate[2])
        else:
            if gen[0] == "H":
                qc.h(gen[1])
            elif gen[0] == "S":
                qc.s(gen[1])
            elif gen[0] == "CX":
                qc.cx(gen[1], gen[2])

    if measure:
        qc.measure_all()
    return qc
