from __future__ import annotations

import functools
import math
import operator
import random
from typing import Callable, Dict, List, Set, Tuple, Union

from deap import gp
from qiskit import QuantumCircuit

from . import qnn_common as common
from .constants import (
    CLIFFORD_GATES,
    REPS_MIN,
    REPS_MAX,
    BLOCK_HOF_REUSE_PROB,
    BLOCK_MIN_GATES,
    BLOCK_MAX_GATES,
    MUTATE_ACTION_PROB_1,
    MUTATE_ACTION_PROB_2,
)

BLOCK_HOF: List[List[Tuple]] = []

pset = gp.PrimitiveSet("MAIN", 2)
pset.renameArguments(ARG0="in_val", ARG1="w_val")

pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)


def safe_mod(a, b):
    return a % b if abs(b) > 0.001 else a


pset.addPrimitive(safe_mod, 2)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)


def generate_rand_const():
    return random.randint(-3, 3)


pset.addEphemeralConstant("rand_const_qnn_clifford", generate_rand_const)

_COMPILED_TREE_CACHE: Dict[str, Callable] = {}
_BLOCK_CIRCUIT_CACHE: Dict[Tuple, QuantumCircuit] = {}


def _get_block_circuit(block_gates: tuple, num_qubits: int) -> QuantumCircuit:
    key = (block_gates, num_qubits)
    qc = _BLOCK_CIRCUIT_CACHE.get(key)
    if qc is None:
        qc = QuantumCircuit(num_qubits)
        for b_gate in block_gates:
            if b_gate[0] == "H":
                qc.h(b_gate[1])
            elif b_gate[0] == "S":
                qc.s(b_gate[1])
            elif b_gate[0] == "CX":
                qc.cx(b_gate[1], b_gate[2])
        _BLOCK_CIRCUIT_CACHE[key] = qc
    return qc


def get_compiled_repetition_function(tree: gp.PrimitiveTree) -> Callable:
    key = str(tree)
    compiled = _COMPILED_TREE_CACHE.get(key)
    if compiled is None:
        compiled = gp.compile(tree, pset)
        _COMPILED_TREE_CACHE[key] = compiled
    return compiled


QuantumGen = Union[
    Tuple[str, int],
    Tuple[str, int, int],
    Tuple[str, int, List[Tuple], gp.PrimitiveTree],
]
EvolutionaryIndividual = List[QuantumGen]


def generate_random_gate(num_qubits: int) -> Tuple:
    gate = random.choice(CLIFFORD_GATES)
    if gate == "CX":
        i, j = random.choice(common.enumerate_qubit_pairs(num_qubits))
        return ("CX", i, j) if random.random() > 0.5 else ("CX", j, i)
    return (gate, random.randint(0, num_qubits - 1))


def generate_random_block(num_qubits: int) -> List[Tuple]:
    if len(BLOCK_HOF) > 0 and random.random() < BLOCK_HOF_REUSE_PROB:
        return list(random.choice(BLOCK_HOF))
    length = random.randint(BLOCK_MIN_GATES, BLOCK_MAX_GATES)
    block = [generate_random_gate(num_qubits) for _ in range(length)]
    return common.simplify_gate_sequence(block, num_qubits)


def generate_random_param_block(
    num_qubits: int, param_idx: int, tree: gp.PrimitiveTree | None = None
) -> QuantumGen:
    block_gates = generate_random_block(num_qubits)
    if tree is None:
        expr = gp.genHalfAndHalf(pset, min_=1, max_=4)
        tree = gp.PrimitiveTree(expr)
    return ("PARAM_BLOCK", param_idx, block_gates, tree)


def mutate_block_structure(block: List[Tuple], num_qubits: int) -> List[Tuple]:
    action = random.choice(["INSERT", "DELETE", "REPLACE"])
    if action == "DELETE" and len(block) > 1:
        block.pop(random.randrange(len(block)))
    elif action == "INSERT":
        new_gate = generate_random_gate(num_qubits)
        block.insert(random.randint(0, len(block)), new_gate)
    elif action == "REPLACE":
        block[random.randrange(len(block))] = generate_random_gate(num_qubits)
    return common.simplify_gate_sequence(block, num_qubits)


def generate_guided_individual(
    num_qubits: int,
    length: int,
    max_params: int = 3,
    param_block_prob: float = 0.15,
    **ignored,
) -> EvolutionaryIndividual:
    individual: EvolutionaryIndividual = []
    for q in range(num_qubits):
        if random.random() < 0.5:
            individual.append(("H", q))
    remaining = max(0, length - len(individual))
    for _ in range(remaining):
        if random.random() < param_block_prob:
            individual.append(
                generate_random_param_block(
                    num_qubits, random.randint(0, max_params - 1)
                )
            )
        else:
            individual.append(generate_random_gate(num_qubits))
    return individual


def get_param_indices(
    individual: EvolutionaryIndividual,
) -> Tuple[Set[int], Set[int]]:
    idx = {gen[1] for gen in individual if gen[0] == "PARAM_BLOCK"}
    return idx, idx


def mut_quantum_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    indpb: float,
    max_params: int = 3,
    param_block_prob: float = 0.15,
    **ignored,
) -> Tuple[EvolutionaryIndividual]:
    i = num_qubits
    while i < len(individual):
        if random.random() < indpb:
            gen = individual[i]
            if gen[0] == "PARAM_BLOCK":
                param_idx, block_gates, tree = gen[1], gen[2], gen[3]
                action_roll = random.random()
                if action_roll < MUTATE_ACTION_PROB_1:
                    new_block = mutate_block_structure(list(block_gates), num_qubits)
                    individual[i] = ("PARAM_BLOCK", param_idx, new_block, tree)
                elif action_roll < MUTATE_ACTION_PROB_2:
                    new_block = generate_random_block(num_qubits)
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
                        num_qubits, random.randint(0, max_params - 1)
                    )
                    if random.random() < param_block_prob
                    else generate_random_gate(num_qubits)
                )
            elif action == "INSERT":
                item = (
                    generate_random_param_block(
                        num_qubits, random.randint(0, max_params - 1)
                    )
                    if random.random() < param_block_prob
                    else generate_random_gate(num_qubits)
                )
                individual.insert(i, item)
                i += 1
        i += 1
    return (individual,)


def describe_blocks(individual: EvolutionaryIndividual) -> List[dict]:
    blocks = []
    for position, gen in enumerate(individual):
        if gen[0] == "PARAM_BLOCK":
            param_idx, block_gates, tree = gen[1], gen[2], gen[3]
            blocks.append(
                {
                    "position": position,
                    "param_idx": param_idx,
                    "gate_count": len(block_gates),
                    "repetition_expression": str(tree),
                    "gates": [list(g) for g in block_gates],
                }
            )
    return blocks


def serialize_individual(individual: EvolutionaryIndividual) -> List[dict]:
    serialized = []
    for gen in individual:
        if gen[0] == "PARAM_BLOCK":
            param_idx, block_gates, tree = gen[1], gen[2], gen[3]
            serialized.append(
                {
                    "type": "PARAM_BLOCK",
                    "param_idx": param_idx,
                    "block_gates": [list(g) for g in block_gates],
                    "tree": str(tree),
                }
            )
        else:
            serialized.append({"type": "GATE", "gene": list(gen)})
    return serialized


def deserialize_individual(data: List[dict]) -> EvolutionaryIndividual:
    individual: EvolutionaryIndividual = []
    for item in data:
        if item["type"] == "PARAM_BLOCK":
            block_gates = [tuple(g) for g in item["block_gates"]]
            tree = gp.PrimitiveTree.from_string(item["tree"], pset)
            individual.append(("PARAM_BLOCK", item["param_idx"], block_gates, tree))
        else:
            individual.append(tuple(item["gene"]))
    return individual


def build_quantum_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    input_values: List[float] = [0.0],
    weight_values: dict = {},
    measure: bool = False,
) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    for gen in individual:
        if gen[0] == "PARAM_BLOCK":
            param_idx, block_gates, tree = gen[1], gen[2], gen[3]
            func = get_compiled_repetition_function(tree)

            in_val = (
                input_values[param_idx % len(input_values)] if input_values else 0.0
            )
            w_val = round(weight_values.get(param_idx, 0.0), 2)

            try:
                raw = func(in_val, w_val)
                reps = max(REPS_MIN, min(int(abs(raw)), REPS_MAX))
            except Exception:
                reps = REPS_MIN

            block_qc = _get_block_circuit(tuple(block_gates), num_qubits)
            for _ in range(reps):
                qc.compose(block_qc, inplace=True)
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
