from __future__ import annotations

import math
import random
from typing import List, Tuple, Union

import networkx as nx
from qiskit import QuantumCircuit

import max_cut_common as common

QuantumGen = Union[
    Tuple[str, int, float],
    Tuple[str, int, int],
    Tuple[str, int, int, float],
]
EvolutionaryIndividual = List[QuantumGen]

load_external_maxcut_instance = common.load_external_maxcut_instance


def generate_random_gate(num_qubits: int, graph_instance: nx.Graph) -> QuantumGen:
    if num_qubits < 2:
        return ("RY", 0, random.uniform(0.0, 2 * math.pi))

    gate = random.choice(["RZZ", "CX"])
    if graph_instance.number_of_edges() > 0:
        u, v = random.choice(list(graph_instance.edges()))
        if random.random() > 0.5:
            u, v = v, u
        if gate == "CX":
            return ("CX", u, v)
        return ("RZZ", u, v, random.uniform(0.0, 2 * math.pi))

    return ("CX", 0, min(1, num_qubits - 1))


def generate_suffix(num_qubits: int) -> EvolutionaryIndividual:
    suffix = []
    for q in range(num_qubits):
        suffix.append(("RX", q, random.uniform(0.0, 2 * math.pi)))
        suffix.append(("RY", q, random.uniform(0.0, 2 * math.pi)))
    return suffix


def generate_guided_individual(
    num_qubits: int,
    length: int,
    graph_instance: nx.Graph,
) -> EvolutionaryIndividual:
    individual: EvolutionaryIndividual = [
        ("RY", q, math.pi / 2.0) for q in range(num_qubits)
    ]
    for _ in range(length):
        individual.append(generate_random_gate(num_qubits, graph_instance))
    individual.extend(generate_suffix(num_qubits))
    return individual


def generate_heuristic_individual(
    num_qubits: int,
    graph_instance: nx.Graph,
) -> EvolutionaryIndividual:
    individual: EvolutionaryIndividual = [
        ("RY", q, math.pi / 2.0) for q in range(num_qubits)
    ]
    if graph_instance.number_of_nodes() == 0:
        individual.extend(generate_suffix(num_qubits))
        return individual

    added_edges = set()
    start_node = random.choice(list(graph_instance.nodes()))

    for u, v in nx.dfs_edges(graph_instance, source=start_node):
        individual.append(("RZZ", u, v, random.uniform(0.0, math.pi)))
        added_edges.add(tuple(sorted((u, v))))

    for u, v in graph_instance.edges():
        edge = tuple(sorted((u, v)))
        if edge not in added_edges:
            individual.append(("CX", u, v))
            added_edges.add(edge)

    individual.extend(generate_suffix(num_qubits))
    return individual


def cx_quantum_circuit(
    ind1: EvolutionaryIndividual,
    ind2: EvolutionaryIndividual,
    num_qubits: int,
) -> Tuple[EvolutionaryIndividual, EvolutionaryIndividual]:
    s_len = num_qubits * 2
    if s_len == 0:
        return ind1, ind2

    mid1 = ind1[num_qubits:-s_len]
    suf1 = ind1[-s_len:]
    mid2 = ind2[num_qubits:-s_len]
    suf2 = ind2[-s_len:]

    size = min(len(mid1), len(mid2))
    if size >= 2:
        cxpoint1 = random.randint(1, size)
        cxpoint2 = random.randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        mid1[cxpoint1:cxpoint2], mid2[cxpoint1:cxpoint2] = (
            mid2[cxpoint1:cxpoint2],
            mid1[cxpoint1:cxpoint2],
        )

    for i in range(min(len(mid1), len(mid2))):
        if random.random() < 0.3:
            g1, g2 = mid1[i], mid2[i]
            if g1[0] == "RZZ" and g2[0] == "RZZ" and g1[1] == g2[1] and g1[2] == g2[2]:
                avg_angle = (g1[3] + g2[3]) / 2.0
                mid1[i] = (g1[0], g1[1], g1[2], avg_angle)
                mid2[i] = (g2[0], g2[1], g2[2], avg_angle)

    for i in range(s_len):
        if random.random() < 0.3:
            g1, g2 = suf1[i], suf2[i]
            avg_angle = (g1[2] + g2[2]) / 2.0
            suf1[i] = (g1[0], g1[1], avg_angle)
            suf2[i] = (g2[0], g2[1], avg_angle)

    ind1[:] = ind1[:num_qubits] + mid1 + suf1
    ind2[:] = ind2[:num_qubits] + mid2 + suf2
    return ind1, ind2


def mut_quantum_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    graph_instance: nx.Graph,
    indpb: float,
) -> Tuple[EvolutionaryIndividual]:
    s_len = num_qubits * 2
    prefix = individual[:num_qubits]
    mid = individual[num_qubits:-s_len] if s_len else individual[num_qubits:]
    suf = individual[-s_len:] if s_len else []

    i = 0
    while i < len(mid):
        if random.random() < indpb:
            action = random.choice(["INSERT", "DELETE", "REPLACE"])
            if action == "DELETE" and len(mid) > 1:
                mid.pop(i)
                continue
            if action == "REPLACE":
                mid[i] = generate_random_gate(num_qubits, graph_instance)
            if action == "INSERT":
                mid.insert(i, generate_random_gate(num_qubits, graph_instance))
                i += 1
        i += 1

    for j in range(len(mid)):
        if random.random() < indpb:
            gen = mid[j]
            if gen[0] == "RZZ":
                new_angle = (gen[3] + random.gauss(0.0, 0.5)) % (2 * math.pi)
                mid[j] = (gen[0], gen[1], gen[2], new_angle)

    for j in range(len(suf)):
        if random.random() < indpb:
            gen = suf[j]
            new_angle = (gen[2] + random.gauss(0.0, 0.5)) % (2 * math.pi)
            suf[j] = (gen[0], gen[1], new_angle)

    individual[:] = prefix + mid + suf
    return (individual,)


def gates_commute(g1: QuantumGen, g2: QuantumGen) -> bool:
    q1 = {g1[1]} if g1[0] in ["RX", "RY"] else {g1[1], g1[2]}
    q2 = {g2[1]} if g2[0] in ["RX", "RY"] else {g2[1], g2[2]}
    if q1.isdisjoint(q2):
        return True
    if g1[0] == "RZZ" and g2[0] == "RZZ":
        return True
    if g1[0] == "CX" and g2[0] == "CX" and g1[1] == g2[1]:
        return True
    return False


def simplify_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
) -> EvolutionaryIndividual:
    s_len = num_qubits * 2
    prefix = individual[:num_qubits]
    mid = individual[num_qubits:-s_len] if s_len else individual[num_qubits:]
    suf = individual[-s_len:] if s_len else []
    simplified_mid: EvolutionaryIndividual = []

    for gate in mid:
        merged = False
        for j in range(len(simplified_mid) - 1, -1, -1):
            prev = simplified_mid[j]
            if gate[0] == prev[0] and gate[1] == prev[1] and gate[2] == prev[2]:
                if gate[0] == "RZZ":
                    new_angle = (gate[3] + prev[3]) % (2 * math.pi)
                    if new_angle < 1e-5 or abs(new_angle - 2 * math.pi) < 1e-5:
                        simplified_mid.pop(j)
                    else:
                        simplified_mid[j] = (gate[0], gate[1], gate[2], new_angle)
                    merged = True
                    break
                if gate[0] == "CX":
                    simplified_mid.pop(j)
                    merged = True
                    break
            if not gates_commute(gate, prev):
                break
        if not merged:
            simplified_mid.append(gate)

    return prefix + simplified_mid + suf


def build_quantum_circuit(
    individual: EvolutionaryIndividual,
    num_qubits: int,
    measure: bool = False,
) -> QuantumCircuit:
    qc: QuantumCircuit = QuantumCircuit(num_qubits)
    for gen in individual:
        gate_type = gen[0]
        if gate_type == "RY":
            qc.ry(gen[2], gen[1])
        elif gate_type == "RX":
            qc.rx(gen[2], gen[1])
        elif gate_type == "RZZ":
            qc.rzz(gen[3], gen[1], gen[2])
        elif gate_type == "CX":
            qc.cx(gen[1], gen[2])
    if measure:
        qc.measure_all()
    return qc


def get_cache_key(num_qubits: int, individual: EvolutionaryIndividual) -> tuple:
    rounded = []
    for gen in individual:
        if gen[0] in ["RX", "RY"]:
            rounded.append((gen[0], gen[1], round(gen[2], 3)))
        elif gen[0] == "RZZ":
            rounded.append((gen[0], gen[1], gen[2], round(gen[3], 3)))
        else:
            rounded.append(gen)
    return (num_qubits, tuple(rounded))
