import random
import networkx as nx
from pathlib import Path
import pulp


def get_exact_max_cut(G: nx.Graph) -> int:
    prob = pulp.LpProblem("MaxCut", pulp.LpMaximize)

    y = {i: pulp.LpVariable(f"y_{i}", cat=pulp.LpBinary) for i in G.nodes()}

    x = {e: pulp.LpVariable(f"x_{e[0]}_{e[1]}", cat=pulp.LpBinary) for e in G.edges()}

    prob += pulp.lpSum(G[u][v].get("weight", 1.0) * x[(u, v)] for u, v in G.edges())

    for u, v in G.edges():
        prob += x[(u, v)] <= y[u] + y[v]
        prob += x[(u, v)] <= 2 - (y[u] + y[v])

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    return int(pulp.value(prob.objective))


def generate_maxcut_file(
    num_nodes: int, edge_probability: float, output_path: Path, weight_range=(1, 1)
):
    G = nx.erdos_renyi_graph(n=num_nodes, p=edge_probability)

    if not nx.is_connected(G) and num_nodes > 1:
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = random.choice(list(components[i]))
            v = random.choice(list(components[i + 1]))
            G.add_edge(u, v)

    for u, v in G.edges():
        G[u][v]["weight"] = random.randint(weight_range[0], weight_range[1])

    print(f"Calculando MaxCut óptimo para {num_nodes} nodos ({output_path.name})...")
    optimal_cut = get_exact_max_cut(G)
    num_edges = G.number_of_edges()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"{num_nodes} {num_edges} {optimal_cut}\n")
        for u, v, data in G.edges(data=True):
            f.write(f"{u + 1} {v + 1} {int(data['weight'])}\n")

    print(
        f"[ÉXITO] Nodos: {num_nodes} | Aristas: {num_edges} | Corte Óptimo: {optimal_cut}\n"
    )


if __name__ == "__main__":
    instances_dir = Path(__file__).parent / "max_cut_instances"

    qubit_groups = [(12, 0.35)]

    instances_per_group = 15

    for nodes, p in qubit_groups:
        for _ in range(instances_per_group):
            random_id = random.randint(1000, 9999)
            filename = f"instance_{nodes}nodes_{random_id}.txt"

            generate_maxcut_file(
                num_nodes=nodes,
                edge_probability=p,
                output_path=instances_dir / filename,
                weight_range=(1, 5),
            )
