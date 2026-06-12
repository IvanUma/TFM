import networkx as nx
import os
from networkx.algorithms.approximation.maxcut import one_exchange


def get_best_classical_cut(G):
    """
    Usa el algoritmo de optimización local 'one_exchange' de NetworkX.
    """
    cut_size, _ = one_exchange(G)
    return cut_size


def update_benchmarks(root_folder="max_cut_instances"):
    if not os.path.exists(root_folder):
        print(f"[ERROR] La carpeta '{root_folder}' no existe.")
        return

    current_file = os.path.abspath(__file__)

    for root, dirs, files in os.walk(root_folder):
        for filename in files:
            file_path = os.path.join(root, filename)

            if os.path.abspath(file_path) == current_file:
                continue

            with open(file_path, "r") as f:
                lines = f.readlines()

            if not lines:
                continue

            try:
                header = lines[0].split()
            except Exception:
                continue

            print(f"[*] Procesando: {os.path.relpath(file_path, root_folder)}")

            G = nx.Graph()
            try:
                for line in lines[1:]:
                    if line.strip():
                        parts = list(map(int, line.split()))
                        if len(parts) >= 2:
                            G.add_edge(parts[0] - 1, parts[1] - 1)

                if G.number_of_edges() == 0:
                    continue

                approx_cut = get_best_classical_cut(G)

                new_header = f"{header[0]} {header[1]} {approx_cut}\n"
                lines[0] = new_header

                with open(file_path, "w") as f:
                    f.writelines(lines)

                print(f"    -> Actualizado con corte: {approx_cut}")
            except Exception as e:
                print(f"    -> [ERROR] Fallo al procesar {filename}: {e}")


if __name__ == "__main__":
    update_benchmarks()
