import networkx as nx
import numpy as np

from .utils import compute_qaoa_energy_analytically


def run_sanity_check():
    print("=== EXPLORACIÓN DE CONVENCIONES DEL MOTOR ANALÍTICO ===")

    g = nx.Graph()
    g.add_edge(0, 1, weight=1.0)

    combinaciones = [
        {"gamma": np.pi / 4, "beta": np.pi / 8, "nombre": "Caso 1 (pi/4, pi/8)"},
        {"gamma": 3 * np.pi / 4, "beta": np.pi / 8, "nombre": "Caso 2 (3pi/4, pi/8)"},
        {"gamma": np.pi / 4, "beta": 3 * np.pi / 8, "nombre": "Caso 3 (pi/4, 3pi/8)"},
        {
            "gamma": 3 * np.pi / 4,
            "beta": 3 * np.pi / 8,
            "nombre": "Caso 4 (3pi/4, 3pi/8)",
        },
    ]

    exito = False

    for combo in combinaciones:
        gammas = [combo["gamma"]]
        betas = [combo["beta"]]

        try:
            resultado = compute_qaoa_energy_analytically(g, gammas, betas)

            if abs(resultado) < 1e-10:
                resultado = 0.0

            print(f"[{combo['nombre']}] -> Valor calculado: {resultado:.6f}")

            if abs(resultado - 1.0) < 1e-5:
                print("--> ¡ÓPTIMO DETECTADO! El motor llegó a 1.0 puro.\n")
                exito = True

        except Exception as e:
            print(f"[ERROR] Falló en {combo['nombre']}: {e}\n")

    if exito:
        print(" CONCLUSIÓN: Motor validado. Convención de signos identificada.")
        print(
            "Ya puedes ejecutar tu 'main.py'. El algoritmo evolutivo se encargará solo de usar estos cuadrantes."
        )
    else:
        print(
            " ALERTA: Ninguna combinación dio 1.0. Revisa el manejo del término de identidad."
        )


if __name__ == "__main__":
    run_sanity_check()
