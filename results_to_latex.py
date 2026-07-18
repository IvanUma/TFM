from __future__ import annotations

import argparse
import json
from pathlib import Path


def escape_latex(text: str) -> str:
    replacements = {
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
    }
    for char, escaped in replacements.items():
        text = text.replace(char, escaped)
    return text


def load_result(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_domain(data: dict) -> str:
    if "per_instance_validation" in data.get("results", {}):
        return "maxcut"
    if "dataset" in data.get("config", {}):
        return "qnn"
    return "unknown"


def _speedup(val_a: float, val_b: float, higher_is_better: bool = False) -> str:
    if val_b <= 0:
        return "--"
    ratio = val_a / val_b
    return f"{ratio:.2f}x"


def comparison_table(clifford_data: dict, rotation_data: dict) -> str:
    domain = detect_domain(clifford_data)
    c_res = clifford_data["results"]
    p_res = rotation_data["results"]
    c_timing = clifford_data["timing"]
    p_timing = rotation_data["timing"]

    if domain == "maxcut":
        quality_rows = [
            (r"Approx.\ ratio medio (valid.)",
             f"{c_res['best_validation_avg_approx_ratio']:.6f}",
             f"{p_res['best_validation_avg_approx_ratio']:.6f}",
             _speedup(p_res['best_validation_avg_approx_ratio'],
                      c_res['best_validation_avg_approx_ratio'], higher_is_better=True)),
            (r"Profundidad mejor individuo",
             str(c_res['best_individual_depth']),
             str(p_res['best_individual_depth']),
             "--"),
        ]
        label_suffix = "maxcut"
        caption = "Comparación Clifford vs. Rotación en MaxCut"
    elif domain == "qnn":
        quality_rows = [
            (r"Precisión validación (final)",
             f"{c_res['final_validation_accuracy']:.4f}",
             f"{p_res['final_validation_accuracy']:.4f}",
             _speedup(p_res['final_validation_accuracy'],
                      c_res['final_validation_accuracy'], higher_is_better=True)),
            (r"Precisión test (final)",
             f"{c_res['final_test_accuracy']:.4f}",
             f"{p_res['final_test_accuracy']:.4f}",
             _speedup(p_res['final_test_accuracy'],
                      c_res['final_test_accuracy'], higher_is_better=True)),
            (r"Profundidad mejor individuo",
             str(c_res['best_individual_depth']),
             str(p_res['best_individual_depth']),
             "--"),
        ]
        label_suffix = "qnn"
        caption = "Comparación Clifford vs. Rotación en QNN"
    else:
        quality_rows = []
        label_suffix = "unknown"
        caption = "Comparación Clifford vs. Rotación"

    timing_rows = [
        (r"Total (simulación pura)",
         f"{c_timing['total_simulation_seconds']:.4f}",
         f"{p_timing['total_simulation_seconds']:.4f}",
         _speedup(p_timing['total_simulation_seconds'], c_timing['total_simulation_seconds'])),
        (r"Total (reloj)",
         f"{c_timing['total_wall_seconds']:.4f}",
         f"{p_timing['total_wall_seconds']:.4f}",
         _speedup(p_timing['total_wall_seconds'], c_timing['total_wall_seconds'])),
        (r"Media / generación (sim.)",
         f"{c_timing['avg_simulation_seconds_per_generation']:.4f}",
         f"{p_timing['avg_simulation_seconds_per_generation']:.4f}",
         _speedup(p_timing['avg_simulation_seconds_per_generation'],
                  c_timing['avg_simulation_seconds_per_generation'])),
    ]

    rows_buf = []
    rows_buf.append(r"\midrule")
    rows_buf.append(r"\multicolumn{4}{l}{\textit{Calidad de la Solución}} \\")
    rows_buf.append(r"\midrule")
    for metric, c_val, p_val, ratio in quality_rows:
        rows_buf.append(f"{metric} & {c_val} & {p_val} & {ratio} \\\\")

    rows_buf.append(r"\midrule")
    rows_buf.append(r"\multicolumn{4}{l}{\textit{Tiempos de Ejecución (s)}} \\")
    rows_buf.append(r"\midrule")
    for metric, c_val, p_val, ratio in timing_rows:
        rows_buf.append(f"{metric} & {c_val} & {p_val} & {ratio} \\\\")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Métrica} & \textbf{Clifford} & \textbf{Rotación} & \textbf{Ratio (Rot./Clif.)} \\",
        *rows_buf,
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{{caption}.}}",
        f"\\label{{tab:comparison-{label_suffix}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def history_table(clifford_data: dict, rotation_data: dict, step: int = 15) -> str:
    domain = detect_domain(clifford_data)
    c_hist = clifford_data["history"]
    p_hist = rotation_data["history"]

    total_gens = len(c_hist["generation"])
    indices = list(range(0, total_gens, step))
    if indices[-1] != total_gens - 1:
        indices.append(total_gens - 1)

    if domain == "maxcut":
        c_metric_key = "best_avg_ar"
        p_metric_key = "best_avg_ar"
        metric_label = "Approx. Ratio"
        label_suffix = "maxcut"
        caption = (
            f"Evolución en MaxCut: calidad de la solución y "
            f"tiempo de simulación (muestreado cada {step} generaciones)"
        )
    elif domain == "qnn":
        c_metric_key = "best_accuracy"
        p_metric_key = "best_accuracy"
        metric_label = "Precisión (valid.)"
        label_suffix = "qnn"
        caption = (
            f"Evolución en QNN: precisión de validación y "
            f"tiempo de simulación (muestreado cada {step} generaciones)"
        )
    else:
        c_metric_key = "best_avg_ar"
        p_metric_key = "best_avg_ar"
        metric_label = "Métrica"
        label_suffix = "unknown"
        caption = f"Evolución (muestreado cada {step} generaciones)"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{c | cc | cc}",
        r"\toprule",
        r"& \multicolumn{2}{c|}{\textbf{Clifford}} & \multicolumn{2}{c}{\textbf{Rotación}} \\",
        rf"\textbf{{Generación}} & \textbf{{{metric_label}}} & \textbf{{Tiempo Sim. (s)}} & \textbf{{{metric_label}}} & \textbf{{Tiempo Sim. (s)}} \\",
        r"\midrule",
    ]

    for i in indices:
        gen = c_hist["generation"][i]
        c_metric = c_hist[c_metric_key][i]
        c_sim = c_hist["simulation_seconds"][i]
        p_metric = p_hist[p_metric_key][i]
        p_sim = p_hist["simulation_seconds"][i]

        lines.append(
            f"{gen} & {c_metric:.4f} & {c_sim:.4f} & {p_metric:.4f} & {p_sim:.4f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{tab:history-{label_suffix}}}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def build_document(
    clifford_path: Path | None,
    rotation_path: Path | None,
) -> str:
    sections = []

    clifford_data = load_result(clifford_path) if clifford_path else None
    rotation_data = load_result(rotation_path) if rotation_path else None

    if clifford_data and rotation_data:
        sections.append(comparison_table(clifford_data, rotation_data))
        sections.append(history_table(clifford_data, rotation_data, step=15))

    return "\n\n".join(section for section in sections if section)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convierte JSONs de resultados (MaxCut/QNN) a tablas LaTeX comparativas."
    )
    parser.add_argument(
        "--clifford",
        type=Path,
        default=None,
        help="JSON de resultados del enfoque Clifford",
    )
    parser.add_argument(
        "--rotation",
        type=Path,
        default=None,
        help="JSON de resultados del enfoque Rotación",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Fichero .tex de salida"
    )
    args = parser.parse_args()

    if args.clifford is None or args.rotation is None:
        parser.error("Debes indicar --clifford y --rotation")

    document = build_document(args.clifford, args.rotation)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(document)
        f.write("\n")

    print(f"LaTeX generado en {args.output}")


if __name__ == "__main__":
    main()
