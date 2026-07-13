from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


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


def config_summary_table(data: dict, label_suffix: str) -> str:
    cfg = data["config"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Parámetro & Valor \\",
        r"\midrule",
        f"Enfoque & {escape_latex(cfg['approach'])} \\\\",
        f"Qubits del circuito & {cfg['circuit_qubits']} \\\\",
        f"Backend de simulación & {escape_latex(cfg['simulator_method'])} "
        f"({escape_latex(cfg['simulator_device'])}) \\\\",
        f"Instancias entrenamiento & {cfg['training_instances']} \\\\",
        f"Instancias validación & {cfg['validation_instances']} \\\\",
        f"Generaciones (configuradas / ejecutadas) & {cfg['generations_configured']} / "
        f"{cfg['generations_run']} \\\\",
        f"$\\mu$ / $\\lambda$ & {cfg['mu']} / {cfg['lambda']} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{Configuración de la ejecución ({escape_latex(cfg['approach'])})}}",
        f"\\label{{tab:config-{label_suffix}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def results_table(data: dict, label_suffix: str) -> str:
    res = data["results"]
    timing = data["timing"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{ll}",
        r"\toprule",
        r"Métrica & Valor \\",
        r"\midrule",
        f"Approx.\\ ratio medio (validación) & {res['best_validation_avg_approx_ratio']:.6f} \\\\",
        f"Profundidad del mejor individuo & {res['best_individual_depth']} \\\\",
        f"Tiempo total de simulación (s) & {timing['total_simulation_seconds']:.4f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{Resultados de validación ({escape_latex(data['config']['approach'])})}}",
        f"\\label{{tab:results-{label_suffix}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def per_instance_table(data: dict, label_suffix: str) -> str:
    per_instance = data["results"].get("per_instance_validation", [])
    if not per_instance:
        return ""

    rows = "\n".join(
        f"{escape_latex(item['instance'])} & {item['approx_ratio']:.6f} \\\\"
        for item in per_instance
    )
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Instancia & Approx.\ ratio \\",
        r"\midrule",
        rows,
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{Approx.\\ ratio por instancia de validación ({escape_latex(data['config']['approach'])})}}",
        f"\\label{{tab:per-instance-{label_suffix}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def timing_table(data: dict, label_suffix: str) -> str:
    timing = data["timing"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"Métrica de tiempo & Segundos \\",
        r"\midrule",
        f"Total (reloj) & {timing['total_wall_seconds']:.4f} \\\\",
        f"Total (CPU) & {timing['total_cpu_seconds']:.4f} \\\\",
        f"Total (simulación pura) & {timing['total_simulation_seconds']:.4f} \\\\",
        f"Media por generación (reloj) & {timing['avg_wall_seconds_per_generation']:.4f} \\\\",
        f"Media por generación (CPU) & {timing['avg_cpu_seconds_per_generation']:.4f} \\\\",
        f"Media por generación (simulación pura) & {timing['avg_simulation_seconds_per_generation']:.4f} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{Tiempos de ejecución ({escape_latex(data['config']['approach'])})}}",
        f"\\label{{tab:timing-{label_suffix}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def comparison_table(clifford_data: dict, parametric_data: dict) -> str:
    c_timing = clifford_data["timing"]
    p_timing = parametric_data["timing"]
    c_res = clifford_data["results"]
    p_res = parametric_data["results"]
    c_cfg = clifford_data["config"]
    p_cfg = parametric_data["config"]

    def speedup(parametric_value: float, clifford_value: float) -> str:
        if clifford_value <= 0:
            return "--"
        return f"{parametric_value / clifford_value:.2f}x"

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Métrica} & \textbf{Clifford} & \textbf{Rotación (Paramétrico)} & \textbf{Ratio (Rot./Clif.)} \\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Configuración}} \\",
        r"\midrule",
        f"Backend de simulación & {escape_latex(c_cfg['simulator_method'])} & {escape_latex(p_cfg['simulator_method'])} & -- \\\\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Calidad de la Solución}} \\",
        r"\midrule",
        f"Approx.\\ ratio medio (valid.) & {c_res['best_validation_avg_approx_ratio']:.6f} & {p_res['best_validation_avg_approx_ratio']:.6f} & {speedup(p_res['best_validation_avg_approx_ratio'], c_res['best_validation_avg_approx_ratio'])} \\\\",
        f"Profundidad mejor individuo & {c_res['best_individual_depth']} & {p_res['best_individual_depth']} & -- \\\\",
        r"\midrule",
        r"\multicolumn{4}{l}{\textit{Tiempos de Ejecución (s)}} \\",
        r"\midrule",
        f"Total (simulación pura) & {c_timing['total_simulation_seconds']:.4f} & {p_timing['total_simulation_seconds']:.4f} & {speedup(p_timing['total_simulation_seconds'], c_timing['total_simulation_seconds'])} \\\\",
        f"Total (reloj) & {c_timing['total_wall_seconds']:.4f} & {p_timing['total_wall_seconds']:.4f} & {speedup(p_timing['total_wall_seconds'], c_timing['total_wall_seconds'])} \\\\",
        f"Total (CPU) & {c_timing['total_cpu_seconds']:.4f} & {p_timing['total_cpu_seconds']:.4f} & {speedup(p_timing['total_cpu_seconds'], c_timing['total_cpu_seconds'])} \\\\",
        f"Media / generación (sim.) & {c_timing['avg_simulation_seconds_per_generation']:.4f} & {p_timing['avg_simulation_seconds_per_generation']:.4f} & {speedup(p_timing['avg_simulation_seconds_per_generation'], c_timing['avg_simulation_seconds_per_generation'])} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Comparación detallada Clifford vs. Paramétrico: Configuración, Calidad y Rendimiento.}",
        r"\label{tab:comparison-detailed}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def history_table(clifford_data: dict, parametric_data: dict, step: int = 15) -> str:
    c_hist = clifford_data["history"]
    p_hist = parametric_data["history"]

    total_gens = len(c_hist["generation"])
    indices = list(range(0, total_gens, step))
    if indices[-1] != total_gens - 1:
        indices.append(total_gens - 1)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{c | cc | cc}",
        r"\toprule",
        r"& \multicolumn{2}{c|}{\textbf{Clifford}} & \multicolumn{2}{c}{\textbf{Rotación (Paramétrico)}} \\",
        r"\textbf{Generación} & \textbf{Approx. Ratio} & \textbf{Tiempo Sim. (s)} & \textbf{Approx. Ratio} & \textbf{Tiempo Sim. (s)} \\",
        r"\midrule",
    ]

    for i in indices:
        gen = c_hist["generation"][i]
        c_ar = c_hist["best_avg_ar"][i]
        c_sim = c_hist["simulation_seconds"][i]
        p_ar = p_hist["best_avg_ar"][i]
        p_sim = p_hist["simulation_seconds"][i]

        lines.append(
            f"{gen} & {c_ar:.4f} & {c_sim:.4f} & {p_ar:.4f} & {p_sim:.4f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            f"\\caption{{Evolución del algoritmo: Calidad de la solución y tiempo de simulación (muestreado cada {step} generaciones)}}",
            r"\label{tab:history-evolution}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def build_document(
    clifford_path: Optional[Path],
    parametric_path: Optional[Path],
) -> str:
    sections = []

    clifford_data = load_result(clifford_path) if clifford_path else None
    parametric_data = load_result(parametric_path) if parametric_path else None

    if clifford_data:
        sections.append(config_summary_table(clifford_data, "clifford"))
        sections.append(results_table(clifford_data, "clifford"))
        sections.append(per_instance_table(clifford_data, "clifford"))
        sections.append(timing_table(clifford_data, "clifford"))

    if parametric_data:
        sections.append(config_summary_table(parametric_data, "parametric"))
        sections.append(results_table(parametric_data, "parametric"))
        sections.append(per_instance_table(parametric_data, "parametric"))
        sections.append(timing_table(parametric_data, "parametric"))

    if clifford_data and parametric_data:
        sections.append(comparison_table(clifford_data, parametric_data))
        sections.append(history_table(clifford_data, parametric_data, step=15))

    return "\n\n".join(section for section in sections if section)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serializa el JSON de resultados de main.py a tablas LaTeX para la memoria."
    )
    parser.add_argument(
        "--clifford",
        type=Path,
        default=None,
        help="JSON de resultados del enfoque Clifford",
    )
    parser.add_argument(
        "--parametric",
        type=Path,
        default=None,
        help="JSON de resultados del enfoque paramétrico",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Fichero .tex de salida"
    )
    args = parser.parse_args()

    if args.clifford is None and args.parametric is None:
        parser.error("Debes indicar al menos --clifford o --parametric")

    document = build_document(args.clifford, args.parametric)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(document)
        f.write("\n")

    print(f"LaTeX generado en {args.output}")


if __name__ == "__main__":
    main()
