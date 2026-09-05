"""At most eight named figures, drawn only from persisted payload.

Rendering never re-runs an experiment. A section without a persisted payload is skipped
rather than filled with a placeholder number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGURE_LIMIT = 8
FIGURE_NAMES: tuple[str, ...] = (
    "synthetic_head_winner_map_d4",
    "synthetic_head_winner_map_d8",
    "structure_effect_forest",
    "real_head_metric_comparison",
    "oracle_ladder",
    "teacher_pool_vs_student_recovery",
    "disagreement_future_failure",
    "branch_decision_tree",
)

FIGURE_SOURCE_KEYS: dict[str, str] = {
    "synthetic_head_winner_map_d4": "synthetic_winner_map_d4",
    "synthetic_head_winner_map_d8": "synthetic_winner_map_d8",
    "structure_effect_forest": "structure_contrasts",
    "real_head_metric_comparison": "real_teachers",
    "oracle_ladder": "real_oracle",
    "teacher_pool_vs_student_recovery": "distillation",
    "disagreement_future_failure": "failure_detection",
    "branch_decision_tree": "branches",
}


def _save(figure: plt.Figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def _winner_map(rows: Sequence[Mapping[str, Any]], title: str, path: Path) -> Path:
    figure, axes = plt.subplots(figsize=(5.2, 4.4))
    rho = [-0.8, 0.0, 0.8]
    lookup = {(float(row["rho_I"]), float(row["rho_M"])): row for row in rows}
    axes.set_xticks(range(3), [str(value) for value in rho])
    axes.set_yticks(range(3), [str(value) for value in rho])
    axes.set_xlabel("rho_I")
    axes.set_ylabel("rho_M")
    axes.set_title(title)
    for x, interval in enumerate(rho):
        for y, magnitude in enumerate(rho):
            row = lookup.get((interval, magnitude))
            label = "n/a" if row is None else f"{row.get('best_head','?')}\n{row.get('gap','')}"
            axes.text(x, y, label, ha="center", va="center", fontsize=8)
    axes.set_xlim(-0.5, 2.5)
    axes.set_ylim(-0.5, 2.5)
    axes.grid(True, alpha=0.3)
    return _save(figure, path)


def _bar(rows: Sequence[Mapping[str, Any]], title: str, path: Path, *, label: str, value: str) -> Path:
    figure, axes = plt.subplots(figsize=(5.6, 3.6))
    names = [str(row.get(label, index)) for index, row in enumerate(rows)]
    values = [float(row.get(value, 0.0)) for row in rows]
    axes.barh(names, values, color="#4C72B0")
    axes.axvline(0.0, color="#333333", linewidth=0.8)
    axes.set_title(title)
    axes.set_xlabel(value)
    return _save(figure, path)


def _decision_tree(rows: Sequence[Mapping[str, Any]], path: Path) -> Path:
    figure, axes = plt.subplots(figsize=(6.4, 3.8))
    axes.axis("off")
    axes.set_title("branch decision tree")
    for index, row in enumerate(rows):
        eligible = "confirmatory" if row.get("confirmatory_eligible") else "diagnostic"
        role = row.get("scientific_role") or "-"
        axes.text(
            0.02,
            0.92 - index * 0.16,
            f"{row.get('branch','?')}  [{eligible}]  {role}",
            fontsize=9,
            family="monospace",
        )
    return _save(figure, path)


def render_figures(payload: Mapping[str, Any], out_dir: Path) -> list[Path]:
    """Render every figure whose persisted payload exists, never exceeding the limit."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for name in FIGURE_NAMES:
        if len(written) >= FIGURE_LIMIT:
            break
        rows = payload.get(FIGURE_SOURCE_KEYS[name])
        if not rows:
            continue
        path = directory / f"{name}.png"
        if name.startswith("synthetic_head_winner_map"):
            written.append(_winner_map(rows, name.replace("_", " "), path))
        elif name == "branch_decision_tree":
            written.append(_decision_tree(rows, path))
        elif name == "structure_effect_forest":
            written.append(_bar(rows, "structure effect", path, label="contrast", value="effect"))
        elif name == "real_head_metric_comparison":
            written.append(_bar(rows, "real teacher sCRPS", path, label="head", value="sCRPS"))
        elif name == "oracle_ladder":
            written.append(_bar(rows, "oracle ladder", path, label="level", value="gain"))
        elif name == "teacher_pool_vs_student_recovery":
            written.append(_bar(rows, "student recovery", path, label="variant", value="recovery"))
        else:
            written.append(_bar(rows, "future failure detection", path, label="feature_set", value="AUPRC"))

    if len(written) > FIGURE_LIMIT:
        raise ValueError("the frozen figure limit was exceeded")
    return written


__all__ = ["FIGURE_LIMIT", "FIGURE_NAMES", "FIGURE_SOURCE_KEYS", "render_figures"]
