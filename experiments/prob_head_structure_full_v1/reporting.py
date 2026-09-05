"""Tables A-T, the STATUS document, and the frozen console summary.

Everything here reads persisted stage payloads. Reporting never fits a model, never
re-runs an experiment, and never promotes a diagnostic result to confirmatory.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

TABLE_IDS: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRST")

TABLE_TITLES: dict[str, str] = {
    "A": "repository / environment / source audit",
    "B": "dataset support and split",
    "C": "distribution numerical validation",
    "D": "teacher parameter count and runtime",
    "E": "synthetic 18-cell sCRPS",
    "F": "synthetic winner share and oracle gain",
    "G": "temporal structure contrasts",
    "H": "synthetic change-sensor result",
    "I": "real teacher benchmark",
    "J": "real winner share and oracle ladder",
    "K": "CDF pool result",
    "L": "distillation result and recovery",
    "M": "student compression/latency",
    "N": "structure-regret predictability",
    "O": "structure-conditioned distillation",
    "P": "disagreement future-failure detection",
    "Q": "actionable sensor policy",
    "R": "negative controls",
    "S": "every gate",
    "T": "final recommended research axis",
}

TABLE_SOURCE_KEYS: dict[str, str] = {
    "A": "audit",
    "B": "dataset_support",
    "C": "likelihood_validation",
    "D": "teacher_runtime",
    "E": "synthetic_cells",
    "F": "synthetic_oracle",
    "G": "structure_contrasts",
    "H": "change_sensor",
    "I": "real_teachers",
    "J": "real_oracle",
    "K": "cdf_pool",
    "L": "distillation",
    "M": "compression",
    "N": "regret_predictability",
    "O": "structure_conditioned",
    "P": "failure_detection",
    "Q": "action_policy",
    "R": "controls",
    "S": "gates",
    "T": "recommendation_axis",
}

STATUS_FIRST_LINE_PREFIX = "FINAL RECOMMENDATION:"
STATUS_PATH = "results/prob_head_structure_full_v1/STATUS.md"

STATUS_SECTIONS: tuple[str, ...] = (
    "What was attempted",
    "What was frozen",
    "Runtime tier",
    "Environment",
    "Dataset support",
    "Numerical likelihood validation",
    "Synthetic specialization",
    "Temporal structure effect",
    "Real teacher quality",
    "Real complementarity",
    "CDF pooling",
    "A distillation",
    "B structure-conditioned routing",
    "C disagreement sensor",
    "Controls",
    "Compression/runtime",
    "Confirmatory vs diagnostic evidence",
    "Gate table",
    "Final recommendation",
    "What must not be claimed",
    "Exact next research action",
)

CONSOLE_ORDER: tuple[str, ...] = (
    "FINAL RECOMMENDATION",
    "A verdict",
    "B verdict",
    "C verdict",
    "HEAD specialization verdict",
    "synthetic core effect",
    "real oracle headroom",
    "best teacher pool gain",
    "student recovery",
    "disagreement sensor AUPRC",
    "total GPU/wall time",
    "scientific FAIL list",
    "diagnostic continuation list",
    "commit SHA",
    "push status",
    "STATUS path",
)

FORBIDDEN_CLAIMS: tuple[str, ...] = (
    "M5 and the other development datasets are not external confirmation",
    "a diagnostic continuation is not confirmatory evidence",
    "raw NLL cannot rank different distribution families",
    "Tweedie deviance is not the full density",
    "no continuous target was rounded into a count likelihood",
    "a teacher pool advantage is not by itself a distillation success",
    "a disagreement correlation is not a distribution-shift cause",
    "a synthetic structure effect is not a real-data causal effect",
    "one seed is not a general effect",
)

NEXT_ACTION_BY_RECOMMENDATION: dict[str, str] = {
    "RECOMMEND_A_DISTRIBUTION_DISTILLATION": (
        "design an untouched external confirmatory study for distribution-space distillation"
    ),
    "RECOMMEND_C_DISAGREEMENT_SENSOR": (
        "design an untouched external confirmatory study for the disagreement sensor"
    ),
    "RECOMMEND_B_STRUCTURE_CONDITIONED_ROUTING": (
        "design an untouched external confirmatory study for structure-conditioned routing"
    ),
    "RECOMMEND_CHARACTERIZATION_ONLY": (
        "write up the characterization result and stop new method development on this axis"
    ),
    "ALL_NEW_METHOD_BRANCHES_NO_GO": (
        "stop method development on this axis and reconsider the expert set"
    ),
    "INTEGRITY_BLOCKED_NO_SCIENTIFIC_VERDICT": (
        "repair the integrity failure and re-run before drawing any scientific conclusion"
    ),
}


def build_tables(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Assemble every declared table from persisted payload sections."""
    tables: dict[str, dict[str, Any]] = {}
    for identifier in TABLE_IDS:
        source = TABLE_SOURCE_KEYS[identifier]
        rows = payload.get(source)
        if rows is None:
            rows, note = [], "NOT_PRODUCED"
        elif isinstance(rows, Mapping):
            rows, note = [dict(rows)], "from persisted payload"
        elif isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            rows, note = [dict(row) if isinstance(row, Mapping) else {"value": row} for row in rows], (
                "from persisted payload"
            )
        else:
            rows, note = [{"value": rows}], "from persisted payload"
        tables[identifier] = {
            "id": identifier,
            "title": TABLE_TITLES[identifier],
            "source_key": source,
            "rows": rows,
            "note": note,
        }
    return tables


def _render_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "```\nNOT_PRODUCED\n```"
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    divider = "─" * len(header)
    body = [
        "  ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns)
        for row in rows
    ]
    return "```\n" + "\n".join([header, divider, *body]) + "\n```"


def render_status(
    payload: Mapping[str, Any], tables: Mapping[str, Mapping[str, Any]]
) -> str:
    """Render STATUS.md with the exact first line and the frozen section order."""
    recommendation = str(payload.get("recommendation", "INTEGRITY_BLOCKED_NO_SCIENTIFIC_VERDICT"))
    lines: list[str] = [f"{STATUS_FIRST_LINE_PREFIX} {recommendation}", ""]

    section_body: dict[str, list[str]] = {name: [] for name in STATUS_SECTIONS}
    section_body["What was attempted"] = [
        "PROB-HEAD-STRUCTURE-FULL-v1 screened whether NB, hurdle-shifted NB and full Tweedie "
        "heads specialize by temporal structure, and whether that supports distribution-space "
        "distillation (A), structure-conditioned routing (B) or a disagreement sensor (C)."
    ]
    section_body["What was frozen"] = [
        "The authoritative preregistration, its companion hash, the protected-artifact baseline, "
        "every gate threshold, every seed and the runtime-tier rule were frozen before any fit.",
        _render_rows(tables["A"]["rows"]),
    ]
    section_body["Runtime tier"] = [f"tier: {payload.get('runtime_tier', 'NOT_SELECTED')}"]
    section_body["Environment"] = [_render_rows([dict(payload.get("environment", {}))])]
    section_body["Dataset support"] = [_render_rows(tables["B"]["rows"])]
    section_body["Numerical likelihood validation"] = [_render_rows(tables["C"]["rows"])]
    section_body["Synthetic specialization"] = [
        _render_rows(tables["E"]["rows"]),
        _render_rows(tables["F"]["rows"]),
    ]
    section_body["Temporal structure effect"] = [_render_rows(tables["G"]["rows"])]
    section_body["Real teacher quality"] = [
        _render_rows(tables["D"]["rows"]),
        _render_rows(tables["I"]["rows"]),
    ]
    section_body["Real complementarity"] = [_render_rows(tables["J"]["rows"])]
    section_body["CDF pooling"] = [_render_rows(tables["K"]["rows"])]
    section_body["A distillation"] = [_render_rows(tables["L"]["rows"])]
    section_body["B structure-conditioned routing"] = [
        _render_rows(tables["N"]["rows"]),
        _render_rows(tables["O"]["rows"]),
    ]
    section_body["C disagreement sensor"] = [
        _render_rows(tables["H"]["rows"]),
        _render_rows(tables["P"]["rows"]),
        _render_rows(tables["Q"]["rows"]),
    ]
    section_body["Controls"] = [_render_rows(tables["R"]["rows"])]
    section_body["Compression/runtime"] = [_render_rows(tables["M"]["rows"])]
    section_body["Confirmatory vs diagnostic evidence"] = [
        _render_rows(list(payload.get("branches", [])))
    ]
    section_body["Gate table"] = [_render_rows(tables["S"]["rows"])]
    section_body["Final recommendation"] = [
        f"[판정] {recommendation}",
        _render_rows(tables["T"]["rows"]),
    ]
    section_body["What must not be claimed"] = [
        "\n".join(f"- {claim}" for claim in FORBIDDEN_CLAIMS)
    ]
    section_body["Exact next research action"] = [
        NEXT_ACTION_BY_RECOMMENDATION.get(recommendation, "resolve the recommendation token first")
    ]

    for index, name in enumerate(STATUS_SECTIONS, 1):
        lines.append(f"## {index}. {name}")
        lines.append("")
        body = [item for item in section_body[name] if item]
        lines.extend(body or ["NOT_PRODUCED"])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def console_summary(payload: Mapping[str, Any]) -> list[str]:
    """The sixteen frozen console lines, in order, with no log noise before them."""
    branches = {item["branch"]: item for item in payload.get("branches", [])}
    observations = dict(payload.get("observations", {}))
    runtime = dict(payload.get("runtime", {}))
    verdicts = dict(payload.get("verdicts", {}))
    diagnostic = [
        item["scientific_role"]
        for item in payload.get("branches", [])
        if item.get("scientific_role")
    ]
    values = {
        "FINAL RECOMMENDATION": payload.get("recommendation", "NOT_PRODUCED"),
        "A verdict": verdicts.get("A", "NOT_PRODUCED"),
        "B verdict": verdicts.get("B", "NOT_PRODUCED"),
        "C verdict": verdicts.get("C", "NOT_PRODUCED"),
        "HEAD specialization verdict": verdicts.get("HEAD", "NOT_PRODUCED"),
        "synthetic core effect": observations.get("synthetic_effect", "NOT_PRODUCED"),
        "real oracle headroom": observations.get("real_oracle_headroom", "NOT_PRODUCED"),
        "best teacher pool gain": observations.get("teacher_pool_gain", "NOT_PRODUCED"),
        "student recovery": observations.get("student_recovery", "NOT_PRODUCED"),
        "disagreement sensor AUPRC": observations.get("sensor_auprc", "NOT_PRODUCED"),
        "total GPU/wall time": (
            f"{runtime.get('gpu_hours', 'NOT_PRODUCED')} GPU-h / "
            f"{runtime.get('wall_hours', 'NOT_PRODUCED')} wall-h"
        ),
        "scientific FAIL list": ", ".join(payload.get("failed_gates", [])) or "none",
        "diagnostic continuation list": ", ".join(diagnostic) or "none",
        "commit SHA": payload.get("commit", "NOT_PRODUCED"),
        "push status": payload.get("push", "NOT_PRODUCED"),
        "STATUS path": payload.get("status_path", STATUS_PATH),
    }
    _ = branches
    return [f"{label}: {values[label]}" for label in CONSOLE_ORDER]


__all__ = [
    "CONSOLE_ORDER",
    "FORBIDDEN_CLAIMS",
    "NEXT_ACTION_BY_RECOMMENDATION",
    "STATUS_FIRST_LINE_PREFIX",
    "STATUS_PATH",
    "STATUS_SECTIONS",
    "TABLE_IDS",
    "TABLE_TITLES",
    "build_tables",
    "console_summary",
    "render_status",
]
