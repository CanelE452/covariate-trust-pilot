"""Study 4 report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


def fmt_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    text = [[("" if v is None else str(v)) for v in r] for r in rows]
    widths = [len(h) for h in headers]
    for row in text:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    head = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    body = "\n".join("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)) for r in text)
    return "```\n" + head + "\n" + "─" * len(head) + "\n" + body + "\n```\n"


def _num(value: Any, digits: int = 4) -> str:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "nan" if f != f else f"{f:.{digits}f}"


def _load(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / "tables" / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        # A deliberately empty table (for example an unevaluable fresh window).
        return pd.DataFrame()


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _gate_table(gate: dict) -> str:
    rows = []
    for c in gate["criteria"]:
        value = c["value"]
        if isinstance(value, dict):
            shown = ", ".join(
                f"{k}={v if isinstance(v, bool) else _num(v, 4)}" for k, v in value.items()
            )
        elif isinstance(value, bool):
            shown = str(value)
        else:
            shown = _num(value, 4)
        rows.append([c["cid"], c["description"], shown, str(c["threshold"]), str(c["passed"])])
    return fmt_table(["id", "criterion", "value", "threshold", "passed"], rows)


def _policy_table(frame: pd.DataFrame, k: int, objective: str = "wql") -> str:
    sub = frame[(frame["k"] == k) & (frame["objective"] == objective)]
    if sub.empty:
        return "결과 없음\n"
    sub = sub.sort_values("mean_loss")
    rows = [
        [r["policy"], _num(r["mean_loss"], 5), _num(r.get("rel_vs_no_premium", float("nan"))),
         _num(r.get("oracle_recovery", float("nan")), 3),
         _num(r.get("mean_selected", float("nan")), 2),
         _num(r.get("max_zone_share", float("nan")), 3)]
        for _, r in sub.iterrows()
    ]
    return fmt_table(
        ["policy", "portfolio loss", "vs NO_PREMIUM", "oracle recovery", "slots/day", "max zone share"],
        rows,
    )


def build_report(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    manifest = _load_json(run_dir / "manifest.json") or {}
    prereg = _load_json(run_dir / "preregistration.json") or {}
    gates = {name: _load_json(run_dir / "tables" / f"gate_{name}.json")
             for name in ("ba0", "ba1", "ba2", "ba3", "ba4", "ba5")}
    policy = _load(run_dir, "policy_summary.csv")
    value_dist = _load(run_dir, "value_distribution.csv")
    zone_summary = _load(run_dir, "zone_summary.csv")
    month_summary = _load(run_dir, "month_summary.csv")
    boot = _load(run_dir, "bootstrap_summary.csv")
    overlap = _load(run_dir, "selection_overlap.csv")
    random_summary = _load(run_dir, "random_policy_summary.csv")
    fresh = _load(run_dir, "fresh_confirmation.csv")
    models = _load(run_dir, "../models/validation_results.csv")

    out: list[str] = []
    add = out.append

    add("# Study 4 — Budgeted Premium Forecast Slot Allocation\n")
    add(f"- run id: `{manifest.get('run_id', run_dir.name)}`\n")
    add(f"- preregistration SHA-256: `{manifest.get('preregistration_sha256', 'n/a')}`\n")
    add(f"- 시작 commit `{manifest.get('start_commit', 'n/a')}`, "
        f"diff hash `{manifest.get('start_diff_sha256', 'n/a')}`\n")
    add(f"- 실행 {manifest.get('started_at','?')} → {manifest.get('finished_at','?')}, "
        f"{_num(manifest.get('wall_seconds', float('nan')),1)} s, "
        f"peak RSS {_num(manifest.get('peak_rss_mb', float('nan')),1)} MB\n")
    add("\n태그: `[확인]` 코드·데이터·결과로 직접 확인 / `[추정]` 해석 / `[미검증]` 범위 밖.\n")

    add("\n## 1. 연구 질문\n")
    add("매일 07 UTC origin에서 4개 NYISO zone 중 최대 K개(K=1,2)에만 premium forecast(M3)를 "
        "적용할 수 있을 때, 미래 부하를 보기 전에 대상을 골라 portfolio forecast loss를 "
        "줄일 수 있는가. [확인]\n")

    add("\n## 2. 검증 범위와 비범위\n")
    add(
        "**검증**: 동일한 realized load에 대해 budget 하 zone 선택이 portfolio WQL을 바꾸는가.\n\n"
        "**비검증**: 실제 금전 절감, 다기간 재고·capacity state, partial-feedback/bandit "
        "learning, 4 zone·단일 정보원·단일 모델 밖 일반화. premium slot은 **추상 단위**이며 "
        "ECMWF 구매가격이나 GPU 과금이 아니다. [확인]\n"
    )

    add("\n## 3. 기존 NYISO–ECMWF 자산 provenance\n")
    prov = manifest.get("study3_assets", {})
    if prov:
        add(fmt_table(["asset", "sha256 (앞 16)", "rows"],
                      [[k, str(v.get("sha256", ""))[:16], v.get("rows", "")]
                       for k, v in sorted(prov.items())]))
    add(f"- Study 3 run: `{manifest.get('study3_run_dir','?')}` — **read-only로만 참조**했고 "
        f"실행 후 hash 재검증 결과 변경 없음: {not manifest.get('study3_hashes_changed', True)} [확인]\n")
    add(f"- 재추론 여부: {manifest.get('reinference_performed', 'no')} "
        "(기존 prediction cache에 M1/M3 WQL이 이미 있어 재추론하지 않았다) [확인]\n")

    add("\n## 4. Base M1과 premium M3 정의\n")
    add("Study 3 정의를 그대로 재사용한다: context는 동일하고 future block은 calendar가 동일하며 "
        "M3만 00Z ECMWF forecast temperature를 추가로 받는다. `assert_fair_comparison`이 이를 "
        "강제한다. [확인]\n")

    add("\n## 5. Premium slot budget의 의미\n")
    add("비싼 weather-conditioned 실행 슬롯 / premium pipeline capacity / 제한된 expert "
        "처리 슬롯. slot cost는 추상 단위 1이다. [확인]\n")

    add("\n## 6. Train · validation · retrospective test\n")
    days = manifest.get("portfolio_days", {})
    add(fmt_table(["split", "기간", "complete portfolio days"],
                  [[k, manifest.get("period_labels", {}).get(k, ""), v] for k, v in days.items()]))
    add("retrospective test 기간의 aggregate M1/M3 결과는 Study 3에서 이미 관찰됐으므로 "
        "**untouched confirmation이 아니다** — `RETROSPECTIVE_HELDOUT_PILOT`이라 부른다. [확인]\n")
    if manifest.get("excluded_days"):
        add(f"\n제외된 날짜 수: {manifest['excluded_days']} (사유는 "
            "`data/portfolio_origins.parquet`의 exclusion 로그) [확인]\n")

    add("\n## 7. Fresh confirmation\n")
    if fresh.empty:
        add(f"{manifest.get('fresh_status', 'NOT_EVALUABLE')} — "
            f"complete portfolio day {manifest.get('fresh_days', 0)}개 [확인]\n")
    else:
        add(fmt_table(list(fresh.columns), fresh.values.tolist()))

    add("\n## 8. Premium value distribution\n")
    if not value_dist.empty:
        cols = ["group", "n", "mean", "median", "sd", "positive_rate", "p05", "p95", "skew"]
        cols = [c for c in cols if c in value_dist.columns]
        add(fmt_table(cols, [[_num(r[c], 4) if c != "group" and c != "n" else r[c]
                              for c in cols] for _, r in value_dist.iterrows()]))

    add("\n## 9. Oracle budget headroom\n")
    for k in (1, 2):
        add(f"\n### K = {k}\n")
        add(_policy_table(policy, k))

    add("\n## 10. Simple heuristics\n")
    add("P3 ROUND_ROBIN / P4 BASE_UNCERTAINTY / P5 REVISION_MAGNITUDE / P6 RECENT_BASE_ERROR / "
        "P7 REPORTED_RELIABILITY 중 validation K=1 WQL이 가장 낮은 것을 best simple heuristic으로 "
        f"고정했다: **{manifest.get('best_simple_heuristic','?')}** [확인]\n")

    add("\n## 11. Value predictor\n")
    add(f"- 후보: {', '.join(prereg.get('value_models', {}).get('candidates', []))}\n")
    add(f"- **선택된 모델: {manifest.get('selected_value_model','?')}** "
        f"(validation K=1 portfolio WQL 하나로만 선택, test에서 재선택하지 않음) [확인]\n")
    if not models.empty:
        add(fmt_table(list(models.columns)[:4],
                      [[_num(r[c], 5) if c == "validation_wql_k1" else r[c]
                        for c in list(models.columns)[:4]] for _, r in models.iterrows()]))
    if manifest.get("model_failures"):
        add(f"\n실패한 후보(조용히 대체하지 않고 기록): {manifest['model_failures']} [확인]\n")

    add("\n## 12. K = 1 결과\n")
    add(_policy_table(policy, 1))
    add("\n## 13. K = 2 결과\n")
    add(_policy_table(policy, 2))

    add("\n## 14. Objective-specific selector\n")
    if not overlap.empty:
        add(fmt_table(list(overlap.columns),
                      [[_num(v, 4) if isinstance(v, float) else v for v in row]
                       for row in overlap.values.tolist()]))

    add("\n## 15. Zone · month 결과\n")
    if not zone_summary.empty:
        add(fmt_table(list(zone_summary.columns),
                      [[_num(v, 5) if isinstance(v, float) else v for v in row]
                       for row in zone_summary.values.tolist()]))
    if not month_summary.empty:
        add("\n")
        add(fmt_table(list(month_summary.columns),
                      [[_num(v, 5) if isinstance(v, float) else v for v in row]
                       for row in month_summary.head(24).values.tolist()]))

    for idx, name in ((16, "ba0"), (17, "ba1"), (18, "ba2"), (19, "ba3"), (20, "ba4"), (21, "ba5")):
        gate = gates[name]
        add(f"\n## {idx}. Gate {name.upper()}\n")
        if gate is None:
            add(f"{name.upper()}: NOT_RUN — 앞 단계가 통과하지 않으면 실행하지 않는다. [확인]\n")
            continue
        add(f"**status: {gate['status']}** — {gate['decision']}\n\n")
        add(_gate_table(gate))
        if gate["fail_reasons"]:
            add("\n사유:\n")
            for reason in gate["fail_reasons"]:
                add(f"- {reason}\n")

    add("\n## 22. 관찰\n")
    if not random_summary.empty:
        add("random policy의 무작위 변동(2000회)과 week-cluster bootstrap 불확실성은 별도로 잰다:\n")
        add(fmt_table(list(random_summary.columns),
                      [[_num(v, 5) if isinstance(v, float) else v for v in row]
                       for row in random_summary.values.tolist()]))
    if not boot.empty:
        add("\n주요 paired 비교 (ISO week cluster bootstrap):\n")
        cols = ["label", "method_a", "method_b", "rel_improvement", "diff_ci_low",
                "diff_ci_high", "week_win_rate", "n_weeks"]
        cols = [c for c in cols if c in boot.columns]
        add(fmt_table(cols, [[_num(r[c], 4) if isinstance(r[c], float) else r[c] for c in cols]
                             for _, r in boot.iterrows()]))

    add("\n## 23. 판정\n")
    add(f"**{manifest.get('final_status','?')}** [확인]\n")

    add("\n## 24. 한계\n")
    add(
        "- zone이 4개뿐이고 정보원은 future temperature 하나뿐이다. [확인]\n"
        "- premium slot 비용은 **추상 단위**이며 실제 금액이 아니다. [확인]\n"
        "- 과거에는 모든 M1/M3 결과를 안다고 두는 **full-information offline setting**이다. "
        "P8은 그래서 deployable baseline이 아니다. [확인]\n"
        "- partial-feedback / bandit learning은 평가하지 않았다. [미검증]\n"
        "- 다기간 inventory·capacity state는 평가하지 않았다. [미검증]\n"
        "- retrospective test는 Study 3와 기간을 공유한다 — untouched confirmation이 아니다. [확인]\n"
        "- 금전 가치는 검증하지 않았다. [미검증]\n"
        "- Chronos-2 한 모델뿐이다. [확인]\n"
        "- q90 pinball은 one-shot asymmetric-loss proxy일 뿐 다기간 inventory decision의 "
        "증거가 아니다. [확인]\n"
    )

    add("\n## 25. 다음 단계\n")
    status = manifest.get("final_status", "")
    if status.startswith("BUDGETED_ACQUISITION_NO_GO"):
        add("- budget headroom이 없다. 이 방법 주제는 여기서 종료한다.\n")
    elif status.startswith("SIMPLE_RULE"):
        add("- 단순 규칙으로 충분하다. 복잡한 value model 개발은 No-Go이고 운영 가이드만 남긴다.\n")
    elif status.startswith("VALUE_NOT_PREDICTABLE"):
        add("- headroom은 있으나 ex ante로 예측되지 않는다. 현재 feature로는 method No-Go.\n")
    elif status.startswith("FORECAST_VALUE_ROUTING_GO") or status.startswith("DECISION_VALUE"):
        add("- 통과했더라도 **바로 논문 GO가 아니다**. 최소한 더 많은 series, 여러 정보원, "
            "실제 비용 또는 compute cost, partial-feedback learning, FAME·active feature "
            "acquisition 직접 비교가 필요하다. [확인]\n")
    else:
        add("- 판정이 INCONCLUSIVE이므로 자동 확대 없이 원인만 보고한다.\n")

    add("\n## 26. 재현 명령\n")
    add("```\n" + "\n".join(prereg.get("commands", [])) + "\n```\n")

    path = run_dir / "reports" / "budget_acquisition_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(out), encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# Figures (one chart per file, no subplots, default matplotlib colours)
# --------------------------------------------------------------------------


def _fig_save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def build_figures(
    figures_dir: Path,
    *,
    policy_summary: pd.DataFrame,
    daily: pd.DataFrame,
    labels: pd.DataFrame,
    features: pd.DataFrame,
    predictions: pd.DataFrame | None,
    overlap_frame: pd.DataFrame,
    fresh: pd.DataFrame,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures_dir = Path(figures_dir)
    size = (10.0, 5.5)

    # BA1 - portfolio WQL by policy and budget
    wql = policy_summary[policy_summary["objective"] == "wql"]
    if not wql.empty:
        pivot = wql.pivot_table(index="policy", columns="k", values="mean_loss").sort_values(
            by=list(wql["k"].unique())[0]
        )
        fig, ax = plt.subplots(figsize=size)
        x = np.arange(pivot.shape[0], dtype=float)
        width = 0.8 / max(pivot.shape[1], 1)
        for i, k in enumerate(pivot.columns):
            ax.bar(x + i * width - 0.4 + width / 2, pivot[k].to_numpy(), width, label=f"K={k}")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index.tolist(), rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("portfolio WQL (lower is better)")
        ax.set_title("Figure BA1 - portfolio WQL by policy and budget")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        _fig_save(fig, figures_dir / "figure_ba1_policy_wql.png")

    # BA2 - daily oracle headroom
    no_prem = daily[(daily["policy"].str.startswith("P0")) & (daily["objective"] == "wql")]
    oracle = daily[(daily["policy"].str.startswith("P11")) & (daily["objective"] == "wql")]
    if not no_prem.empty and not oracle.empty:
        merged = no_prem.merge(oracle, on=["portfolio_date", "k"], suffixes=("_base", "_oracle"))
        head = merged["loss_base"] - merged["loss_oracle"]
        fig, ax = plt.subplots(figsize=size)
        ax.hist(head, bins=60)
        ax.axvline(0.0, linestyle="--", linewidth=1.0, color="black")
        ax.axvline(float(head.mean()), linewidth=1.5, color="red")
        ax.set_xlabel("daily oracle headroom: WQL(NO_PREMIUM) - WQL(ORACLE_K)")
        ax.set_ylabel("number of portfolio days")
        ax.set_title(f"Figure BA2 - daily oracle headroom (mean {head.mean():.5f})")
        ax.grid(axis="y", alpha=0.3)
        _fig_save(fig, figures_dir / "figure_ba2_oracle_headroom.png")

    # BA3 - predicted versus actual premium value
    if predictions is not None and not predictions.empty:
        fig, ax = plt.subplots(figsize=size)
        ax.scatter(predictions["predicted"], predictions["actual"], s=6, alpha=0.35)
        ax.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
        ax.axvline(0.0, linestyle="--", linewidth=1.0, color="black")
        ax.set_xlabel("predicted premium value")
        ax.set_ylabel("actual premium value V_wql")
        corr = predictions[["predicted", "actual"]].corr().iloc[0, 1]
        ax.set_title(f"Figure BA3 - predicted vs actual premium value (r = {corr:.3f})")
        ax.grid(alpha=0.3)
        _fig_save(fig, figures_dir / "figure_ba3_predicted_vs_actual.png")

    # BA4 - oracle recovery by policy
    rec = wql.dropna(subset=["oracle_recovery"]) if "oracle_recovery" in wql.columns else pd.DataFrame()
    if not rec.empty:
        pivot = rec.pivot_table(index="policy", columns="k", values="oracle_recovery")
        fig, ax = plt.subplots(figsize=size)
        x = np.arange(pivot.shape[0], dtype=float)
        width = 0.8 / max(pivot.shape[1], 1)
        for i, k in enumerate(pivot.columns):
            ax.bar(x + i * width - 0.4 + width / 2, pivot[k].to_numpy(), width, label=f"K={k}")
        ax.axhline(1.0, linestyle="--", linewidth=1.0, color="black")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index.tolist(), rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("aggregate oracle recovery")
        ax.set_title("Figure BA4 - share of the oracle headroom recovered")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        _fig_save(fig, figures_dir / "figure_ba4_oracle_recovery.png")

    # BA5 - premium-positive rate by zone
    if not labels.empty:
        rate = labels.groupby("zone")["premium_positive"].mean()
        fig, ax = plt.subplots(figsize=size)
        ax.bar(np.arange(len(rate), dtype=float), rate.to_numpy())
        ax.axhline(0.5, linestyle="--", linewidth=1.0, color="black")
        ax.set_xticks(np.arange(len(rate), dtype=float))
        ax.set_xticklabels(rate.index.tolist())
        ax.set_ylabel("share of origins where premium helped")
        ax.set_title("Figure BA5 - premium-positive rate by zone")
        ax.grid(axis="y", alpha=0.3)
        _fig_save(fig, figures_dir / "figure_ba5_zone_positive_rate.png")

    # BA6 / BA7 - drivers against premium value
    joined = labels.merge(features, on=["zone", "origin_utc"], how="inner")
    for tag, column, title in (
        ("ba6", "revision_rms", "Figure BA6 - weather revision magnitude vs premium value"),
        ("ba7", "base_interval_width_mean", "Figure BA7 - base uncertainty vs premium value"),
    ):
        if column not in joined.columns:
            continue
        sub = joined[[column, "v_wql"]].dropna()
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=size)
        ax.scatter(sub[column], sub["v_wql"], s=6, alpha=0.3)
        ax.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
        ax.set_xlabel(column)
        ax.set_ylabel("premium value V_wql")
        corr = sub.corr().iloc[0, 1]
        ax.set_title(f"{title} (r = {corr:.3f})")
        ax.grid(alpha=0.3)
        _fig_save(fig, figures_dir / f"figure_{tag}_{column}.png")

    # BA8 - selection overlap
    if not overlap_frame.empty:
        fig, ax = plt.subplots(figsize=size)
        x = np.arange(len(overlap_frame), dtype=float)
        ax.bar(x, overlap_frame["overlap"].to_numpy())
        ax.axhline(0.95, linestyle="--", linewidth=1.0, color="black")
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={int(k)}" for k in overlap_frame["k"]], rotation=0)
        ax.set_ylabel("selection overlap (WQL selector vs q90 selector)")
        ax.set_title("Figure BA8 - do the two objectives pick different zones?")
        ax.grid(axis="y", alpha=0.3)
        _fig_save(fig, figures_dir / "figure_ba8_selection_overlap.png")

    # BA9 - weekly regret of the value policy
    value = daily[(daily["policy"].str.startswith("P9")) & (daily["objective"] == "wql")]
    if not value.empty and not oracle.empty:
        merged = value.merge(oracle, on=["portfolio_date", "k"], suffixes=("_value", "_oracle"))
        merged["regret"] = merged["loss_value"] - merged["loss_oracle"]
        weekly = merged.groupby(["iso_week_value", "k"], as_index=False)["regret"].mean()
        fig, ax = plt.subplots(figsize=size)
        for k, group in weekly.groupby("k"):
            ax.plot(np.arange(len(group), dtype=float), group["regret"].to_numpy(),
                    linewidth=1.2, label=f"K={k}")
        ax.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
        ax.set_xlabel("ISO week index")
        ax.set_ylabel("weekly mean regret vs oracle")
        ax.set_title("Figure BA9 - value-policy regret by week")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        _fig_save(fig, figures_dir / "figure_ba9_weekly_regret.png")

    # BA10 - fresh confirmation
    fig, ax = plt.subplots(figsize=size)
    if fresh.empty:
        ax.text(0.5, 0.5, "fresh confirmation NOT_EVALUABLE\n(no 2026-07 portfolio days)",
                ha="center", va="center", fontsize=12)
        ax.set_axis_off()
    else:
        x = np.arange(len(fresh), dtype=float)
        ax.bar(x, fresh["mean_loss"].to_numpy())
        ax.set_xticks(x)
        ax.set_xticklabels(fresh["policy"].tolist(), rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("portfolio WQL")
        ax.grid(axis="y", alpha=0.3)
    ax.set_title("Figure BA10 - fresh confirmation")
    _fig_save(fig, figures_dir / "figure_ba10_fresh_confirmation.png")
