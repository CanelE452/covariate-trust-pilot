"""Study 4 daily portfolio construction and budgeted allocation policies.

A portfolio is one UTC date's 07 UTC origin across all four zones.  Each policy
scores the zones and spends at most K premium slots on the highest scores; the
portfolio loss is the mean task loss over the four zones, where a selected zone
contributes its premium loss and an unselected zone its base loss.

Because the realised load does not depend on which forecast is used, every
policy is scored against the same outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .acquisition_features import ORIGIN_COLUMN, ZONE_COLUMN

DATE_COLUMN = "portfolio_date"
NO_PREMIUM = "P0_NO_PREMIUM"
ALL_PREMIUM = "P1_ALL_PREMIUM_REFERENCE"
RANDOM_K = "P2_RANDOM_K"
ROUND_ROBIN = "P3_ROUND_ROBIN"
BASE_UNCERTAINTY = "P4_BASE_UNCERTAINTY"
REVISION_MAGNITUDE = "P5_REVISION_MAGNITUDE"
RECENT_BASE_ERROR = "P6_RECENT_BASE_ERROR"
REPORTED_RELIABILITY = "P7_REPORTED_RELIABILITY"
RECENT_UTILITY = "P8_RECENT_FULL_INFORMATION_UTILITY"
VALUE_WQL = "P9_VALUE_PREDICTOR_WQL"
VALUE_Q90 = "P10_VALUE_PREDICTOR_Q90"
ORACLE = "P11_ORACLE_K"

SIMPLE_HEURISTICS = (ROUND_ROBIN, BASE_UNCERTAINTY, REVISION_MAGNITUDE,
                     RECENT_BASE_ERROR, REPORTED_RELIABILITY)
#: Policies whose score is on the value scale, so abstention is meaningful.
VALUE_SCALE_POLICIES = (RECENT_UTILITY, VALUE_WQL, VALUE_Q90, ORACLE)


class PortfolioError(ValueError):
    pass


@dataclass
class PortfolioSet:
    frame: pd.DataFrame          # complete portfolios only, one row per (date, zone)
    excluded: pd.DataFrame       # dates dropped and why
    zones: tuple[str, ...]

    @property
    def n_days(self) -> int:
        # An empty split (for example a fresh window with no data yet) never got
        # the date column, so report zero rather than raising.
        if self.frame.empty or DATE_COLUMN not in self.frame.columns:
            return 0
        return int(self.frame[DATE_COLUMN].nunique())


def build_portfolios(frame: pd.DataFrame, zones: list[str]) -> PortfolioSet:
    """Keep only dates where all four zones have a complete base and premium task."""
    work = frame.copy()
    work[DATE_COLUMN] = pd.DatetimeIndex(work[ORIGIN_COLUMN]).normalize()
    required = set(zones)

    keep_dates, excluded = [], []
    for date, group in work.groupby(DATE_COLUMN):
        present = set(group[ZONE_COLUMN])
        finite = group[["wql_m1", "wql_m3", "q90_m1", "q90_m3"]].notna().all(axis=1)
        if present != required:
            excluded.append({DATE_COLUMN: date, "reason": "missing_zone",
                             "n_zones": len(present),
                             "missing": ",".join(sorted(required - present))})
        elif not finite.all():
            excluded.append({DATE_COLUMN: date, "reason": "non_finite_loss",
                             "n_zones": len(present), "missing": ""})
        else:
            keep_dates.append(date)

    kept = work[work[DATE_COLUMN].isin(keep_dates)].copy()
    kept = kept.sort_values([DATE_COLUMN, ZONE_COLUMN]).reset_index(drop=True)
    return PortfolioSet(kept, pd.DataFrame(excluded), tuple(zones))


def top_k_mask(
    scores: np.ndarray, k: int, only_positive: bool, tie_order: np.ndarray
) -> np.ndarray:
    """Select at most k entries by score; ties broken by ``tie_order`` (ascending).

    ``only_positive`` implements abstention: a slot is not spent on a zone whose
    score is not strictly positive.
    """
    scores = np.asarray(scores, dtype=float)
    n = scores.size
    mask = np.zeros(n, dtype=bool)
    if k <= 0:
        return mask
    valid = np.isfinite(scores)
    if only_positive:
        valid &= scores > 0.0
    if not valid.any():
        return mask
    candidates = np.flatnonzero(valid)
    order = np.lexsort((tie_order[candidates], -scores[candidates]))
    mask[candidates[order[:k]]] = True
    return mask


def _tie_order(zones: pd.Series) -> np.ndarray:
    return np.argsort(np.argsort(zones.to_numpy().astype(str)))


def heuristic_scores(policy: str, day: pd.DataFrame, day_index: int,
                     reliability_sign: float) -> np.ndarray:
    n = len(day)
    if policy == ROUND_ROBIN:
        # Depends only on the calendar position and the zone order.
        rank = (np.arange(n) - day_index) % n
        return -rank.astype(float)
    if policy == BASE_UNCERTAINTY:
        return day["base_interval_width_mean"].to_numpy(dtype=float)
    if policy == REVISION_MAGNITUDE:
        return day["revision_rms"].to_numpy(dtype=float)
    if policy == RECENT_BASE_ERROR:
        return day["recent_base_wql_28"].to_numpy(dtype=float)
    if policy == REPORTED_RELIABILITY:
        return reliability_sign * day["reported_reliability_ratio"].to_numpy(dtype=float)
    raise PortfolioError(f"not a simple heuristic: {policy}")


def portfolio_losses(day: pd.DataFrame, mask: np.ndarray, objective: str) -> float:
    base = day[f"{objective}_m1"].to_numpy(dtype=float)
    premium = day[f"{objective}_m3"].to_numpy(dtype=float)
    return float(np.where(mask, premium, base).mean())


def evaluate_policy(
    portfolios: PortfolioSet,
    policy: str,
    k: int,
    *,
    objective: str = "wql",
    scores: pd.Series | None = None,
    reliability_sign: float = -1.0,
    allow_abstention: bool = True,
    only_positive: bool = True,
) -> pd.DataFrame:
    """Per-day loss and selection for one policy at one budget."""
    frame = portfolios.frame
    rows = []
    for day_index, (date, day) in enumerate(frame.groupby(DATE_COLUMN, sort=True)):
        day = day.sort_values(ZONE_COLUMN)
        tie = _tie_order(day[ZONE_COLUMN])
        if policy == NO_PREMIUM:
            mask = np.zeros(len(day), dtype=bool)
        elif policy == ALL_PREMIUM:
            mask = np.ones(len(day), dtype=bool)
        elif policy in SIMPLE_HEURISTICS:
            s = heuristic_scores(policy, day, day_index, reliability_sign)
            mask = top_k_mask(s, k, only_positive=False, tie_order=tie)
        elif policy == ORACLE:
            s = day[f"v_{objective}"].to_numpy(dtype=float)
            mask = top_k_mask(s, k, only_positive=allow_abstention and only_positive, tie_order=tie)
        elif scores is not None:
            s = scores.loc[day.index].to_numpy(dtype=float)
            use_positive = allow_abstention and only_positive and policy in VALUE_SCALE_POLICIES
            mask = top_k_mask(s, k, only_positive=use_positive, tie_order=tie)
        else:
            raise PortfolioError(f"policy {policy} needs a score series")

        if int(mask.sum()) > k and policy != ALL_PREMIUM:
            raise PortfolioError(f"{policy} exceeded the budget on {date}")
        rows.append(
            {
                DATE_COLUMN: date,
                "policy": policy,
                "k": k,
                "objective": objective,
                "loss": portfolio_losses(day, mask, objective),
                "n_selected": int(mask.sum()),
                "selected_zones": ",".join(sorted(day[ZONE_COLUMN][mask])),
                "iso_week": pd.Timestamp(date).strftime("%G-W%V"),
            }
        )
    return pd.DataFrame(rows)


def random_policy_distribution(
    portfolios: PortfolioSet, k: int, n_repetitions: int, seed: int, objective: str = "wql"
) -> tuple[pd.DataFrame, dict]:
    """Mean per-day loss over ``n_repetitions`` random top-K draws."""
    rng = np.random.default_rng(seed)
    frame = portfolios.frame
    days = list(frame.groupby(DATE_COLUMN, sort=True))
    per_rep = np.empty(n_repetitions, dtype=float)
    accumulated = np.zeros(len(days), dtype=float)
    for r in range(n_repetitions):
        losses = np.empty(len(days), dtype=float)
        for i, (_, day) in enumerate(days):
            day = day.sort_values(ZONE_COLUMN)
            chosen = rng.choice(len(day), size=min(k, len(day)), replace=False)
            mask = np.zeros(len(day), dtype=bool)
            mask[chosen] = True
            losses[i] = portfolio_losses(day, mask, objective)
        accumulated += losses
        per_rep[r] = losses.mean()
    mean_daily = accumulated / n_repetitions
    daily = pd.DataFrame(
        {
            DATE_COLUMN: [d for d, _ in days],
            "policy": RANDOM_K,
            "k": k,
            "objective": objective,
            "loss": mean_daily,
            "n_selected": k,
            "selected_zones": "",
            "iso_week": [pd.Timestamp(d).strftime("%G-W%V") for d, _ in days],
        }
    )
    summary = {
        "policy": RANDOM_K,
        "k": k,
        "objective": objective,
        "n_repetitions": n_repetitions,
        "mean_of_repetition_means": float(per_rep.mean()),
        "sd_of_repetition_means": float(per_rep.std(ddof=1)),
        "p025": float(np.percentile(per_rep, 2.5)),
        "p975": float(np.percentile(per_rep, 97.5)),
    }
    return daily, summary


def oracle_recovery(
    loss_no_premium: np.ndarray, loss_policy: np.ndarray, loss_oracle: np.ndarray
) -> dict[str, float]:
    """Daily and aggregate share of the oracle headroom that a policy recovers."""
    base = np.asarray(loss_no_premium, dtype=float)
    policy = np.asarray(loss_policy, dtype=float)
    oracle = np.asarray(loss_oracle, dtype=float)
    denom = base - oracle
    valid = denom > 0.0
    daily = np.full(base.shape, np.nan)
    daily[valid] = (base[valid] - policy[valid]) / denom[valid]
    aggregate_denom = float(base.mean() - oracle.mean())
    return {
        "daily_mean_recovery": float(np.nanmean(daily)) if valid.any() else float("nan"),
        "excluded_fraction": float(1.0 - valid.mean()),
        "aggregate_recovery": float((base.mean() - policy.mean()) / aggregate_denom)
        if abs(aggregate_denom) > 1e-12 else float("nan"),
    }


def selection_overlap(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Jaccard-style agreement between two policies' selected zone sets."""
    merged = a.merge(b, on=[DATE_COLUMN, "k"], suffixes=("_a", "_b"))
    if merged.empty:
        return float("nan")
    agree = []
    for _, row in merged.iterrows():
        sa = set(filter(None, str(row["selected_zones_a"]).split(",")))
        sb = set(filter(None, str(row["selected_zones_b"]).split(",")))
        union = sa | sb
        agree.append(1.0 if not union else len(sa & sb) / len(union))
    return float(np.mean(agree))


def zone_selection_rates(daily: pd.DataFrame, zones: tuple[str, ...]) -> dict[str, float]:
    counts = {z: 0 for z in zones}
    total = 0
    for value in daily["selected_zones"]:
        picked = [z for z in str(value).split(",") if z]
        for z in picked:
            counts[z] = counts.get(z, 0) + 1
        total += len(picked)
    if total == 0:
        return {z: 0.0 for z in zones}
    return {z: counts[z] / total for z in zones}
