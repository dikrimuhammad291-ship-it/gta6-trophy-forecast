"""Forecast GTA VI (PS5) trophy earn rates on Exophase.

For every trophy in a case, forecast its earn rate (percent of tracked players
who hold it) at the case's target, as five quantiles, written to <id>.out.

Model
-----
Each trophy's rate is fit on the logit scale as a straight line in log time,

    logit(rate) = a + b * ln(1 + h),

where h is hours since release, over the last 24 hours of the case's history
(the whole history when fewer than three reads fall in the window).  Log time
is concave, so the line bends the launch surge down instead of extrapolating
it straight past 100%, and the recent window tracks the current pace rather
than the explosive first hours.  Each trophy's slope is shrunk toward its
tier's mean slope (story bronze trophies climb fast, platinum and gold
slowly).  The fit runs on the distinct page states in the history
(consecutive identical reads collapse to one state, so a frozen page does
not outweigh a moving one), with recency weights, and the logit scale is
the scale the contest scores on, so the fit optimises the right errors.

Refreshes: Exophase refreshes a game's counts hours apart, not continuously.
Between refreshes the page is frozen; at a refresh the tracked-player count
jumps and most rates fall.  When the page has sat unchanged for a long time
and the horizon is short, a refresh probably lands inside the horizon and the
answer is the diluted post-refresh page, so the median is shifted down by an
expected dilution (measured from refreshes in the history when there are
any, else a prior that is large on launch day and fades as the player base
settles).  The shift fades out for long horizons, where the trend already
averages over many refreshes.

Spread: each trophy's quantiles sit on the logit scale around its median,
with a spread combining the fit's residual, a random-walk term that widens
with the horizon, and a refresh term for where the target falls inside a
refresh cycle.  The lower side is a little wider than the upper side: a
refresh is a downside shock, while growth between refreshes is smooth.

Deterministic: pure float arithmetic, no randomness, cases processed in name
order, so two runs write byte-identical files.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at LEVELS.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Fallback release instant (the case's game.release_date overrides it).
RELEASE_FALLBACK = datetime(2026, 11, 19, tzinfo=timezone.utc)

# --- Tunable model constants ---------------------------------------------
FIT_WINDOW_H = 24.0     # fit window: last N hours of history
RW_SIGMA = 0.25         # logit spread per sqrt(day) of horizon (random walk)
SPREAD_FLOOR = 0.35     # minimum logit spread: never claim false precision
SPREAD_CAP = 2.0        # maximum logit spread
SLOPE_SHRINK_K = 3.0    # tier-shrinkage strength for per-trophy slopes
TIER_PRIOR_K = 3.0      # shrinkage of small-tier mean slopes toward global
RECENCY_HALF_LIFE_H = 12.0  # recency weight half-life for the fits
REFRESH_INTERVAL_PRIOR_H = 5.0   # assumed hours between refreshes, no data
DILUTION_PRIOR = 0.85   # launch-day refresh dilution depth (prior). The
# launch-night refresh the rules document cut rates to roughly a fifth to a
# quarter of their pre-refresh values (638 -> 3,868 players tracked, most
# trophies' earned counts barely moving); the prior centers there and fades
# as the player base settles. It only binds when no refresh is in the
# history: with refreshes observed, the measured dilution replaces it.
DILUTION_DECAY_H = 24.0  # how fast the dilution prior fades
DOWN_MULT = 1.2         # lower-side spread multiplier (refresh downside)
UP_MULT = 0.9           # upper-side spread multiplier


def parse_time(text):
    """ISO 8601 UTC timestamp -> aware datetime (handles 'Z' and fractions)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, deterministically."""
    return math.floor(x * 10000.0 + 0.5) / 10000.0


def wls(xs, ys, ws):
    """Weighted least squares of y on x. Returns (slope, intercept)."""
    sw = sum(ws)
    if sw <= 0:
        return 0.0, (sum(ys) / len(ys) if ys else 0.0)
    sx = sum(w * x for x, w in zip(xs, ws))
    sy = sum(w * y for y, w in zip(ys, ws))
    sxx = sum(w * x * x for x, w in zip(xs, ws))
    sxy = sum(w * x * y for x, y, w in zip(xs, ys, ws))
    denom = sw * sxx - sx * sx
    if abs(denom) <= 1e-12:
        return 0.0, sy / sw
    slope = (sw * sxy - sx * sy) / denom
    return slope, (sy - slope * sx) / sw


def distinct_states(history):
    """Collapse consecutive identical reads to one state each.

    A state is (time, players_tracked, rates list); its time is when it was
    first seen.  Returns the states in time order.
    """
    states = []
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if states and states[-1][2] == key:
            continue
        states.append((parse_time(read["at"]), read["players_tracked"],
                       key, list(read["rates"])))
    return [(t, p, r) for (t, p, _k, r) in states]


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    horizon_min = float(case["horizon_minutes"])
    horizon_days = horizon_min / 1440.0
    horizon_h = horizon_min / 60.0

    release = RELEASE_FALLBACK
    game = case.get("game") or {}
    if game.get("release_date"):
        try:
            release = datetime.fromisoformat(game["release_date"]).replace(
                tzinfo=timezone.utc)
        except ValueError:
            pass
    target = parse_time(case["target"])
    checkpoint = parse_time(case["at"])
    h_target = (target - release).total_seconds() / 3600.0
    x_target = math.log1p(max(0.0, h_target))
    hours_since_release = max(
        0.0, (checkpoint - release).total_seconds() / 3600.0)

    if not history:
        # No data at all: flat 1% with a wide spread beats crashing.
        return {"trophies": [{"id": t["id"], "q": _quantiles(1.0, 1.5, 1.5)}
                            for t in trophies]}

    states = distinct_states(history)
    n_states = len(states)
    times = [s[0] for s in states]
    players = [float(s[1]) for s in states]
    hours = [(t - release).total_seconds() / 3600.0 for t in times]
    xs = [math.log1p(max(0.0, h)) for h in hours]
    last_rates = states[-1][2]

    # Recency weights: exponential decay from the newest state.
    last_t = times[-1]
    weights = []
    for t in times:
        age_h = (last_t - t).total_seconds() / 3600.0
        weights.append(max(0.05, 0.5 ** (age_h / RECENCY_HALF_LIFE_H)))

    # --- Refreshes seen in the history ------------------------------------
    # A refresh is a state whose player count differs from the previous one.
    refresh_gaps_h = []
    refresh_dilutions = []
    for j in range(1, n_states):
        if players[j] == players[j - 1]:
            continue
        gap_h = (times[j] - times[j - 1]).total_seconds() / 3600.0
        refresh_gaps_h.append(gap_h)
        ratios = []
        for i in range(len(trophies)):
            before = states[j - 1][2][i]
            after = states[j][2][i]
            if before is None or after is None or before <= 0:
                continue
            ratios.append(after / before)
        if ratios:
            ratios.sort()
            refresh_dilutions.append(ratios[len(ratios) // 2])
    if refresh_gaps_h:
        interval_h = sum(refresh_gaps_h) / len(refresh_gaps_h)
    else:
        interval_h = REFRESH_INTERVAL_PRIOR_H
    if refresh_dilutions:
        dilution = sum(refresh_dilutions) / len(refresh_dilutions)
        dilution = min(max(dilution, 0.05), 1.0)
    else:
        dilution = 1.0 - DILUTION_PRIOR * math.exp(
            -hours_since_release / DILUTION_DECAY_H)
    # Time the page has sat unchanged, in hours.
    frozen_h = (checkpoint - times[-1]).total_seconds() / 3600.0
    if n_states >= 2:
        frozen_h = max(frozen_h,
                       (times[-1] - times[-2]).total_seconds() / 3600.0)
    p_refresh = min(0.95, (frozen_h + horizon_h) / max(interval_h, 0.5))
    # The refresh adjustment only matters when the horizon is short against
    # one refresh cycle; over many cycles the trend already averages it out.
    cycle_weight = min(1.0, interval_h / max(horizon_h, 1.0))
    refresh_mult = 1.0 - p_refresh * (1.0 - dilution) * cycle_weight
    refresh_mult = min(max(refresh_mult, 0.15), 1.0)
    # Spread for where the target falls inside a refresh cycle.
    refresh_std = 0.5 * abs(math.log(max(dilution, 0.05)))
    refresh_std = min(refresh_std, 0.8)

    # --- Per-trophy log-time fits on the logit scale -----------------------
    # First pass: raw slopes, to build tier means.
    raw = []  # per trophy: (slope, intercept, resid, n_points, pts)
    for i in range(len(trophies)):
        pts = [(xs[j], logit(states[j][2][i]), weights[j])
               for j in range(n_states) if states[j][2][i] is not None]
        # Prefer the recent window; fall back to all points when too few.
        window = []
        for j in range(n_states):
            if states[j][2][i] is None:
                continue
            age_h = (last_t - times[j]).total_seconds() / 3600.0
            if age_h <= FIT_WINDOW_H:
                window.append((xs[j], logit(states[j][2][i]), weights[j]))
        use = window if len(window) >= 3 else pts
        if len(use) >= 2 and max(p[0] for p in use) > min(p[0] for p in use):
            slope, inter = wls([p[0] for p in use], [p[1] for p in use],
                               [p[2] for p in use])
            resid = _fit_resid([p[0] for p in use], [p[1] for p in use],
                               slope, inter)
        elif use:
            slope, inter = 0.0, use[-1][1]
            resid = 0.0
        else:
            slope, inter, resid = 0.0, logit(0.01), 0.0
        raw.append((slope, inter, resid, len(use), use))

    # Tier mean slopes, small tiers shrunk toward the global mean.
    tiers = {}
    for t, (slope, _i, _r, n, _u) in zip(trophies, raw):
        tiers.setdefault(t["tier"], []).append((slope, max(n, 1)))
    all_slopes = [(s, w) for _t, lst in tiers.items() for (s, w) in lst]
    global_slope = _wmean(all_slopes)
    tier_slope = {}
    for tier, lst in tiers.items():
        mean_t = _wmean(lst)
        n_t = len(lst)
        tier_slope[tier] = (n_t * mean_t + TIER_PRIOR_K * global_slope) / (
            n_t + TIER_PRIOR_K)

    rw = RW_SIGMA * math.sqrt(max(horizon_days, 0.0))
    rows = []
    for i, trophy in enumerate(trophies):
        slope, inter, resid, n_pts, use = raw[i]
        last_rate = last_rates[i]
        if last_rate is not None and last_rate >= 99.9:
            # A trophy everyone holds stays held: pin it near 100%.
            rows.append({"id": trophy["id"],
                         "q": _quantiles(99.99, 0.05, 0.05)})
            continue
        if n_pts >= 2:
            # Shrink the slope toward the tier mean, re-anchor the intercept.
            t_slope = tier_slope[trophy["tier"]]
            slope = (n_pts * slope + SLOPE_SHRINK_K * t_slope) / (
                n_pts + SLOPE_SHRINK_K)
            sw = sum(w for _x, _y, w in use)
            inter = sum(w * (y - slope * x) for x, y, w in use) / sw
            resid = _fit_resid([p[0] for p in use], [p[1] for p in use],
                               slope, inter)
            median = inv_logit(inter + slope * x_target)
            spread_extra = 0.0
        else:
            # Too few points to fit a line: stay flat.
            median = last_rate if last_rate is not None else 1.0
            spread_extra = 0.3
        median *= refresh_mult
        median = clamp_rate(median)

        resid_c = min(max(resid, 0.05), 1.2)
        spread = math.sqrt(resid_c ** 2 + rw ** 2 + refresh_std ** 2
                           + spread_extra ** 2)
        if n_pts == 0:
            spread *= 1.5  # nobody holds it yet: the start is uncertain
        spread = min(max(spread, SPREAD_FLOOR), SPREAD_CAP)
        rows.append({"id": trophy["id"],
                     "q": _quantiles(median, spread * DOWN_MULT,
                                     spread * UP_MULT)})
    return {"trophies": rows}


def _wmean(pairs):
    sw = sum(w for _v, w in pairs)
    if sw <= 0:
        return 0.0
    return sum(v * w for v, w in pairs) / sw


def _fit_resid(xs, ys, slope, inter):
    """Std of the residuals of a line fit."""
    if not xs:
        return 0.0
    diffs = [y - (inter + slope * x) for x, y in zip(xs, ys)]
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    return math.sqrt(max(var, 0.0))


def _quantiles(median, spread_down, spread_up):
    """Five quantiles around median on the logit scale (asymmetric spread)."""
    center = logit(median)
    out = []
    for tau, z in zip(LEVELS, Z):
        s = spread_down if z < 0 else spread_up
        out.append(round4(inv_logit(center + z * s)))
    # Rounding must not break the ordering.
    for k in range(1, len(out)):
        if out[k] < out[k - 1]:
            out[k] = out[k - 1]
    return [min(max(v, 0.0), 100.0) for v in out]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
