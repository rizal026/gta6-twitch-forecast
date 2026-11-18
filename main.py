"""Forecast GTA VI's launch-day Twitch viewers, hour by hour.

For each hour h = 2..24 of the UTC release day, the program fits, on the
past launches in the case's own comparables, a small log-linear model of
that hour's viewers:

    log(y_h) = a + b*log(p1) + c*log(base) + d*log(ch1) + e*dst

where p1 is the launch's hour-1 print, base its mean hourly viewers over
the 7 days before release day, ch1 its hour-1 channel count, and dst a
0/1 flag for US daylight saving on the release date (the US midnight
unlock lands an hour later once daylight saving ends). The early hours
track hour 1 almost one-for-one (b near 1); the daytime peak barely does
(b near 0) and is instead carried by the pre-release baseline, which is
why the baseline is in the model. Quantiles come from the fit's own
residuals (empirical quantiles at the five scored levels), and the median
is blended 20% toward the plain median-ratio profile, which steadies the
hours where the fit has few comparables. Everything is computed from the
case file at run time, so the same program generalises to the final case.
Deterministic: no randomness, no network, standard library only.

Interface (same as the kit's starter): read every <id>.in in the input
directory, write one <id>.out per case with {"hours": [{"hour": h, "q":
[q05, q25, q50, q75, q95]}]} for hours 2..24. Input directory is
$MAKEFUN_INPUT_DIR (else /input), output $MAKEFUN_OUTPUT_DIR (else
/output). Runs in well under a second per case on one CPU.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Median blend toward the plain ratio profile, and the minimum number of
# comparables a per-hour fit needs before falling back to that profile.
RATIO_BLEND = 0.2
MIN_FIT_ROWS = 12
MIN_DAY_HOURS = 20
BASE_WINDOW_HOURS = 7 * 24


def parse_ts(s):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def us_dst_on(dt):
    """True when US daylight saving is in effect on dt's date.

    Second Sunday of March to first Sunday of November.
    """
    year = dt.year

    def nth_sunday(month, n):
        first = datetime(year, month, 1, tzinfo=timezone.utc)
        days = (6 - first.weekday()) % 7
        return first + timedelta(days=days + 7 * (n - 1))

    return nth_sunday(3, 2) <= dt < nth_sunday(11, 1)


def round2(x):
    """x rounded to 2 decimals, half up (whole numbers come back as int)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def quantile(sorted_vals, tau):
    """Linear-interpolation quantile of an already sorted list."""
    n = len(sorted_vals)
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def ols(x_rows, y_vals):
    """Least squares y ~ x (rows already include no intercept; one is added).

    Returns (coefficients, residuals) or (None, None) when singular.
    Gaussian elimination with partial pivoting; deterministic.
    """
    x = [[1.0] + list(r) for r in x_rows]
    k = len(x[0])
    n = len(x)
    a = [[sum(x[r][i] * x[r][j] for r in range(n)) for j in range(k)] for i in range(k)]
    b = [sum(x[r][i] * y_vals[r] for r in range(n)) for i in range(k)]
    m = [a[i][:] + [b[i]] for i in range(k)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(m[r][c]))
        m[c], m[piv] = m[piv], m[c]
        if abs(m[c][c]) < 1e-12:
            return None, None
        for r in range(k):
            if r != c:
                f = m[r][c] / m[c][c]
                for cc in range(c, k + 1):
                    m[r][cc] -= f * m[c][cc]
    coef = [m[i][k] / m[i][i] for i in range(k)]
    resid = [y_vals[r] - sum(coef[j] * x[r][j] for j in range(k)) for r in range(n)]
    return coef, resid


def comparable_stats(comp):
    """One comparable's release-day profile, pre-release baseline, channels, DST.

    Returns (profile {hour: viewers}, base, channels_hour1, dst) or None
    when the comparable has no usable release day.
    """
    day_start = parse_ts(comp["day_start"])
    profile = {}
    ch1 = None
    for row in comp.get("hourly", []):
        viewers = row.get("viewers")
        if viewers is None or viewers <= 0:
            continue
        t = parse_ts(row["from"])
        dh = (t - day_start).total_seconds() / 3600.0
        if 0 <= dh < 24:
            profile[int(dh) + 1] = viewers
            if dh < 1:
                ch1 = row.get("channels")
    if 1 not in profile or len(profile) < MIN_DAY_HOURS:
        return None
    pre = [r["viewers"] for r in comp.get("hourly", [])
           if -BASE_WINDOW_HOURS <= (parse_ts(r["from"]) - day_start).total_seconds() / 3600.0 < 0
           and r.get("viewers") is not None and r["viewers"] > 0]
    base = sum(pre) / len(pre) if pre else None
    dst = 1.0 if us_dst_on(day_start) else 0.0
    return profile, base, ch1, dst


def case_baseline(case):
    """The case's own mean hourly viewers over the 7 days before day_start."""
    day_start = parse_ts(case["day_start"])
    pre = [r["viewers"] for r in case.get("history_hourly", [])
           if -BASE_WINDOW_HOURS <= (parse_ts(r["from"]) - day_start).total_seconds() / 3600.0 < 0
           and r.get("viewers") is not None and r["viewers"] > 0]
    return sum(pre) / len(pre) if pre else None


def ratio_quantiles(stats, p1, h):
    """Quantiles of log(y_h / y_1) across comparables, anchored on p1."""
    lrs = sorted(math.log(prof[h] / prof[1])
                 for prof, _, _, _ in stats if h in prof and 1 in prof)
    if not lrs:
        return None
    return [max(1.0, p1 * math.exp(quantile(lrs, t))) for t in LEVELS]


def persistence_quantiles(p1, h):
    """House-baseline-style fallback: flat median, widening spread."""
    return [max(1.0, p1 * math.exp(z * 0.15 * math.sqrt(h - 1))) for z in Z]


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    ch1 = first.get("channels")
    base_t = case_baseline(case)
    dst_t = 1.0 if us_dst_on(parse_ts(case["day_start"])) else 0.0

    stats = []
    for comp in case.get("comparables", []):
        s = comparable_stats(comp)
        if s is not None:
            stats.append(s)

    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        if h == 1:
            continue
        # Per-hour regression on the comparables that have this hour.
        rows = []
        for prof, base_c, ch_c, dst_c in stats:
            if h not in prof or not base_c or not ch_c:
                continue
            rows.append(([math.log(prof[1]), math.log(base_c), math.log(ch_c), dst_c],
                         math.log(prof[h])))
        quantiles = None
        if len(rows) >= MIN_FIT_ROWS:
            coef, resid = ols([r[0] for r in rows], [r[1] for r in rows])
            if coef is not None:
                xt = [math.log(p1),
                      math.log(base_t) if base_t else sum(r[0][1] for r in rows) / len(rows),
                      math.log(ch1) if ch1 else sum(r[0][2] for r in rows) / len(rows),
                      dst_t]
                mu = coef[0] + sum(coef[j + 1] * xt[j] for j in range(4))
                sres = sorted(resid)
                qs = [mu + quantile(sres, t) for t in LEVELS]
                # Blend the median toward the plain ratio profile.
                rq = ratio_quantiles(stats, p1, h)
                if rq is not None:
                    med_r = math.log(max(rq[2], 1.0))
                    mu_b = (1 - RATIO_BLEND) * mu + RATIO_BLEND * med_r
                    qs = [mu_b + (q - mu) for q in qs]
                quantiles = [max(1.0, math.exp(q)) for q in qs]
        if quantiles is None:
            quantiles = ratio_quantiles(stats, p1, h)
        if quantiles is None:
            quantiles = persistence_quantiles(p1, h)
        # Guard against wild extrapolation: no quantile may leave the band
        # spanned by the comparables' observed ratios for this hour, widened
        # 2x either way. It never binds on the practice cases; it only stops
        # an extreme hour-1 print (a launch far bigger than any comparable)
        # from pushing a forecast orders of magnitude off the observed range.
        ratios = sorted(prof[h] / prof[1]
                        for prof, _, _, _ in stats if h in prof and 1 in prof)
        if ratios:
            lo = p1 * ratios[0] / 2.0
            hi = p1 * ratios[-1] * 2.0
            quantiles = [min(hi, max(lo, q)) for q in quantiles]
        # Round, clamp, and enforce non-decreasing order.
        out_q = []
        prev = 0.0
        for q in quantiles:
            v = min(1e9, max(1.0, round2(q)))
            v = max(v, prev)
            out_q.append(v)
            prev = v
        hours.append({"hour": h, "q": out_q})
    hours.sort(key=lambda r: r["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
