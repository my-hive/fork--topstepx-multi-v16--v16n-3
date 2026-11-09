"""Stop-run sweep-and-revert: price pokes beyond prior RTH high/low then closes back inside the range -> fade, stop at sweep extreme.
One trade per session per symbol, hold to exit_sec."""
import numpy as np
import pandas as pd
from instruments import SPECS

FAMILY = "prior-extreme-sweep-revert"
PARAMS = {"entry_start": 31200, "entry_end": 46800, "exit_sec": 53000, "min_stop_frac": 0.1, "max_risk": 290}
SYMBOLS = ("MES", "MNQ", "MGC", "M6E")


def _run(d, s):
    sec, sess = d.ct_sec.to_numpy(), d.session.astype(str).to_numpy()
    close, hi, lo = d.close.to_numpy(), d.high.to_numpy(), d.low.to_numpy()
    rth = (sec >= 30600) & (sec < 53400)
    g = pd.DataFrame({"s": sess[rth], "h": hi[rth], "l": lo[rth]}).groupby("s", sort=True)
    ph, pl = g.h.max().shift(1).reindex(sess).to_numpy(), g.l.min().shift(1).reindex(sess).to_numpy()
    pv = SPECS[s].point_value
    out, risk = np.zeros(len(d), dtype=np.int64), np.zeros(len(d))
    bounds = np.flatnonzero(np.r_[True, sess[1:] != sess[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        rng = ph[a] - pl[a]
        if not np.isfinite(rng) or rng <= 0:
            continue
        hi_run, lo_run, side, stop = -np.inf, np.inf, 0, 0.0
        for j in range(a, b):
            if sec[j] >= PARAMS["exit_sec"]:
                break
            if side == 0 and PARAMS["entry_start"] <= sec[j] < PARAMS["entry_end"]:
                hi_run, lo_run = max(hi_run, hi[j]), min(lo_run, lo[j])
                if hi_run > ph[j] and close[j] < ph[j] and hi_run - close[j] >= PARAMS["min_stop_frac"] * rng * 0.1:
                    side, stop = -1, hi_run
                elif lo_run < pl[j] and close[j] > pl[j]:
                    side, stop = 1, lo_run
                if side:
                    risk[j:b] = abs(close[j] - stop) * pv
                    if risk[j] > PARAMS["max_risk"] or risk[j] == 0:
                        side = 2
                        continue
                    out[j] = side
            elif side in (1, -1):
                if (close[j] - stop) * side <= 0 and (hi[j] >= stop if side == -1 else lo[j] <= stop):
                    side = 2
                    continue
                out[j] = side
    return out, risk


def signals(panel):
    return {s: (_run(d, s)[0] if s in SYMBOLS else np.zeros(len(d), dtype=np.int64)) for s, d in panel.items()}


def entry_risk(panel):
    return {s: (_run(d, s)[1] if s in SYMBOLS else np.zeros(len(d))) for s, d in panel.items()}
