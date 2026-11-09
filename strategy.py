"""Prior-session range breakout (first break only), one trade/session, real protective stop.
Long when price first closes above prior-session high (US hours), short below prior low. Stop at entry -/+ stop_frac*prior range.
Hold to exit_sec. Declared risk = stop distance * point value (<= $300 enforced by skipping oversized stops)."""
import numpy as np
import pandas as pd
from instruments import SPECS

FAMILY = "prior-range-breakout-stop"
PARAMS = {"stop_frac": 0.5, "entry_start": 30600, "entry_end": 46800, "exit_sec": 53000, "max_risk": 290}
SYMBOLS = ("MGC", "M6E", "M6B", "M6A")


def _prep(d, s):
    sec, sess = d.ct_sec.to_numpy(), d.session.astype(str).to_numpy()
    close, hi, lo = d.close.to_numpy(), d.high.to_numpy(), d.low.to_numpy()
    rth = (sec >= 30600) & (sec < 53400)
    g = pd.DataFrame({"s": sess[rth], "h": hi[rth], "l": lo[rth]}).groupby("s", sort=True)
    ph, pl = g.h.max().shift(1), g.l.min().shift(1)
    return sec, sess, close, ph.reindex(sess).to_numpy(), pl.reindex(sess).to_numpy()


def _run(d, s):
    sec, sess, close, ph, pl = _prep(d, s)
    pv = SPECS[s].point_value
    out = np.zeros(len(d), dtype=np.int64)
    risk = np.zeros(len(d))
    start = 0
    for i in range(1, len(d) + 1):
        if i < len(d) and sess[i] == sess[start]:
            continue
        idx = np.arange(start, i); start = i
        rng = ph[idx[0]] - pl[idx[0]]
        if not np.isfinite(rng) or rng <= 0:
            continue
        stop_d = PARAMS["stop_frac"] * rng
        r = stop_d * pv
        risk[idx] = r
        if r > PARAMS["max_risk"]:
            continue
        side, stop = 0, 0.0
        for j in idx:
            if sec[j] >= PARAMS["exit_sec"]:
                break
            if side == 0 and PARAMS["entry_start"] <= sec[j] < PARAMS["entry_end"] and not done_flag(out, idx, j):
                if close[j] > ph[j]:
                    side, stop = 1, close[j] - stop_d
                elif close[j] < pl[j]:
                    side, stop = -1, close[j] + stop_d
            elif side != 0 and (close[j] - stop) * side <= 0:
                out[j] = 0
                side = 2  # stopped, no re-entry
                continue
            if side in (1, -1):
                out[j] = side
    return out, risk


def done_flag(out, idx, j):
    return False


def signals(panel):
    return {s: (_run(d, s)[0] if s in SYMBOLS else np.zeros(len(d), dtype=np.int64)) for s, d in panel.items()}


def entry_risk(panel):
    return {s: (_run(d, s)[1] if s in SYMBOLS else np.zeros(len(d))) for s, d in panel.items()}
