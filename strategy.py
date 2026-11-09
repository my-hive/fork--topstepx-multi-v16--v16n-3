"""Multi-index prior-RTH-session reversal, entered at next RTH open.
Mechanism adopted from v16n-5 (index-prior-day-reversal); changes: (a) drop the
prior-range gate, (b) enter at the RTH open instead of 09:30, (c) breadth across
NQ/YM/RTY so one index's bad streak is offset by the others.
Economic story: a large completed cash-session move overreacts; the next session's
opening flow absorbs the imbalance and partially retraces it intraday."""
import numpy as np
import pandas as pd

FAMILY = "index-prior-day-fade-breadth"
PARAMS = {"rth_start": 30600, "rth_end": 53400, "entry_sec": 30600, "exit_sec": 53000, "risk_points": 14}
SYMBOLS = ("NQ", "YM", "RTY")


def _run(d, symbol):
    sec = d.ct_sec.to_numpy()
    sess = d.session.astype(str)
    rth = (sec >= PARAMS["rth_start"]) & (sec < PARAMS["rth_end"])
    g = d[rth].groupby(sess[rth])
    move = (g.close.last() - g.open.first()).shift(1)
    side = -np.sign(move)
    lookup = pd.Series(side.to_numpy(), index=move.index.astype(str))
    target = np.zeros(len(d), dtype=np.int64)
    current = sess.map(lookup).fillna(0).to_numpy(dtype=np.int64)
    active = (sec >= PARAMS["entry_sec"]) & (sec < PARAMS["exit_sec"])
    target[active] = current[active]
    risk = np.zeros(len(d))
    risk[np.flatnonzero(np.diff(np.r_[0, target]) != 0)] = PARAMS["risk_points"] * 20.0
    return target, risk


def signals(panel):
    return {s: _run(d, s)[0] if s in SYMBOLS else np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}


def entry_risk(panel):
    return {s: _run(d, s)[1] if s in SYMBOLS else np.zeros(len(d)) for s, d in panel.items()}
