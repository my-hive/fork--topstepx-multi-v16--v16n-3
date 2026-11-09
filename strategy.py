"""Prior US-session gold overreaction: fade yesterday RTH direction in next US session."""
import numpy as np
import pandas as pd

FAMILY = "metals-overnight-gap-fade"
PARAMS = {"rth_start": 30600, "entry_sec": 34200, "exit_sec": 53000, "stop_risk": 290}
TRADED = "MGC"


def _run(d):
    sec = d.ct_sec.to_numpy()
    sess = d.session.astype(str)
    rth = (sec >= PARAMS["rth_start"]) & (sec < PARAMS["exit_sec"])
    g = d[rth].groupby(sess[rth])
    prior_move = (g.close.last() - g.open.first()).shift(1)
    side = -np.sign(prior_move)
    by_session = pd.Series(side.to_numpy(), index=prior_move.index.astype(str))
    target = np.zeros(len(d), dtype=np.int64)
    current = sess.map(by_session).fillna(0).to_numpy(dtype=np.int64)
    active = (sec >= PARAMS["entry_sec"]) & (sec < PARAMS["exit_sec"])
    target[active] = current[active]
    risk = np.zeros(len(d))
    risk[np.flatnonzero(np.diff(np.r_[0, target]) != 0)] = PARAMS["stop_risk"]
    return target, risk


def signals(panel):
    return {s: _run(d)[0] if s == TRADED else np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}


def entry_risk(panel):
    return {s: _run(d)[1] if s == TRADED else np.zeros(len(d)) for s, d in panel.items()}
