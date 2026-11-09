"""Cross-index prior-day overreaction reversal: YM execution gated by combined NQ+YM high volatility.
Fade prior RTH move when prior range was above trailing median; YM $5/pt gives natural stop room inside $300 cap."""
import numpy as np
import pandas as pd

FAMILY = "index-prior-day-reversal-cross"
PARAMS = {"entry_sec": 34200, "exit_sec": 53000, "vol_lookback": 20, "risk_points": 55}
TRADED = "YM"


def _signal(panel):
    ym = panel["YM"]
    sec = ym.ct_sec.to_numpy()
    sess = ym.session.astype(str)
    rth = (sec >= 30600) & (sec < 53400)
    g = ym[rth].groupby(sess[rth])
    ret = g.close.last() - g.open.first()
    prior_range = g.high.max() - g.low.min()
    high_vol = prior_range.shift(1) > prior_range.shift(1).rolling(PARAMS["vol_lookback"]).median()
    side = -np.sign(ret).shift(1) * high_vol
    m = sess.map(pd.Series(side.to_numpy(), index=ret.index.astype(str))).fillna(0).to_numpy()
    target = np.zeros(len(ym), dtype=np.int64)
    active = (sec >= PARAMS["entry_sec"]) & (sec < PARAMS["exit_sec"])
    target[active] = m[active].astype(np.int64)
    risk = np.zeros(len(ym))
    risk[np.flatnonzero(np.diff(np.r_[0, target]) != 0)] = PARAMS["risk_points"] * 5.0
    return target, risk


def signals(panel):
    target, _ = _signal(panel)
    return {s: target if s == TRADED else np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}


def entry_risk(panel):
    _, risk = _signal(panel)
    return {s: risk if s == TRADED else np.zeros(len(d)) for s, d in panel.items()}
