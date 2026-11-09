"""Cross-asset risk confirmation: NQ prior-RTH reversal executed ONLY when YM agrees on sign.
If NQ and YM had opposing prior-day moves, stay flat (mixed macro day).
Declares 14-point NQ risk ($280) inside $300 cap."""
import numpy as np
import pandas as pd

FAMILY = "index-cross-confirm-reversal"
PARAMS = {"entry_sec": 34200, "exit_sec": 53000, "vol_lookback": 20, "risk_points": 14}
TRADED = "NQ"


def _signal(panel):
    nq, ym = panel["NQ"], panel["YM"]
    sec = nq.ct_sec.to_numpy()
    sess = nq.session.astype(str)
    rth_nq = (sec >= 30600) & (sec < 53400)
    g_nq = nq[rth_nq].groupby(sess[rth_nq])
    ret_nq = g_nq.close.last() - g_nq.open.first()
    prior_range = g_nq.high.max() - g_nq.low.min()
    high_vol = prior_range.shift(1) > prior_range.shift(1).rolling(PARAMS["vol_lookback"]).median()

    sec_ym = ym.ct_sec.to_numpy()
    sess_ym = ym.session.astype(str)
    rth_ym = (sec_ym >= 30600) & (sec_ym < 53400)
    g_ym = ym[rth_ym].groupby(sess_ym[rth_ym])
    ret_ym = g_ym.close.last() - g_ym.open.first()

    aligned = np.sign(ret_nq) == np.sign(ret_ym).reindex(ret_nq.index)
    side = -np.sign(ret_nq).shift(1) * high_vol * aligned.shift(1)

    m = sess.map(pd.Series(side.to_numpy(), index=ret_nq.index.astype(str))).fillna(0).to_numpy()
    target = np.zeros(len(nq), dtype=np.int64)
    active = (sec >= PARAMS["entry_sec"]) & (sec < PARAMS["exit_sec"])
    target[active] = m[active].astype(np.int64)
    risk = np.zeros(len(nq))
    risk[np.flatnonzero(np.diff(np.r_[0, target]) != 0)] = PARAMS["risk_points"] * 20.0
    return target, risk


def signals(panel):
    target, _ = _signal(panel)
    return {s: target if s == TRADED else np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}


def entry_risk(panel):
    _, risk = _signal(panel)
    return {s: risk if s == TRADED else np.zeros(len(d)) for s, d in panel.items()}
