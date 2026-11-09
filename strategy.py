"""Prior-session RTH return reversal, high-range gated, on the equity-index complex.
One trade per session per symbol; fades prior RTH move when prior range > trailing median."""
import numpy as np
import pandas as pd

FAMILY = "index-prior-day-reversal"
PARAMS = {"rth_start": 30600, "rth_end": 53400, "entry_sec": 34200, "exit_sec": 53000, "vol_lookback": 20}
SYMBOLS = {"ES": 5.0, "YM": 50.0, "RTY": 5.0}   # symbol: protective stop points (<= $300/contract)


def signals(panel):
    result = {}
    for s, d in panel.items():
        out = np.zeros(len(d), dtype=np.int64)
        if s not in SYMBOLS:
            result[s] = out
            continue
        sec = d.ct_sec.to_numpy()
        sess = d.session.astype(str)
        rth = (sec >= PARAMS["rth_start"]) & (sec < PARAMS["rth_end"])
        g = d[rth].groupby(sess[rth])
        ret = g.close.last() - g.open.first()
        rng = g.high.max() - g.low.min()
        high_vol = rng.shift(1) > rng.shift(1).rolling(PARAMS["vol_lookback"]).median()
        side = -np.sign(ret).shift(1) * high_vol
        m = sess.map(pd.Series(side.to_numpy(), index=ret.index.astype(str))).fillna(0).to_numpy()
        a = (sec >= PARAMS["entry_sec"]) & (sec < PARAMS["exit_sec"])
        out[a] = m[a].astype(np.int64)
        result[s] = out
    return result


def entry_risk(panel):
    from instruments import SPECS
    return {s: np.full(len(d), SYMBOLS.get(s, 1.0) * SPECS[s].point_value, dtype=float)
            for s, d in panel.items()}
