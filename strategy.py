"""Asia-to-London FX reversal: fade completed 18:00-07:00 CT move, hold London morning.
One trade per session, causal fixed-dollar protective stop."""
import numpy as np
from instruments import SPECS

FAMILY = "fx-asia-london-reversal"
PARAMS = {"asia_start": 64800, "entry_sec": 25200, "exit_sec": 36000, "stop_risk": 250}
SYMBOLS = ("M6E", "M6B", "M6A")

def _run(d, symbol):
    sec = d.ct_sec.to_numpy()
    session = d.session.astype(str).to_numpy()
    close, high, low = d.close.to_numpy(), d.high.to_numpy(), d.low.to_numpy()
    target, risk = np.zeros(len(d), dtype=np.int64), np.zeros(len(d))
    pv = SPECS[symbol].point_value
    bounds = np.flatnonzero(np.r_[True, session[1:] != session[:-1], True])
    for a, b in zip(bounds[:-1], bounds[1:]):
        start = np.flatnonzero(sec[a:b] >= PARAMS["asia_start"])
        entry = np.flatnonzero((sec[a:b] >= PARAMS["entry_sec"]) & (sec[a:b] < PARAMS["asia_start"]))
        if not len(start) or not len(entry):
            continue
        i0, i = a + start[0], a + entry[0]
        if i <= i0 or sec[i] >= PARAMS["exit_sec"]:
            continue
        side = -int(np.sign(close[i - 1] - close[i0]))
        if not side:
            continue
        stop = close[i - 1] - side * PARAMS["stop_risk"] / pv
        risk[i:b] = PARAMS["stop_risk"]
        for j in range(i, b):
            if sec[j] >= PARAMS["exit_sec"]:
                break
            if (side > 0 and low[j] <= stop) or (side < 0 and high[j] >= stop):
                break
            target[j] = side
    return target, risk

def signals(panel):
    return {s: _run(d, s)[0] if s in SYMBOLS else np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}

def entry_risk(panel):
    return {s: _run(d, s)[1] if s in SYMBOLS else np.zeros(len(d)) for s, d in panel.items()}
