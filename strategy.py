"""v16n-3: energy opening-move continuation, vol-gated, one trade/session, protective stop.
Move = 08:30 open -> 09:00 close (CT). Trade only if |move| > z * trailing mean |move| (prior sessions).
Hold to exit_ct. Stop = stop_mult * trailing mean |move| from entry; declared as entry risk."""
import numpy as np
from instruments import SPECS

FAMILY = "energy-open-move-continuation"
PARAMS = {"z": 1.5, "stop_mult": 2.0, "window": 40, "exit_ct": 14 * 3600 + 50 * 60}
SYMBOLS = ("QM", "QG")
OPEN, DECIDE = 8 * 3600 + 30 * 60, 9 * 3600


def _state(df):
    sec = df.ct_sec.to_numpy()
    close, opn = df.close.to_numpy(), df.open.to_numpy()
    groups = list(df.groupby("session", sort=False).indices.values())
    info = []
    for g in groups:
        s = sec[g]
        a = np.flatnonzero((s >= OPEN) & (s < DECIDE))
        d = np.flatnonzero((s >= DECIDE) & (s < 15 * 3600))
        if not len(a) or not len(d) or d[0] <= a[0] or s[d[0]] > DECIDE + 120:
            info.append(None)
            continue
        info.append((g, g[d[0]], opn[g[a[0]]], close[g[d[0]]]))
    return sec, close, info


def signals(panel):
    result = {}
    for symbol, df in panel.items():
        out = np.zeros(len(df), dtype=np.int64)
        if symbol in SYMBOLS:
            sec, close, info = _state(df)
            hist = []
            for it in info:
                if it is None:
                    continue
                g, di, o, c = it
                mv = abs(c - o)
                if len(hist) >= PARAMS["window"]:
                    avg = np.mean(hist[-PARAMS["window"]:])
                    if avg > 0 and mv > PARAMS["z"] * avg and c != o:
                        side = int(np.sign(c - o))
                        stop = c - side * PARAMS["stop_mult"] * avg
                        for j in g[g > di]:
                            if sec[j] >= PARAMS["exit_ct"] or (close[j] - stop) * side <= 0:
                                break
                            out[j] = side
                        out[di] = side
                hist.append(mv)
        result[symbol] = out
    return result


def entry_risk(panel):
    result = {}
    for symbol, df in panel.items():
        r = np.zeros(len(df))
        if symbol in SYMBOLS:
            sec, close, info = _state(df)
            hist = []
            for it in info:
                if it is None:
                    continue
                g, di, o, c = it
                mv = abs(c - o)
                if len(hist) >= PARAMS["window"]:
                    avg = np.mean(hist[-PARAMS["window"]:])
                    r[g] = PARAMS["stop_mult"] * avg * SPECS[symbol].point_value
                hist.append(mv)
        result[symbol] = r
    return result
