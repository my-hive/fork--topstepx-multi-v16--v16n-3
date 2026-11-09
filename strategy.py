"""v16n-3: FX daily time-series momentum, executed as one intraday US-hours trade per session.
Side = sign of (last completed session close / close N sessions earlier). Hold entry_ct..exit_ct CT."""
import numpy as np

FAMILY = "fx-daily-tsmom-intraday"
PARAMS = {"n_sessions": 20, "entry_ct": 8 * 3600 + 30 * 60, "exit_ct": 15 * 3600, "risk_dollars": 250}
SYMBOLS = ("6E", "6B", "6J", "6A", "6C")


def entry_risk(panel):
    return {s: np.full(len(df), PARAMS["risk_dollars"], dtype=float) for s, df in panel.items()}


def signals(panel):
    result = {}
    n = PARAMS["n_sessions"]
    for symbol, df in panel.items():
        out = np.zeros(len(df), dtype=np.int64)
        if symbol in SYMBOLS:
            sec = df.ct_sec.to_numpy()
            groups = list(df.groupby("session", sort=False).indices.values())
            last = np.array([df.close.to_numpy()[g[-1]] for g in groups])
            for k, g in enumerate(groups):
                if k <= n:
                    continue
                side = int(np.sign(last[k - 1] - last[k - 1 - n]))
                hold = (sec[g] >= PARAMS["entry_ct"]) & (sec[g] < PARAMS["exit_ct"])
                out[g[hold]] = side
        result[symbol] = out
    return result
