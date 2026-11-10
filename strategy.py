"""Turn-of-month Treasury drift: long 10-year Treasury (ZN) on calendar days >=28 or <=3.
08:30-14:50 CT hold; fixed dollar protective stop ($280) inside $300 cap.
Story: passive pension/index duration rebalancing into month-end creates structural buying."""
import numpy as np

FAMILY = "rates-turn-of-month-drift"
PARAMS = {"entry_sec": 30600, "exit_sec": 53400, "stop_risk": 280, "start_dom": 28, "end_dom": 3}
TRADED = "ZN"


def _run(d):
    sec = d.ct_sec.to_numpy()
    dom = d.index.tz_convert("America/Chicago").day.to_numpy()
    tom = (dom >= PARAMS["start_dom"]) | (dom <= PARAMS["end_dom"])
    active = tom & (sec >= PARAMS["entry_sec"]) & (sec < PARAMS["exit_sec"])
    target = active.astype(np.int64)
    risk = np.zeros(len(d))
    risk[np.flatnonzero(np.diff(np.r_[0, target]) > 0)] = PARAMS["stop_risk"]
    return target, risk


def signals(panel):
    return {s: _run(d)[0] if s == TRADED else np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}


def entry_risk(panel):
    return {s: _run(d)[1] if s == TRADED else np.zeros(len(d)) for s, d in panel.items()}
