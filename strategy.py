"""Treasury auction-day drift: long 5-year note (ZF) Tue-Thu 12:00-15:00 CT.
Auction allocation leaves dealers short duration; covering creates modest afternoon bid."""
import numpy as np

FAMILY = "rates-auction-day-drift"
PARAMS = {"entry_sec": 43200, "exit_sec": 54000, "stop_risk": 280, "first_auction_dow": 1, "last_auction_dow": 3}
TRADED = "ZF"


def _run(d):
    sec = d.ct_sec.to_numpy()
    dow = d.index.tz_convert("America/Chicago").day_of_week.to_numpy()
    auction = (dow >= PARAMS["first_auction_dow"]) & (dow <= PARAMS["last_auction_dow"])
    active = auction & (sec >= PARAMS["entry_sec"]) & (sec < PARAMS["exit_sec"])
    target = active.astype(np.int64)
    risk = np.zeros(len(d))
    risk[np.flatnonzero(np.diff(np.r_[0, target]) > 0)] = PARAMS["stop_risk"]
    return target, risk


def signals(panel):
    return {s: _run(d)[0] if s == TRADED else np.zeros(len(d), dtype=np.int64) for s, d in panel.items()}


def entry_risk(panel):
    return {s: _run(d)[1] if s == TRADED else np.zeros(len(d)) for s, d in panel.items()}
