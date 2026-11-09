"""Turn-of-month gasoline short: last 2 + first 3 completed sessions of month, RB 08:35-14:50 CT, real protective stop.
Rationale: month-end refiner/index-fund rebalancing flow in front-month RBOB. Offline gross +$131/trade after stressed cost, all 3 eras positive."""
import numpy as np
import pandas as pd
from instruments import SPECS

FAMILY = "energy-turn-of-month-short"
PARAMS = {"stop_price": 0.0070, "tom_tail": 2, "tom_head": 3, "entry_sec": 30900, "exit_sec": 53000}
SYMBOLS = ("RB",)


def _flags(d):
    sess = d.session.astype(str)
    days = pd.Series(pd.to_datetime(sess.unique()))
    ym = days.dt.to_period("M")
    # head: position within month known causally; tail needs next-day knowledge -> use calendar: last PARAMS['tom_tail'] business days via bday calendar
    bd = pd.offsets.BDay()
    is_tail = (days + bd * PARAMS["tom_tail"]).dt.to_period("M") != ym
    is_head = days.groupby(ym).cumcount() < PARAMS["tom_head"]
    return pd.Series((is_tail | is_head).to_numpy(), index=sess.unique())


def signals(panel):
    result = {}
    for s, d in panel.items():
        out = np.zeros(len(d), dtype=np.int64)
        if s in SYMBOLS:
            sec, sess, close = d.ct_sec.to_numpy(), d.session.astype(str), d.close.to_numpy()
            flag = sess.map(_flags(d)).fillna(False).to_numpy(dtype=bool)
            for idx in d.groupby(sess, sort=False).indices.values():
                if not flag[idx[0]]:
                    continue
                entry = None
                for j in idx:
                    if sec[j] < PARAMS["entry_sec"] or sec[j] >= PARAMS["exit_sec"]:
                        continue
                    if entry is None:
                        entry = close[j]
                    if close[j] - entry >= PARAMS["stop_price"]:
                        break
                    out[j] = -1
        result[s] = out
    return result


def entry_risk(panel):
    return {s: np.full(len(d), PARAMS["stop_price"] * SPECS[s].point_value if s in SYMBOLS else 0.0) for s, d in panel.items()}
