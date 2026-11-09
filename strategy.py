"""Walk-forward NQ daily direction classifier, executed intraday only.
Features use completed prior sessions: overnight gap, prior return/range, 3/10-session returns.
Refit logistic classifier every 60 sessions using last 1,000 completed sessions."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

FAMILY = "index-walkforward-daily-classifier"
PARAMS = {"train_sessions": 1000, "refit_sessions": 60, "warmup_sessions": 500, "confidence": 0.03, "entry_sec": 30900, "exit_sec": 53100, "risk_dollars": 250}
SYMBOLS = ("NQ",)


def entry_risk(panel):
    return {s: np.full(len(d), PARAMS["risk_dollars"], dtype=float) for s, d in panel.items()}


def signals(panel):
    result = {}
    for s, d in panel.items():
        out = np.zeros(len(d), dtype=np.int64)
        if s not in SYMBOLS:
            result[s] = out
            continue
        sec, sess = d.ct_sec.to_numpy(), d.session.astype(str)
        def agg(lo, hi, col, fn):
            return d[(sec >= lo) & (sec < hi)].groupby(sess[(sec >= lo) & (sec < hi)])[col].agg(fn)
        o, c = agg(30900, 30960, "open", "first"), agg(53000, 53100, "close", "last")
        pre, h, l = agg(30000, 30600, "close", "last"), agg(30600, 53400, "high", "max"), agg(30600, 53400, "low", "min")
        x = pd.concat({"o": o, "c": c, "pre": pre, "h": h, "l": l}, axis=1).dropna()
        ret = np.log(x.c / x.o)
        vol = ret.abs().rolling(PARAMS["refit_sessions"]).mean().shift(1)
        features = pd.DataFrame({"gap": np.log(x.o / x.pre), "prev": ret.shift(1), "range": np.log(x.h / x.l).shift(1), "p3": np.log(x.c).diff(3).shift(1), "p10": np.log(x.c).diff(10).shift(1)}).div(vol, axis=0).clip(-5, 5)
        ok = features.dropna().index
        f, y = features.loc[ok], (ret.loc[ok] > 0).astype(int)
        side = np.zeros(len(f), dtype=np.int64)
        model = None
        for i in range(PARAMS["warmup_sessions"], len(f)):
            if i % PARAMS["refit_sessions"] == 0:
                a = max(0, i - PARAMS["train_sessions"])
                model = LogisticRegression(C=0.1).fit(f.iloc[a:i], y.iloc[a:i])
            if model is not None:
                p = model.predict_proba(f.iloc[i:i + 1])[0, 1] - 0.5
                side[i] = int(np.sign(p)) if abs(p) >= PARAMS["confidence"] else 0
        mapped = sess.map(pd.Series(side, index=f.index.astype(str))).fillna(0).to_numpy(dtype=np.int64)
        active = (sec >= PARAMS["entry_sec"]) & (sec < PARAMS["exit_sec"])
        out[active] = mapped[active]
        result[s] = out
    return result
