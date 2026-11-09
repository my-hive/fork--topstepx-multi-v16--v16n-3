"""NQ prior-session RTH return reversal (overreaction of last day's index move), 1 contract."""
import numpy as np
import pandas as pd
FAMILY = "index-prior-day-reversal"
PARAMS = {"rth_start": 30600, "rth_end": 53400, "entry_sec": 34200, "exit_sec": 53000, "risk_points": 14, "vol_lookback": 20}
def signals(panel):
    result={}
    for s,d in panel.items():
        out=np.zeros(len(d),dtype=np.int64)
        if s!='NQ': result[s]=out; continue
        sec=d.ct_sec.to_numpy(); sess=d.session.astype(str)
        rth=(sec>=PARAMS['rth_start'])&(sec<PARAMS['rth_end'])
        g=d[rth].groupby(sess[rth])
        ret=g.close.last()-g.open.first()
        prior_range=(g.high.max()-g.low.min())
        high_vol=prior_range.shift(1) > prior_range.shift(1).rolling(PARAMS['vol_lookback']).median()
        side=-np.sign(ret).shift(1)*high_vol  # prior session return; prior-only volatility gate
        m=sess.map(pd.Series(side.to_numpy(),index=ret.index.astype(str))).fillna(0).to_numpy()
        a=(sec>=PARAMS['entry_sec'])&(sec<PARAMS['exit_sec'])
        out[a]=m[a].astype(np.int64); result[s]=out
    return result
def entry_risk(panel): return {s:np.full(len(d),PARAMS['risk_points']*20.0,dtype=float) for s,d in panel.items()}
