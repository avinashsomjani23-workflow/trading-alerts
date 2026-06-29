"""SCRATCH: how reliable is the sticky daily bias as a DIRECTIONAL call?
Not 'does it equal the structure' (tautology) — but 'when bias=up, does price
actually go up over the next N days?' Measured across 18y, all FX+gold.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, sys
sys.path.insert(0, ".")
from dealing_range import detect_swings

MT5="backtest/mt5_data"
PAIRS=["EURUSD","GBPUSD","USDJPY","USDCHF","NZDUSD","XAUUSD"]
HOR=[5,10,20]

def load(p):
    df=pd.read_csv(f"{MT5}/{p}_D1.csv"); df["ts"]=pd.to_datetime(df["time_server"]); return df

def replay(df):
    sw=sorted(detect_swings(df.rename(columns={"high":"High","low":"Low"}),
              lookback=3,min_leg_atr_mult=None),key=lambda s:s["idx"])
    n=len(df); out=[None]*n; swi=0; conf=[]; direction=None
    for t in range(n):
        while swi<len(sw) and sw[swi]["idx"]+3<=t:
            conf.append(sw[swi]); swi+=1
            highs=[x["price"] for x in conf if x["type"]=="high"]
            lows=[x["price"] for x in conf if x["type"]=="low"]
            if len(highs)>=2 and len(lows)>=2:
                HH=highs[-1]>highs[-2]; HL=lows[-1]>lows[-2]
                LH=highs[-1]<highs[-2]; LL=lows[-1]<lows[-2]
                if HH and HL: direction="up"
                elif LH and LL: direction="down"
        out[t]=direction
    return out

print("Sticky bias DIRECTIONAL accuracy (close[t+H] vs close[t]):")
agg={h:[0,0] for h in HOR}
for p in PAIRS:
    df=load(p); st=replay(df); C=df.close.values; n=len(df)
    line=f"  {p}: "
    for h in HOR:
        ok=tot=0
        for t in range(n-h):
            d=st[t]
            if d is None: continue
            up=C[t+h]>C[t]
            ok+= (up and d=="up") or ((not up) and d=="down"); tot+=1
        agg[h][0]+=ok; agg[h][1]+=tot
        line+=f"H{h}={100*ok/tot:.0f}%  "
    print(line)
print("  POOLED: "+"  ".join(f"H{h}={100*agg[h][0]/agg[h][1]:.1f}%" for h in HOR))

# with-bias vs counter: average forward move IN bias direction (ATR-normalised, H10)
print("\nForward 10d move in bias direction, ATR-normalised (edge, not hit-rate):")
for p in PAIRS:
    df=load(p); st=replay(df); C=df.close.values
    tr=pd.concat([(df.high-df.low),(df.high-df.close.shift()).abs(),(df.low-df.close.shift()).abs()],axis=1).max(axis=1)
    atr=tr.rolling(14).mean().values; n=len(df); vals=[]
    for t in range(n-10):
        d=st[t]
        if d is None or np.isnan(atr[t]) or atr[t]<=0: continue
        mv=(C[t+10]-C[t])/atr[t]
        vals.append(mv if d=="up" else -mv)
    vals=np.array(vals)
    print(f"  {p}: mean={vals.mean():+.3f} ATR  median={np.median(vals):+.3f}  %positive={100*(vals>0).mean():.0f}%")
