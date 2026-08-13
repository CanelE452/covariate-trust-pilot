import os, numpy as np, pandas as pd
R=r"E:\CODING\proj\covariate-trust-pilot\results"; OUT=os.path.join(R,"pointhurdle_recoverability")
SEED=20260813; rng=np.random.default_rng(SEED)
panel=pd.read_parquet(os.path.join(OUT,"multi_origin_paired_panel.parquet"))
steps=pd.read_parquet(os.path.join(OUT,"_steps_cache.parquet"))
W=np.round(np.arange(0,1.0001,0.05),2)
cl={}
for w in W:
    yh=w*steps["hurdle_mean_prediction"]+(1-w)*steps["point_mean_prediction"]
    cl[w]=np.sqrt(((steps["y_observed"]-yh)**2).groupby([steps["dataset_id"],steps["series_id"],steps["origin"]]).mean())
comb=pd.DataFrame(cl); comb.index.names=["dataset_id","series_id","origin_id"]
P=panel.set_index(["dataset_id","series_id","origin_id"]).sort_index(); comb=comb.reindex(P.index)
ORIG={ds:sorted(d["origin_id"].unique()) for ds,d in panel.groupby("dataset_id")}
def d3(test_ds,w_tr,eta=8.0):
    te=P.loc[test_ds]; cte=comb.loc[test_ds]; origins=ORIG[test_ds]; out=[]; idx=[]
    for k,o in enumerate(origins):
        sl=te.xs(o,level="origin_id")
        if k==0: wv=np.full(len(sl),w_tr)
        else:
            h=te[te.index.get_level_values("origin_id").isin(origins[:k])]
            cp=h["loss_point"].groupby(level="series_id").mean().reindex(sl.index).values
            ch=h["loss_hurdle"].groupby(level="series_id").mean().reindex(sl.index).values
            m=np.minimum(cp,ch); a=np.exp(-eta*(ch-m)); b=np.exp(-eta*(cp-m))
            wv=np.where(np.isfinite(a/(a+b)),a/(a+b),0.5)
        wq=np.clip(np.round(wv/0.05)*0.05,0,1).round(2)
        cc=cte.xs(o,level="origin_id").reindex(sl.index)
        out.append(np.array([cc.iloc[i][wq[i]] for i in range(len(sl))]))
        idx.append(pd.MultiIndex.from_product([sl.index,[o]],names=["series_id","origin_id"]))
    v=pd.Series(np.concatenate(out),index=pd.MultiIndex.from_tuples(
        [(s,o) for ii,o in zip(idx,origins) for s in ii.get_level_values(0)],names=["series_id","origin_id"]))
    return v.reindex(te.index)
B=2000; rows=[]
for train_ds,test_ds in (("m5","favorita"),("favorita","m5")):
    ctr=comb.loc[train_ds]; w_tr=float(ctr.mean().idxmin())
    te=P.loc[test_ds]; gl=te["loss_hurdle"].values
    orc=te[["loss_point","loss_hurdle"]].min(axis=1).values
    pol=d3(test_ds,w_tr).values
    half=comb.loc[test_ds][0.5].values
    sids=te.index.get_level_values("series_id").unique().to_numpy()
    pos={s:np.where(te.index.get_level_values("series_id")==s)[0] for s in sids}
    dd,cc2=[],[]
    for _ in range(B):
        pk=rng.choice(sids,size=len(sids),replace=True); ix=np.concatenate([pos[s] for s in pk])
        g=gl[ix].mean(); p=pol[ix].mean(); o=orc[ix].mean()
        dd.append(100*(g-p)/g); cc2.append(100*(g-p)/(g-o))
    lo,hi=np.percentile(dd,[2.5,97.5]); clo,chi=np.percentile(cc2,[2.5,97.5])
    rows.append(dict(train=train_ds,test=test_ds,policy="D3 exp-weighted eta=8",
        rel_improve=np.mean(dd),ci_low=lo,ci_high=hi,excludes_zero=bool(lo>0 or hi<0),
        capture=np.mean(cc2),capture_lo=clo,capture_hi=chi,
        vs_5050_pp=100*(half.mean()-pol.mean())/half.mean()))
r=pd.DataFrame(rows); print(r.to_string(index=False,float_format=lambda x:"%.4f"%x))
r.to_csv(os.path.join(OUT,"policy_bootstrap_dynamic.csv"),index=False)
