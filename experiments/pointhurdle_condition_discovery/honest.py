"""Hyperparameters (eta, normalization) selected on the TRAINING dataset only,
then applied once to the held-out dataset.  This is the protocol-compliant version."""
import os, numpy as np, pandas as pd
R=r"E:\CODING\proj\covariate-trust-pilot\results"; EVS=os.path.join(R,"external_validity_screen")
OUT=os.path.join(R,"pointhurdle_recoverability"); SEED=20260813; rng=np.random.default_rng(SEED)
W=np.round(np.arange(0,1.0001,0.05),2)
RAW={"m5":os.path.join(EVS,"rule_replication","independent_raw_predictions.parquet"),
     "favorita":os.path.join(EVS,"favorita_independent","independent_raw_predictions.parquet")}
st=[]
for ds,p in RAW.items():
    d=pd.read_parquet(p); d=d[d["target_mask"]].copy(); d["dataset_id"]=ds; st.append(d)
st=pd.concat(st,ignore_index=True)
key=[st["dataset_id"],st["series_id"],st["origin"]]
rmse=lambda e: np.sqrt(e.groupby(key).mean())
P=pd.DataFrame({"loss_point":rmse((st["y_observed"]-st["point_mean_prediction"])**2),
                "loss_hurdle":rmse((st["y_observed"]-st["hurdle_mean_prediction"])**2)})
comb=pd.DataFrame({w:rmse((st["y_observed"]-(w*st["hurdle_mean_prediction"]+(1-w)*st["point_mean_prediction"]))**2) for w in W})
for f in (P,comb): f.index.names=["dataset_id","series_id","origin_id"]
P=P.sort_index(); comb=comb.reindex(P.index)
ORIG={ds:sorted(P.loc[ds].index.get_level_values("origin_id").unique()) for ds in ("m5","favorita")}
CV={ds:(comb.loc[ds].to_numpy(),{float(w):j for j,w in enumerate(comb.loc[ds].columns)}) for ds in ORIG}

def d3(ds,w_tr,eta,norm):
    p=P.loc[ds]; c=comb.loc[ds]; cv,cp_=CV[ds]; origins=ORIG[ds]
    pos={o:np.where(p.index.get_level_values("origin_id")==o)[0] for o in origins}
    out=np.empty(len(p))
    for k,o in enumerate(origins):
        ix=pos[o]; sl=p.iloc[ix]
        if k==0: wv=np.full(len(ix),w_tr)
        else:
            h=p[p.index.get_level_values("origin_id").isin(origins[:k])]
            a=h["loss_point"].groupby(level="series_id").mean().reindex(sl.index.get_level_values("series_id")).values
            b=h["loss_hurdle"].groupby(level="series_id").mean().reindex(sl.index.get_level_values("series_id")).values
            if norm=="per_series_scale":
                s=np.maximum((a+b)/2,1e-9); a,b=a/s,b/s
            elif norm=="per_origin_scale":
                s=max(np.nanmean(np.concatenate([a,b])),1e-9); a,b=a/s,b/s
            m=np.minimum(a,b); ea=np.exp(-eta*(b-m)); eb=np.exp(-eta*(a-m))
            wv=np.where(np.isfinite(ea/(ea+eb)),ea/(ea+eb),0.5)
        wq=np.clip(np.round(np.asarray(wv,float)/0.05)*0.05,0,1).round(2)
        out[ix]=[cv[ix[i],cp_[float(wq[i])]] for i in range(len(ix))]
    return pd.Series(out,index=p.index)

GRID=[(eta,norm) for eta in (0.5,2.0,8.0,32.0) for norm in ("raw","per_series_scale","per_origin_scale")]
rows=[]
for train_ds,test_ds in (("m5","favorita"),("favorita","m5")):
    w_tr=float(comb.loc[train_ds].mean().idxmin())
    # --- select eta, norm on the TRAINING dataset only
    best=None
    for eta,norm in GRID:
        L=d3(train_ds,float(comb.loc[test_ds].mean().idxmin()) if False else w_tr,eta,norm).mean()
        if best is None or L<best[0]: best=(L,eta,norm)
    _,eta_s,norm_s=best
    pte=P.loc[test_ds]; glv=pte["loss_hurdle"]      # train-domain global choice = hurdle both ways
    orc_conv=comb.loc[test_ds].min(axis=1)
    sel=d3(test_ds,w_tr,eta_s,norm_s)
    half=comb.loc[test_ds][0.5]
    # bootstrap, series clusters, all origins of a series together
    sid=pte.index.get_level_values("series_id"); us=sid.unique().to_numpy()
    pos={s:np.where(sid==s)[0] for s in us}
    gv,sv,ov,hv=glv.values,sel.values,orc_conv.values,half.values
    imp,cap,vs50=[],[],[]
    for _ in range(2000):
        pk=rng.choice(us,size=len(us),replace=True); ix=np.concatenate([pos[s] for s in pk])
        g,s_,o,h=gv[ix].mean(),sv[ix].mean(),ov[ix].mean(),hv[ix].mean()
        imp.append(100*(g-s_)/g); cap.append(100*(g-s_)/(g-o)); vs50.append(100*(h-s_)/h)
    lo,hi=np.percentile(imp,[2.5,97.5])
    rows.append(dict(train=train_ds,test=test_ds,eta_selected=eta_s,norm_selected=norm_s,
        w_train_domain=w_tr,rel_improve=np.mean(imp),ci_low=lo,ci_high=hi,
        excludes_zero=bool(lo>0 or hi<0),capture_convex_oracle=np.mean(cap),
        vs_5050=np.mean(vs50)))
r=pd.DataFrame(rows)
print("HYPERPARAMETERS SELECTED ON THE TRAINING DATASET ONLY (protocol-compliant)")
print(r.to_string(index=False,float_format=lambda x:"%.4f"%x))
r.to_csv(os.path.join(OUT,"policy_honest_hyperparam_selection.csv"),index=False)
print()
print("compare: previous run picked eta/norm AFTER seeing held-out results ->")
print("  that selection is not deployable and its numbers were optimistic.")
