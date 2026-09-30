"""Cross-variant E06 numerical report and required diagnostic figures."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np

METRICS={"true_body_susceptibility_mae_si":"lower","true_body_susceptibility_rmse_si":"lower",
 "true_body_signed_bias_si":"zero","background_mae_si":"lower","susceptibility_mae_si":"lower",
 "susceptibility_rmse_si":"lower","iou":"higher","dice":"higher","precision":"higher","recall":"higher",
 "predicted_to_true_volume_ratio":"one","center_depth_absolute_error_m":"lower",
 "top_depth_absolute_error_m":"lower","bottom_depth_absolute_error_m":"lower","thickness_absolute_error_m":"lower",
 "tmi_mae_nt":"lower","tmi_rmse_nt":"lower","tmi_relative_l2":"lower","tmi_correlation":"higher"}

def _read(path):
    with path.open(newline="",encoding="utf-8") as stream:return list(csv.DictReader(stream))
def _number(row,key):
    try:return float(row[key])
    except (KeyError,TypeError,ValueError):return None
def _score(x,d):return x if d=="lower" else (-x if d=="higher" else abs(x-(1 if d=="one" else 0)))

def build_report(evaluation_root:Path,identifiers,output:Path,checkpoint="primary"):
    tables={v:_read(evaluation_root/v/checkpoint/"per_sample_metrics.csv") for v in identifiers};output.mkdir(parents=True,exist_ok=True)
    if len({tuple(r["sample_id"] for r in rows) for rows in tables.values()})!=1:raise ValueError("E06 sample IDs/order differ")
    aggregates=[]
    for variant,rows in tables.items():
        for metric in METRICS:
            values=np.array([x for r in rows if (x:=_number(r,metric)) is not None and np.isfinite(x)])
            if values.size:aggregates.append({"variant":variant,"metric":metric,"mean":float(values.mean()),"median":float(np.median(values)),
                "q10":float(np.quantile(values,.1)),"q90":float(np.quantile(values,.9)),"valid_count":int(values.size),"failure_count":len(rows)-int(values.size)})
    pairs=[("e06b","e06a"),("e06c","e06a"),("e06d","e06a"),("e06d","e06b"),("e06d","e06c")];comparisons=[]
    for candidate,reference in pairs:
        for metric,direction in METRICS.items():
            wins=losses=ties=valid=0
            for a,b in zip(tables[candidate],tables[reference]):
                av,bv=_number(a,metric),_number(b,metric)
                if av is None or bv is None or not np.isfinite(av+bv):continue
                valid+=1;difference=_score(av,direction)-_score(bv,direction)
                if difference<-1e-12:wins+=1
                elif difference>1e-12:losses+=1
                else:ties+=1
            comparisons.append({"candidate":candidate,"reference":reference,"metric":metric,"paired_wins":wins,"paired_losses":losses,"paired_ties":ties,"valid_count":valid})
    for name,rows in (("aggregate_metrics.csv",aggregates),("paired_comparisons.csv",comparisons)):
        with (output/name).open("w",newline="",encoding="utf-8") as stream:w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    historical={}
    if checkpoint=="primary":
        for old in ("e05c","e05e"):
            path=Path("prediction_outputs/E05/seed_20260727")/old/"test/per_sample_metrics.csv"
            if not path.is_file():historical[old]={"status":"unavailable"};continue
            rows=_read(path)
            if [r["sample_id"] for r in rows]!=[r["sample_id"] for r in next(iter(tables.values()))]:historical[old]={"status":"incompatible sample IDs"};continue
            historical[old]={"status":"compatible","checkpoint_selection":"historical body-only validation MAE",
                "means":{m:float(np.mean([float(r[m]) for r in rows if r.get(m) not in (None,"")])) for m in METRICS if m in rows[0]}}
    result={"checkpoint":checkpoint,"sample_count":len(next(iter(tables.values()))),"canonical_iou":"iou","aggregates":aggregates,"comparisons":comparisons,
        "historical":historical,"historical_comparison_note":"E05C/E use compatible test samples and units but a different body-only checkpoint-selection rule."}
    (output/"suite_report.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    _figures(tables,output)
    return result

def _figures(tables,output):
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    colors={"e06a":"black","e06b":"tab:blue","e06c":"tab:orange","e06d":"tab:green"}
    fig,axis=plt.subplots(figsize=(7,7))
    for name,rows in tables.items():axis.scatter([float(r["true_mean_susceptibility_si"]) for r in rows],[float(r["predicted_mean_susceptibility_on_true_body_si"]) for r in rows],s=15,alpha=.6,label=name,color=colors[name])
    axis.plot([0,.1],[0,.1],"--",color="gray");axis.set(xlim=(0,.1),ylim=(0,.1),aspect="equal",xlabel="True mean body susceptibility (SI)",ylabel="Predicted mean on true-body mask (SI)");axis.legend();fig.savefig(output/"body_susceptibility_1to1.png",dpi=170);plt.close(fig)
    for ykey,ylabel,filename in (("true_body_signed_bias_si","Signed body bias (SI)","body_bias_vs_true.png"),("true_body_susceptibility_mae_si","Body MAE (SI)","body_mae_vs_true.png")):
        fig,axis=plt.subplots(figsize=(8,5))
        for name,rows in tables.items():axis.scatter([float(r["true_mean_susceptibility_si"]) for r in rows],[float(r[ykey]) for r in rows],s=14,alpha=.55,label=name)
        axis.axhline(0,color="gray",lw=1);axis.set(xlabel="True mean body susceptibility (SI)",ylabel=ylabel);axis.legend();fig.savefig(output/filename,dpi=170);plt.close(fig)
    base=tables["e06a"]
    keys=(("true_body_susceptibility_mae_si","Body MAE change"),("iou","IoU change"),("background_mae_si","Background MAE change"),("tmi_rmse_nt","TMI RMSE change"))
    fig,axes=plt.subplots(2,2,figsize=(12,9),constrained_layout=True)
    for axis,(key,title) in zip(axes.ravel(),keys):
        for name in ("e06b","e06c","e06d"):
            delta=[float(a[key])-float(b[key]) for a,b in zip(tables[name],base)];axis.plot(delta,".",alpha=.55,label=name)
        axis.axhline(0,color="black",lw=1);axis.set_title(title+" vs E06A");axis.set_xlabel("Test sample index");axis.legend()
    fig.savefig(output/"paired_changes_vs_e06a.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    for axis,key,title in zip(axes,("top_depth_absolute_error_m","bottom_depth_absolute_error_m","thickness_absolute_error_m"),("Top","Bottom","Thickness")):
        axis.boxplot([[float(r[key]) for r in tables[v] if r[key]] for v in tables],tick_labels=list(tables));axis.set_title(title+" absolute error");axis.set_ylabel("m")
    fig.savefig(output/"depth_boundary_errors.png",dpi=170);plt.close(fig)
