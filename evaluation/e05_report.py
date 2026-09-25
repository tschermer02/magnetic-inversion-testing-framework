"""Build E05 JSON/CSV/Markdown suite comparisons with paired wins."""
from __future__ import annotations
import csv,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

METRICS={"true_body_susceptibility_mae_si":"lower","support_iou":"higher",
 "susceptibility_mae_si":"lower","susceptibility_rmse_si":"lower","susceptibility_relative_l2":"lower",
 "dice":"higher","precision":"higher","recall":"higher","predicted_to_true_volume_ratio":"one",
 "true_body_signed_bias_si":"zero","background_mae_si":"lower","center_depth_absolute_error_m":"lower",
 "tmi_mae_nt":"lower","tmi_rmse_nt":"lower","tmi_relative_l2":"lower","tmi_correlation":"higher",
 "predicted_to_observed_tmi_rms_ratio":"one","predicted_to_observed_tmi_peak_ratio":"one"}

def _read(path):
    with path.open(newline="",encoding="utf-8") as stream:return list(csv.DictReader(stream))
def _number(row,key):
    try:return float(row[key])
    except (ValueError,TypeError,KeyError):return None
def _score(value,direction):
    if direction=="lower":return value
    if direction=="higher":return -value
    return abs(value-(1 if direction=="one" else 0))

def build_suite_report(evaluation_root:Path,variants,parent_identifier,output:Path,*,split="test"):
    tables={identifier:_read(evaluation_root/identifier/split/"per_sample_metrics.csv") for identifier in variants}
    ids={name:[row["sample_id"] for row in rows] for name,rows in tables.items()}
    if len({tuple(value) for value in ids.values()})!=1:raise ValueError("E05 sample IDs/order differ across variants")
    aggregates=[]
    for name,rows in tables.items():
        for metric in METRICS:
            values=np.asarray([value for row in rows if (value:=_number(row,metric)) is not None and np.isfinite(value)])
            if values.size:aggregates.append({"variant":name,"metric":metric,"mean":float(values.mean()),
                "median":float(np.median(values)),"q10":float(np.quantile(values,.1)),"q90":float(np.quantile(values,.9)),"valid_count":len(values)})
    comparisons=[]
    requested=[("e05b","e05a"),("e05c","e05a"),("e05d",parent_identifier),("e05e",parent_identifier),
        ("e05f",parent_identifier),("e05f","e05d"),("e05f","e05e")]
    for candidate,reference in requested:
        if candidate not in tables or reference not in tables:continue
        for metric,direction in METRICS.items():
            wins=losses=ties=valid=0
            for left,right in zip(tables[candidate],tables[reference]):
                a,b=_number(left,metric),_number(right,metric)
                if a is None or b is None or not np.isfinite(a+b):continue
                valid+=1;difference=_score(a,direction)-_score(b,direction)
                if difference<-1e-12:wins+=1
                elif difference>1e-12:losses+=1
                else:ties+=1
            comparisons.append({"candidate":candidate,"reference":reference,"metric":metric,
                "paired_wins":wins,"paired_losses":losses,"paired_ties":ties,"valid_count":valid})
    historical={}
    mapping={"E01":Path("prediction_outputs/E01_soft_tversky_full/combined_test_metrics.csv"),
             "E04":Path("prediction_outputs/E04/combined_test_metrics.csv")}
    if split!="test":historical["status"]="not attempted outside test split"
    else:
        for label,path in mapping.items():
            if not path.is_file():historical[label]={"status":"unavailable"};continue
            rows=_read(path)
            if [row["sample_id"] for row in rows]!=ids[next(iter(ids))]:historical[label]={"status":"incompatible sample IDs"};continue
            shared=[metric for metric in METRICS if metric in rows[0]]
            historical[label]={"status":"compatible","shared_metric_means":{metric:float(np.mean([float(r[metric]) for r in rows])) for metric in shared}}
    result={"split":split,"sample_count":len(next(iter(tables.values()))),"sample_ids_verified":True,
        "units":{"susceptibility":"SI","TMI":"nT","depth":"m"},"metric_definitions_verified":True,
        "parent_identifier":parent_identifier,"aggregates":aggregates,"comparisons":comparisons,
        "historical":historical,"interpretation_warning":"Discuss amplitude, geometry, depth and TMI separately; mean relative L2 alone is insufficient.",
        "timestamp_utc":datetime.now(timezone.utc).isoformat()}
    output.mkdir(parents=True,exist_ok=True)
    (output/"suite_report.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    for name,rows in (("aggregate_metrics.csv",aggregates),("paired_comparisons.csv",comparisons)):
        with (output/name).open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    headline=[row for row in aggregates if row["metric"] in ("true_body_susceptibility_mae_si","support_iou","center_depth_absolute_error_m","tmi_rmse_nt")]
    lines=["# E05 suite comparison",f"",f"Split: `{split}`; {result['sample_count']} identical sample IDs; parent: `{parent_identifier}`.","",
        "Composite validation losses are not compared because objectives differ.","","| Variant | Metric | Mean | Median | Q10 | Q90 |","|---|---|---:|---:|---:|---:|"]
    lines += [f"| {r['variant']} | {r['metric']} | {r['mean']:.6g} | {r['median']:.6g} | {r['q10']:.6g} | {r['q90']:.6g} |" for r in headline]
    lines += ["","Interpret amplitude, geometry, depth fragmentation and magnetic consistency separately; do not declare improvement from mean relative L2 alone."]
    (output/"suite_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result
