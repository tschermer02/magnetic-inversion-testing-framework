"""Compare E01 and E04 on identically evaluated E01 test predictions."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np

METRICS=("susceptibility_mae_si","susceptibility_rmse_si","susceptibility_relative_l2",
         "support_iou","tmi_mae_nt","tmi_rmse_nt","tmi_relative_l2","tmi_correlation")

def _load(directory):
    with (directory/"combined_test_metrics.csv").open(newline="",encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    return rows,{metric:float(np.mean([float(row[metric]) for row in rows])) for metric in METRICS}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e01",type=Path,required=True); parser.add_argument("--e04",type=Path,required=True)
    parser.add_argument("--output",type=Path,default=Path("analysis_outputs/E04_vs_E01")); args=parser.parse_args()
    e01_rows,e01=_load(args.e01); e04_rows,e04=_load(args.e04)
    if [r["sample_id"] for r in e01_rows] != [r["sample_id"] for r in e04_rows]:
        raise ValueError("E01 and E04 sample IDs/order differ")
    ranked=sorted(e04_rows,key=lambda row:float(row["support_iou"]),reverse=True)
    reps={"best":ranked[0]["sample_id"],"median":ranked[len(ranked)//2]["sample_id"],
          "worst":ranked[-1]["sample_id"]}
    result={"sample_count":len(e04_rows),"threshold_si":.001,"E01":e01,"E04":e04,
        "E04_minus_E01":{key:e04[key]-e01[key] for key in METRICS},
        "representative_e04_samples_by_support_iou":reps,
        "representative_files_note":"Each prediction directory contains susceptibility comparison PNGs and per-sample tmi_consistency magnetic maps."}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"comparison.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    lines=["# E04 versus E01 on the E01 test split","","| Metric | E01 | E04 | E04 - E01 |","|---|---:|---:|---:|"]
    lines += [f"| {key} | {e01[key]:.8g} | {e04[key]:.8g} | {e04[key]-e01[key]:.8g} |" for key in METRICS]
    lines += ["","Representative E04 samples: "+", ".join(f"{k}={v}" for k,v in reps.items())+"."]
    (args.output/"comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
