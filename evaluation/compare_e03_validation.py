"""Compare E03 ablations using common validation metrics, never total loss."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path

METRICS=(
    "geological_iou","geological_dice","geological_precision","geological_recall",
    "geological_predicted_to_true_volume_ratio","true_body_susceptibility_mae_si",
    "true_body_susceptibility_bias_si","predicted_to_true_body_mean_ratio",
    "background_susceptibility_mae_si","integrated_absolute_background_leakage_si_cells",
    "tmi_mae_nt","tmi_rmse_nt","tmi_absolute_reference_mse",
    "tmi_floored_relative_mse","tmi_correlation","predicted_to_observed_tmi_rms_ratio",
    "prediction_fraction_near_saturation","zero_susceptibility_mae_si","zero_tmi_mae_nt",
)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audits",nargs="+",type=Path,help="Validation audit output directories")
    parser.add_argument("--output",type=Path,default=Path("analysis_outputs/E03_validation_comparison"))
    args=parser.parse_args(); rows=[]
    for directory in args.audits:
        summary=json.loads((directory/"aggregate_summary.json").read_text(encoding="utf-8"))["overall"]
        metadata=json.loads((directory/"evaluation_metadata.json").read_text(encoding="utf-8"))
        if metadata.get("evaluation_split")!="validation":
            raise ValueError(f"{directory} is not a validation audit")
        row={"experiment":directory.name,"checkpoint_sha256":metadata["checkpoint_sha256"]}
        row.update({metric:summary.get(metric,{}).get("mean") for metric in METRICS})
        rows.append(row)
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/"common_validation_metrics.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    columns=("experiment","geological_iou","geological_dice","true_body_susceptibility_mae_si",
        "background_susceptibility_mae_si","tmi_rmse_nt","tmi_correlation","predicted_to_observed_tmi_rms_ratio")
    lines=["# E03 common validation metrics","",
        "Total validation loss is intentionally excluded because the ablations use different objectives.","",
        "| "+" | ".join(columns)+" |","|"+"---|"*len(columns)]
    for row in rows:
        lines.append("| "+" | ".join(str(row[key]) for key in columns)+" |")
    (args.output/"comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Wrote {args.output/'common_validation_metrics.csv'} and comparison.md")

if __name__=="__main__": main()
