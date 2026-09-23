"""Validate E04 TensorFlow physics against stored E01 TMI on fixed samples."""
from __future__ import annotations
import argparse,csv,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from cnn_inversion_3d.e04_forward import E04TMIForward
from evaluation.tmi_metrics import calculate_tmi_fit_metrics

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=Path("datasets/E01_soft_tversky_full"))
    parser.add_argument("--output",type=Path,default=Path("analysis_outputs/E04_forward_operator_validation"))
    parser.add_argument("--count",type=int,default=5); args=parser.parse_args()
    with (args.dataset/"train_manifest.csv").open(newline="",encoding="utf-8") as stream:
        manifest=list(csv.DictReader(stream))
    indices=np.linspace(0,len(manifest)-1,args.count,dtype=int); operator=E04TMIForward(); rows=[]
    for index in indices:
        row=manifest[index]; path=args.dataset/row["relative_path"]
        with np.load(path) as saved:
            truth=np.asarray(saved["susceptibility"],np.float32)
            observed=np.asarray(saved["tmi"],np.float64)
        predicted=operator(truth[None,...,None]).numpy()[0,...,0].astype(np.float64)
        residual=predicted-observed; metrics=calculate_tmi_fit_metrics(observed,predicted)
        rows.append({"sample_id":row["sample_id"],"rmse_nt":metrics["tmi_rmse_nt"],
            "mae_nt":metrics["tmi_mae_nt"],"maximum_absolute_residual_nt":float(np.max(np.abs(residual))),
            "correlation":metrics["tmi_correlation"],"relative_l2":metrics["tmi_relative_l2"]})
    result={"dataset":str(args.dataset.resolve()),"split":"training only","sample_count":len(rows),
        "tolerance_note":"Float32 TensorFlow versus float64 NumPy generation; small round-off residuals expected.",
        "samples":rows,"maximum_rmse_nt":max(row["rmse_nt"] for row in rows),
        "timestamp_utc":datetime.now(timezone.utc).isoformat()}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"forward_validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))

if __name__=="__main__": main()
