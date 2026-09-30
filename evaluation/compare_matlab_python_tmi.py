"""Compare original MATLAB getPredMag against an independent Python forward run."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import h5py
import numpy as np
from e01_magnetic.config import MagneticSurveyConfig
from forward_modeling.forward_model import TMIForwardModel,make_tensor_grid

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset",type=Path,default=Path("datasets/E01_soft_tversky_full"))
    parser.add_argument("--matlab-output",type=Path,default=Path("analysis_outputs/matlab_python_tmi/matlab_outputs"))
    parser.add_argument("--output",type=Path,default=Path("analysis_outputs/matlab_python_tmi/comparison"))
    parser.add_argument("--limit",type=int,default=None,help="Compare only the first N manifest samples (smoke testing).")
    parser.add_argument("--atol-nt",type=float,default=1e-5);parser.add_argument("--rtol",type=float,default=1e-6)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (args.dataset/"test_manifest.csv").open(newline="",encoding="utf-8") as stream:manifest=list(csv.DictReader(stream))
    if args.limit is not None:
        if args.limit < 1: raise ValueError("--limit must be at least 1")
        manifest=manifest[:args.limit]
    models=[];stored=[]
    for row in manifest:
        with np.load(args.dataset/row["relative_path"]) as saved:
            susceptibility=np.asarray(saved["susceptibility"],np.float64);stored.append(np.asarray(saved["tmi"],np.float64))
        models.append(susceptibility.transpose(2,1,0).ravel(order="F"))
    survey=MagneticSurveyConfig();operator=TMIForwardModel(make_tensor_grid([0,640,0,640,0,240],[10,10],10),
        survey.receiver_xyz,survey.field_strength_nt,survey.inclination_deg,survey.declination_deg,survey.azimuth_deg)
    print(f"Running independent Python forward model for {len(models)} samples...",flush=True)
    python=operator.predict_many(np.stack(models));rows=[];residual_maps=[]
    for index,(row,python_vector,stored_map) in enumerate(zip(manifest,python,stored),1):
        path=args.matlab_output/f"{row['sample_id']}.h5"
        if not path.is_file():raise FileNotFoundError(path)
        with h5py.File(path,"r") as saved:
            matlab=np.asarray(saved["matlab_tmi_xfast_nt"],np.float64).reshape(-1)
        if matlab.size!=python_vector.size:raise ValueError(f"Receiver count differs for {row['sample_id']}")
        residual=python_vector-matlab;python_map=python_vector.reshape(81,81);matlab_map=matlab.reshape(81,81)
        stored_residual=python_map-stored_map
        norm=np.linalg.norm(matlab)
        metrics={"sample_id":row["sample_id"],"mae_nt":float(np.mean(np.abs(residual))),
            "rmse_nt":float(np.sqrt(np.mean(residual**2))),"max_abs_error_nt":float(np.max(np.abs(residual))),
            "relative_l2":float(np.linalg.norm(residual)/norm) if norm else None,
            "correlation":float(np.corrcoef(python_vector,matlab)[0,1]) if np.std(matlab) and np.std(python_vector) else None,
            "python_vs_stored_rmse_nt":float(np.sqrt(np.mean(stored_residual**2))),
            "passed":bool(np.allclose(python_vector,matlab,rtol=args.rtol,atol=args.atol_nt))};rows.append(metrics)
        residual_maps.append((metrics["rmse_nt"],row["sample_id"],matlab_map,python_map,residual.reshape(81,81)))
        print(f"Compared {index}/{len(manifest)}: {row['sample_id']}",flush=True)
    with (args.output/"per_sample_metrics.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    numeric=("mae_nt","rmse_nt","max_abs_error_nt","relative_l2","correlation","python_vs_stored_rmse_nt")
    summary={"sample_count":len(rows),"passed_count":sum(row["passed"] for row in rows),
        "all_passed":all(row["passed"] for row in rows),"atol_nt":args.atol_nt,"rtol":args.rtol,
        "metrics":{key:{"mean":float(np.mean([row[key] for row in rows if row[key] is not None])),
            "maximum":float(np.max([row[key] for row in rows if row[key] is not None]))} for key in numeric}}
    (args.output/"aggregate_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    for label,item in (("best",min(residual_maps)),("worst",max(residual_maps))):
        rmse,sample_id,matlab_map,python_map,residual=item;common=max(np.abs(matlab_map).max(),np.abs(python_map).max(),1e-12)
        dmax=max(np.abs(residual).max(),1e-12);fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
        for axis,(values,title,limit) in zip(axes,((matlab_map,"MATLAB TMI",common),(python_map,"Python TMI",common),(residual,"Python - MATLAB",dmax))):
            image=axis.imshow(values,origin="lower",extent=(0,800,0,800),cmap="RdBu_r",vmin=-limit,vmax=limit)
            axis.set_title(title);axis.set_xlabel("Easting (m)");axis.set_ylabel("Northing (m)");fig.colorbar(image,ax=axis,label="nT")
        fig.suptitle(f"{label.title()} agreement: {sample_id}; RMSE={rmse:.4g} nT")
        fig.savefig(args.output/f"{label}_{sample_id}_comparison.png",dpi=170);plt.close(fig)
    print(json.dumps(summary,indent=2));print(f"Comparison written to {args.output.resolve()}")
if __name__=="__main__":main()
