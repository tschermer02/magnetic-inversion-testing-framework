"""Plot every audited E03 prediction and its freshly forward-modeled TMI."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cnn_inversion_3d.evaluate import save_susceptibility_comparison
from e01_magnetic.config import MagneticSurveyConfig
from evaluation.plot_e01_3d import build_figure
from evaluation.tmi_metrics import calculate_tmi_fit_metrics
from forward_modeling.forward_model import TMIForwardModel,make_tensor_grid

def _magnetic_figure(observed,recovered,path,survey):
    residual=recovered-observed
    panels=((observed,"True / Input TMI"),(recovered,"Forward-modeled predicted TMI"),
            (residual,"Predicted minus true TMI"))
    figure,axes=plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
    for axis,(values,title) in zip(axes,panels):
        limit=max(float(np.max(np.abs(values))),1e-12)
        image=axis.imshow(values,origin="lower",cmap="RdBu_r",vmin=-limit,vmax=limit,
            extent=(survey.receiver_x_min_m,survey.receiver_x_max_m,
                    survey.receiver_y_min_m,survey.receiver_y_max_m))
        axis.set_title(f"{title}\nrange {values.min():.3g} to {values.max():.3g} nT")
        axis.set_xlabel("Easting (m)"); axis.set_ylabel("Northing (m)")
        figure.colorbar(image,ax=axis,shrink=.82,label="TMI (nT)")
    figure.suptitle("E03 magnetic forward-consistency analysis",fontweight="bold")
    figure.savefig(path,dpi=180); plt.close(figure)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit",type=Path,default=Path("prediction_outputs/E03_combined_audit"))
    parser.add_argument("--output",type=Path,default=Path("prediction_outputs/E03_combined_plots"))
    parser.add_argument("--threshold-si",type=float,default=5e-5)
    parser.add_argument("--limit",type=int,help="Plot only the first N samples (useful for a smoke check).")
    parser.add_argument("--plotly",action="store_true",help="Also write an interactive 3-D HTML for every sample.")
    args=parser.parse_args(); source=args.audit/"regenerated_predictions"
    files=sorted(source.glob("sample_*_prediction.npz"))
    if not files: raise FileNotFoundError(f"No regenerated predictions found in {source}")
    if args.limit is not None:
        if args.limit < 1: raise ValueError("--limit must be positive")
        files=files[:args.limit]
    args.output.mkdir(parents=True,exist_ok=True)
    loaded=[]; model_vectors=[]
    for path in files:
        with np.load(path) as saved:
            observed=np.asarray(saved["tmi"],np.float64)
            truth=np.asarray(saved["true_susceptibility"],np.float64)
            predicted=np.asarray(saved["recovered_susceptibility"],np.float64)
        loaded.append((path.stem.removesuffix("_prediction"),observed,truth,predicted))
        model_vectors.append(predicted.transpose(2,1,0).ravel(order="F"))
    survey=MagneticSurveyConfig(); grid=make_tensor_grid([0,640,0,640,0,240],[10,10],10)
    forward=TMIForwardModel(grid,survey.receiver_xyz,survey.field_strength_nt,
        survey.inclination_deg,survey.declination_deg,survey.azimuth_deg)
    print(f"Forward modeling {len(loaded)} E03 predictions together...",flush=True)
    responses=forward.predict_many(np.stack(model_vectors)).reshape(len(loaded),81,81)
    rows=[]
    for index,((sample_id,observed,truth,predicted),recovered) in enumerate(zip(loaded,responses),1):
        sample_dir=args.output/sample_id; sample_dir.mkdir(exist_ok=True)
        save_susceptibility_comparison(truth,predicted,sample_dir/"susceptibility_comparison.png")
        _magnetic_figure(observed,recovered,sample_dir/"magnetic_comparison.png",survey)
        np.save(sample_dir/"true_tmi.npy",observed.astype(np.float32))
        np.save(sample_dir/"recovered_tmi.npy",recovered.astype(np.float32))
        np.save(sample_dir/"tmi_residual.npy",(recovered-observed).astype(np.float32))
        metrics={"sample_id":sample_id,**calculate_tmi_fit_metrics(observed,recovered)}
        (sample_dir/"tmi_consistency_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
        if args.plotly:
            build_figure(sample_id,truth,predicted,threshold=args.threshold_si,
                experiment="E03").write_html(sample_dir/"susceptibility_3d.html",
                    include_plotlyjs="directory",full_html=True)
        rows.append(metrics); print(f"Plotted {index}/{len(loaded)}: {sample_id}",flush=True)
    with (args.output/"tmi_consistency_metrics.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metadata={"source_audit":str(args.audit.resolve()),"sample_count":len(rows),
        "fresh_forward_model":True,"interactive_plotly":args.plotly,
        "susceptibility_threshold_si":args.threshold_si}
    (args.output/"plot_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(f"All E03 plots written to {args.output.resolve()}")

if __name__=="__main__": main()
