"""Complete per-sample E05 evaluation with fixed E01 definitions."""
from __future__ import annotations
import csv,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import tensorflow as tf
from cnn_inversion_3d.dataset import load_magnetic_sample,read_manifest_paths
from cnn_inversion_3d.e05_runner import build_model
from e01_magnetic.config import MagneticSurveyConfig
from evaluation.tmi_metrics import calculate_support_metrics,calculate_susceptibility_metrics,calculate_tmi_fit_metrics
from forward_modeling.forward_model import TMIForwardModel,make_tensor_grid

def _safe_ratio(a,b):return float(a/b) if b else None
def _depth_metrics(truth,prediction,threshold):
    def bounds(values):
        z=np.flatnonzero(np.any(values>=threshold,axis=(1,2)))
        return (int(z[0]),int(z[-1])) if z.size else None
    tb,pb=bounds(truth),bounds(prediction)
    true_profile=np.sum(truth,axis=(1,2));pred_profile=np.sum(prediction,axis=(1,2));z=5+10*np.arange(24)
    tc=_safe_ratio(np.sum(true_profile*z),np.sum(true_profile));pc=_safe_ratio(np.sum(pred_profile*z),np.sum(pred_profile))
    return {"true_top_depth_m":5+10*tb[0] if tb else None,"predicted_top_depth_m":5+10*pb[0] if pb else None,
        "top_depth_absolute_error_m":abs(10*(pb[0]-tb[0])) if tb and pb else None,
        "bottom_depth_absolute_error_m":abs(10*(pb[1]-tb[1])) if tb and pb else None,
        "thickness_absolute_error_m":abs(10*((pb[1]-pb[0]+1)-(tb[1]-tb[0]+1))) if tb and pb else None,
        "true_center_depth_m":tc,"predicted_center_depth_m":pc,
        "center_depth_absolute_error_m":abs(pc-tc) if pc is not None and tc is not None else None}

def _plot(sample_id,label,truth,prediction,observed,recovered,output):
    import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
    occupied=np.argwhere(truth>0);center=tuple(np.rint(occupied.mean(0)).astype(int)) if occupied.size else (12,32,32)
    z,y,x=center;sections=((truth[z],prediction[z],f"Horizontal Z={5+10*z} m"),
        (truth[:,y,:],prediction[:,y,:],f"Vertical Y={5+10*y} m"),
        (truth[:,:,x],prediction[:,:,x],f"Vertical X={5+10*x} m"))
    vmax=max(float(truth.max()),float(prediction.max()),1e-12);dmax=max(float(np.abs(prediction-truth).max()),1e-12)
    fig,axes=plt.subplots(3,3,figsize=(14,12),constrained_layout=True)
    for row,(true_slice,pred_slice,coordinate) in enumerate(sections):
        for col,(values,title,cmap,lo,hi) in enumerate(((true_slice,"True","viridis",0,vmax),
                (pred_slice,"Predicted","viridis",0,vmax),(pred_slice-true_slice,"Residual","RdBu_r",-dmax,dmax))):
            image=axes[row,col].imshow(values,aspect="auto",cmap=cmap,vmin=lo,vmax=hi)
            empty="; empty slice" if not np.any(values) and col<2 else ""
            axes[row,col].set_title(f"{title}: {coordinate}{empty}");fig.colorbar(image,ax=axes[row,col],label="SI")
    prediction_state="empty 3D prediction" if not np.any(prediction) else "nonempty 3D prediction"
    fig.suptitle(f"{label} | {sample_id} | susceptibility; common true/predicted scale; {prediction_state}")
    fig.savefig(output/"susceptibility_comparison.png",dpi=160);plt.close(fig)
    residual=recovered-observed;common=max(float(np.abs(observed).max()),float(np.abs(recovered).max()),1e-12)
    rmax=max(float(np.abs(residual).max()),1e-12);fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    for axis,(values,title,limit) in zip(axes,((observed,"Observed TMI",common),(recovered,"Predicted TMI",common),(residual,"Residual",rmax))):
        image=axis.imshow(values,origin="lower",extent=(0,800,0,800),cmap="RdBu_r",vmin=-limit,vmax=limit)
        axis.set_title(title);axis.set_xlabel("Easting (m)");axis.set_ylabel("Northing (m)");fig.colorbar(image,ax=axis,label="nT")
    fig.suptitle(f"{label} | {sample_id} | TMI; common observed/predicted scale")
    fig.savefig(output/"tmi_comparison.png",dpi=160);plt.close(fig)

def _aggregate(rows):
    result={"sample_count":len(rows)}
    for key in rows[0]:
        values=np.asarray([row[key] for row in rows if isinstance(row.get(key),(int,float)) and row[key] is not None],float)
        finite=values[np.isfinite(values)]
        if finite.size:result[key]={"mean":float(finite.mean()),"median":float(np.median(finite)),
            "q10":float(np.quantile(finite,.1)),"q90":float(np.quantile(finite,.9)),"valid_count":int(finite.size)}
    return result

def evaluate_variant(config,variant,training_output:Path,evaluation_output:Path,*,split="test",plots="all",limit=None,
                     manifest_name=None):
    checkpoint=training_output/"selected.weights.h5"
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    model=build_model(config,variant);model(tf.zeros((1,81,81,1),tf.float32),training=False);model.load_weights(checkpoint)
    manifest_name=manifest_name or f"{split}_manifest.csv"
    paths=read_manifest_paths(dataset_directory=config.dataset.resolve(),manifest_name=manifest_name)
    if limit is not None:paths=paths[:limit]
    loaded=[];vectors=[];evaluation_output.mkdir(parents=True,exist_ok=True)
    prediction_root=evaluation_output/"predictions";prediction_root.mkdir(exist_ok=True)
    for path in paths:
        tmi,truth=load_magnetic_sample(path);prediction=np.asarray(model.inversion_model(
            tf.constant(tmi[None]/config.tmi_scale_nt),training=False)[0,...,0],np.float64)
        true=np.asarray(truth[...,0],np.float64);observed=np.asarray(tmi[...,0],np.float64)
        loaded.append((path.stem,true,prediction,observed));vectors.append(prediction.transpose(2,1,0).ravel(order="F"))
        np.savez_compressed(prediction_root/f"{path.stem}_prediction.npz",tmi=observed,
            true_susceptibility=true,recovered_susceptibility=prediction)
    survey=MagneticSurveyConfig();forward=TMIForwardModel(make_tensor_grid([0,640,0,640,0,240],[10,10],10),
        survey.receiver_xyz,survey.field_strength_nt,survey.inclination_deg,survey.declination_deg,survey.azimuth_deg)
    recovered_maps=forward.predict_many(np.stack(vectors)).reshape(len(loaded),81,81);rows=[]
    for index,((sample_id,truth,prediction,observed),recovered) in enumerate(zip(loaded,recovered_maps),1):
        true_support=truth>=config.support_threshold_si;pred_support=prediction>=config.support_threshold_si
        support=calculate_support_metrics(true_support,pred_support);sus=calculate_susceptibility_metrics(truth,prediction,config.support_threshold_si)
        tmi=calculate_tmi_fit_metrics(observed,recovered);body=truth>0;background=~body;body_error=prediction[body]-truth[body]
        observed_rms=float(np.sqrt(np.mean(observed**2)));predicted_rms=float(np.sqrt(np.mean(recovered**2)))
        observed_peak=float(np.max(np.abs(observed)));predicted_peak=float(np.max(np.abs(recovered)))
        row={"sample_id":sample_id,**{key:sus[key] for key in ("susceptibility_mae_si","susceptibility_rmse_si","susceptibility_relative_l2")},
            **support,"true_body_susceptibility_mae_si":float(np.mean(np.abs(body_error))) if body.any() else 0.,
            "true_body_signed_bias_si":float(np.mean(body_error)) if body.any() else 0.,
            "background_mae_si":float(np.mean(np.abs(prediction[background]))) if background.any() else 0.,
            "total_background_susceptibility_si_cells":float(np.sum(prediction[background])),**_depth_metrics(truth,prediction,config.support_threshold_si),
            **tmi,"observed_tmi_rms_nt":observed_rms,"predicted_tmi_rms_nt":predicted_rms,
            "predicted_to_observed_tmi_rms_ratio":_safe_ratio(predicted_rms,observed_rms),
            "observed_tmi_peak_nt":observed_peak,"predicted_tmi_peak_nt":predicted_peak,
            "predicted_to_observed_tmi_peak_ratio":_safe_ratio(predicted_peak,observed_peak)};rows.append(row)
        if plots=="all":
            sample_output=evaluation_output/"plots"/sample_id;sample_output.mkdir(parents=True,exist_ok=True)
            _plot(sample_id,variant.label,truth,prediction,observed,recovered,sample_output)
        print(f"{variant.label} {split}: {index}/{len(loaded)}",flush=True)
    with (evaluation_output/"per_sample_metrics.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={"experiment":variant.to_dict(),"split":split,"evaluation_manifest":str(manifest_name),
        "support_threshold_si":config.support_threshold_si,
        "sample_ids":[row["sample_id"] for row in rows],"aggregate":_aggregate(rows),
        "empty_support_policy":"true-body MAE/bias=0 for empty truth; both-empty support=1; one-empty support=0; undefined ratios=null",
        "timestamp_utc":datetime.now(timezone.utc).isoformat()}
    (evaluation_output/"aggregate_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    return rows,summary
